#include "roomie/scene/geometry_worker.hpp"

#include <type_traits>
#include <stdexcept>
#include <utility>

#include "roomie/utils/run_logger.hpp"

namespace roomie {

GeometryWorkerThread::GeometryWorkerThread(
    GeometrySchedulerConfig config,
    SceneProvider scene_provider,
    CommandSink command_sink,
    std::size_t delta_journal_capacity)
    : WorkerThread("geometry_worker"),
      scheduler_(std::move(config)),
      delta_journal_(delta_journal_capacity),
      scene_provider_(std::move(scene_provider)),
      command_sink_(std::move(command_sink)) {
  if (!scene_provider_ || !command_sink_) {
    throw std::invalid_argument(
        "GeometryWorkerThread requires scene provider and command sink");
  }
}

void GeometryWorkerThread::onMapCommit(const MapCommit& commit) {
  if (!commit.snapshot || commit.surface != commit.snapshot->surfaceStamp()) {
    return;
  }

  const SurfaceSnapshotPtr previous = std::atomic_load(&latest_surface_);
  if (previous &&
      previous->surfaceStamp().map_epoch != commit.surface.map_epoch) {
    delta_journal_.clear();
  }

  // The surface watermark is enqueued before any work derived from it. A
  // later geometry result therefore cannot pass the reducer with an obsolete
  // preflight surface merely because the map changed between threads.
  if (!command_sink_(SceneCommand{AdvanceSurfaceCommand{commit.surface}})) {
    ++command_queue_rejected_;
    return;
  }
  delta_journal_.append(commit.delta);
  std::atomic_store(&latest_surface_, commit.snapshot);
  ++map_commits_;

  const SceneSnapshot scene = scene_provider_();
  if (scheduler_.indexedObjectCount() < scene.objects().size()) {
    for (const auto& [object_id, object] : scene.objects()) {
      if (!object) {
        continue;
      }
      if (scheduler_.onObjectCreated(
              ObjectCreated{scene.revision(), object_id},
              scene,
              commit.snapshot)) {
        ++scheduled_;
      }
    }
  }
  const GeometryScheduleBatch batch =
      scheduler_.onMapDelta(commit.delta, commit.snapshot);
  scheduled_.fetch_add(batch.accepted);
  RunLogger::logGlobal(
      "geometry_worker",
      "scheduled trigger=map_delta map_revision=" +
          std::to_string(commit.map.map_revision) +
          " surface_revision=" +
          std::to_string(commit.surface.surface_revision) +
          " requested=" + std::to_string(batch.requested) +
          " accepted=" + std::to_string(batch.accepted) +
          " superseded=" + std::to_string(batch.replaced) +
          " conservative_full_scan=" +
          std::string(batch.conservative_full_scan ? "true" : "false"));
}

void GeometryWorkerThread::onSceneCommit(const SceneApplyResult& result) {
  if (!result.accepted() || result.events.empty()) {
    return;
  }
  const SurfaceSnapshotPtr surface = std::atomic_load(&latest_surface_);
  for (const SceneEvent& event : result.events) {
    std::visit(
        [&](const auto& typed_event) {
          using Event = std::decay_t<decltype(typed_event)>;
          if constexpr (std::is_same_v<Event, ObjectCreated>) {
            if (scheduler_.onObjectCreated(
                    typed_event, result.snapshot, surface)) {
              ++scheduled_;
              RunLogger::logGlobal(
                  "geometry_worker",
                  "scheduled trigger=object_created object_id=" +
                      std::to_string(typed_event.object_id) +
                      " scene_revision=" +
                      std::to_string(result.revision));
            }
          } else if constexpr (std::is_same_v<Event, ObbChanged>) {
            if (scheduler_.onObbChanged(
                    typed_event, result.snapshot, surface)) {
              ++scheduled_;
              RunLogger::logGlobal(
                  "geometry_worker",
                  "scheduled trigger=obb_changed object_id=" +
                      std::to_string(typed_event.object_id) +
                      " scene_revision=" +
                      std::to_string(result.revision));
            }
          } else if constexpr (std::is_same_v<Event, ObjectMerged>) {
            scheduler_.onObjectMerged(typed_event);
          } else if constexpr (std::is_same_v<Event, ObjectTombstoned>) {
            scheduler_.onObjectTombstoned(typed_event);
          }
        },
        event);
  }
}

GeometryWorkerStats GeometryWorkerThread::stats() const {
  GeometryWorkerStats result;
  const ChannelStats scheduler_stats = scheduler_.channelStats();
  result.map_commits = map_commits_.load();
  result.scheduled = scheduled_.load();
  result.evaluated = evaluated_.load();
  result.commands_enqueued = commands_enqueued_.load();
  result.superseded = superseded_.load();
  result.pending_superseded = scheduler_stats.replaced;
  result.backpressure_waits = scheduler_stats.producer_wait_count;
  result.retry_attempts = retry_attempts_.load();
  result.retries_scheduled = retries_scheduled_.load();
  result.command_queue_rejected = command_queue_rejected_.load();
  return result;
}

bool GeometryWorkerThread::queueRetry(SceneObjectId object_id,
                                      GeometryTrigger trigger) {
  ++retry_attempts_;
  const SceneSnapshot scene = scene_provider_();
  const SurfaceSnapshotPtr surface = std::atomic_load(&latest_surface_);
  std::optional<GeometryInput> input = scheduler_.currentInput(
      object_id, scene, surface, trigger);
  if (!input) {
    return false;
  }
  pending_retries_.insert_or_assign(object_id, std::move(*input));
  ++scheduled_;
  ++retries_scheduled_;
  RunLogger::logGlobal(
      "geometry_worker",
      "retry_queued object_id=" + std::to_string(object_id) +
          " trigger=" + std::to_string(static_cast<int>(trigger)));
  return true;
}

void GeometryWorkerThread::run() {
  GeometryInput input;
  bool previous_was_retry = false;
  for (;;) {
    if (stopRequested()) {
      pending_retries_.clear();
    }

    bool have_input = false;
    if (!stopRequested() && !pending_retries_.empty() &&
        !previous_was_retry) {
      auto retry = pending_retries_.begin();
      input = std::move(retry->second);
      pending_retries_.erase(retry);
      previous_was_retry = true;
      have_input = true;
    } else if (scheduler_.tryPop(&input)) {
      previous_was_retry = false;
      have_input = true;
    } else if (!stopRequested() && !pending_retries_.empty()) {
      auto retry = pending_retries_.begin();
      input = std::move(retry->second);
      pending_retries_.erase(retry);
      previous_was_retry = true;
      have_input = true;
    } else if (scheduler_.waitPop(&input)) {
      previous_was_retry = false;
      have_input = true;
    }
    if (!have_input) {
      break;
    }

    const GeometryResult result = evaluateGeometry(input);
    ++evaluated_;

    const SceneSnapshot scene = scene_provider_();
    const SurfaceSnapshotPtr surface = std::atomic_load(&latest_surface_);
    if (!surface) {
      ++superseded_;
      RunLogger::logGlobal(
          "geometry_worker",
          "result_superseded object_id=" +
              std::to_string(result.dependency.object.object_id) +
              " reason=no_current_surface");
      continue;
    }
    const GeometryCasCheck check = checkGeometryCas(
        result, scene, surface->surfaceStamp(), delta_journal_);
    const std::optional<ApplyGeometryResultCommand> command =
        makeGeometryApplyCommand(result, surface->surfaceStamp(), check);
    if (!command) {
      ++superseded_;
      RunLogger::logGlobal(
          "geometry_worker",
          "result_superseded object_id=" +
              std::to_string(result.dependency.object.object_id) +
              " reason=cas_decision decision=" +
              std::to_string(static_cast<int>(check.decision)) +
              " source_surface_revision=" +
              std::to_string(
                  result.dependency.surface.surface_revision) +
              " current_surface_revision=" +
              std::to_string(surface->surfaceStamp().surface_revision));
      if (!stopRequested()) {
        queueRetry(result.dependency.object.object_id, input.trigger);
      }
      continue;
    }
    if (command_sink_(SceneCommand{*command})) {
      ++commands_enqueued_;
      RunLogger::logGlobal(
          "geometry_worker",
          "result_accepted object_id=" +
              std::to_string(result.dependency.object.object_id) +
              " decision=" +
              std::to_string(static_cast<int>(check.decision)) +
              " surface_revision=" +
              std::to_string(surface->surfaceStamp().surface_revision));
    } else {
      ++command_queue_rejected_;
      RunLogger::logGlobal(
          "geometry_worker",
          "result_rejected object_id=" +
              std::to_string(result.dependency.object.object_id) +
              " reason=scene_command_queue");
    }
  }
}

void GeometryWorkerThread::onStopRequested() { scheduler_.stop(); }

}  // namespace roomie
