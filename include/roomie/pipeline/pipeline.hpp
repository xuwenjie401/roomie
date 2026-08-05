#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "roomie/artifacts/artifact_intent_builder.hpp"
#include "roomie/artifacts/artifact_runtime_actor.hpp"
#include "roomie/artifacts/embedding_runtime_actor.hpp"
#include "roomie/artifacts/embedding_task_scheduler.hpp"
#include "roomie/artifacts/online_snapshot_worker.hpp"
#include "roomie/artifacts/semantic_index.hpp"
#include "roomie/pipeline/detection_bridge_thread.hpp"
#include "roomie/pipeline/instance_map_thread.hpp"
#include "roomie/pipeline/map_thread.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/pipeline/publisher_persistence_thread.hpp"
#include "roomie/pipeline/python_inference_backend.hpp"
#include "roomie/pipeline/ros_io_thread.hpp"
#include "roomie/pipeline/thread_safe_queue.hpp"
#include "roomie/query/scene_query_gateway.hpp"
#include "roomie/query/scene_mutation_json_adapter.hpp"
#include "roomie/query/semantic_search_provider.hpp"
#include "roomie/scene/geometry_worker.hpp"
#include "roomie/scene/persistence_actor.hpp"
#include "roomie/utils/run_logger.hpp"

namespace roomie {

struct RoomiePipelineRuntimeDependencies {
  std::shared_ptr<DamWorker> dam_worker;
  std::shared_ptr<EmbeddingEncoder> embedding_encoder;
  SemanticQueryVectorEncoder query_encoder;
};

class RoomiePipeline {
 public:
  RoomiePipeline(
      rclcpp::Node& node,
      PipelineConfig config,
      RoomiePipelineRuntimeDependencies runtime_dependencies = {});
  ~RoomiePipeline();

  void start();
  void stop();

  // Live queries pin the reducer snapshot and any referenced online snapshot
  // assets through the same long-lived gateway instance.
  const SceneQueryGateway& sceneQueryGateway() const;

  // Submits an externally-authored patch through the reducer's serialized
  // command queue and, when persistence is enabled, uses the same deadline to
  // wait for the committed revision's durable watermark.
  SceneMutationSubmitResult mutateScene(
      ApplyHumanAnnotationCommand command,
      std::chrono::milliseconds timeout =
          SceneMutationJsonAdapter::kDefaultTimeout);

 private:
  void scheduleMissingEmbeddingJobs(std::vector<EmbeddingJob> jobs,
                                    const SceneSnapshot& snapshot);

  PipelineConfig config_;
  std::shared_ptr<RunLogger> run_logger_;
  ThreadSafeQueue<FrameBundlePtr> mapping_queue_;
  ThreadSafeQueue<FrameBundlePtr> detection_queue_;
  ThreadSafeQueue<InferenceResponse> inference_response_queue_;

  RosIoThread ros_io_thread_;
  MapThread map_thread_;
  PythonInferenceBackend python_backend_;
  InstanceMapThread instance_map_thread_;
  GeometryWorkerThread geometry_worker_thread_;
  std::shared_ptr<AssetStore> asset_store_;
  std::shared_ptr<SnapshotBank> snapshot_bank_;
  std::shared_ptr<const SnapshotAssetProvider> snapshot_asset_provider_;
  RoomiePipelineRuntimeDependencies runtime_dependencies_;
  std::shared_ptr<VersionedSemanticIndex> semantic_index_;
  std::shared_ptr<const SearchProvider> semantic_search_provider_;
  std::unique_ptr<SceneQueryGateway> scene_query_gateway_;
  std::unique_ptr<OnlineSnapshotWorker> online_snapshot_worker_;
  std::unique_ptr<ArtifactIntentBuilder> artifact_intent_builder_;
  std::unique_ptr<PersistenceActor> persistence_actor_;
  std::unique_ptr<ArtifactScheduler> artifact_scheduler_;
  std::unique_ptr<ArtifactRuntimeActor> artifact_runtime_actor_;
  std::unique_ptr<EmbeddingTaskScheduler> embedding_task_scheduler_;
  std::unique_ptr<EmbeddingRuntimeActor> embedding_runtime_actor_;
  mutable std::mutex pending_embedding_jobs_mutex_;
  std::map<std::string, EmbeddingJob> pending_embedding_jobs_;
  // External mutations are serialized across the persistence-admission check
  // and reducer acknowledgement. This closes the check/commit race at the
  // hard undurable-revision boundary for concurrent service callbacks.
  mutable std::mutex scene_mutation_mutex_;
  DetectionBridgeThread detection_bridge_thread_;
  PublisherPersistenceThread publisher_persistence_thread_;

  bool started_ = false;
};

}  // namespace roomie
