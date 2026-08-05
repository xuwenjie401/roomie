#include <chrono>
#include <memory>
#include <utility>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "roomie/pipeline/pipeline.hpp"
#include "roomie/pipeline/pipeline_config.hpp"
#include "roomie/query/scene_mutation_json_adapter.hpp"
#include "roomie/query/scene_query_json_adapter.hpp"
#include "roomie/srv/mutate_scene.hpp"
#include "roomie/srv/query_scene.hpp"
#include "roomie/utils/run_logger.hpp"

namespace roomie {

class RoomiePipelineNode : public rclcpp::Node {
 public:
  RoomiePipelineNode() : Node("roomie_pipeline_node") {
    PipelineConfig config = PipelineConfig::declareAndLoad(*this);
    pipeline_ = std::make_unique<RoomiePipeline>(*this, std::move(config));
    pipeline_->start();
    query_adapter_ = std::make_unique<SceneQueryJsonAdapter>(
        pipeline_->sceneQueryGateway());
    mutation_adapter_ = std::make_unique<SceneMutationJsonAdapter>(
        [this](ApplyHumanAnnotationCommand command,
               std::chrono::milliseconds timeout) {
          return pipeline_->mutateScene(std::move(command), timeout);
        });
    query_scene_service_ = create_service<roomie::srv::QueryScene>(
        "/roomie/query_scene",
        [this](
            const std::shared_ptr<roomie::srv::QueryScene::Request> request,
            std::shared_ptr<roomie::srv::QueryScene::Response> response) {
          const auto started = std::chrono::steady_clock::now();
          const SceneQueryJsonResponse result =
              query_adapter_->dispatch(request->request_json);
          response->success = result.success;
          response->response_json = result.response_json;
          response->error = result.error;
          const double latency_ms =
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - started)
                  .count();
          RunLogger::logGlobal(
              "scene_query",
              "request_completed success=" +
                  std::string(result.success ? "true" : "false") +
                  " calls=" + std::to_string(result.call_count) +
                  " scene_revision=" +
                  std::to_string(result.scene_revision) +
                  " durable_scene_revision=" +
                  std::to_string(result.durable_scene_revision) +
                  " index_generation=" +
                  std::to_string(result.index_generation) +
                  " request_bytes=" +
                  std::to_string(request->request_json.size()) +
                  " response_bytes=" +
                  std::to_string(result.response_json.size()) +
                  " latency_ms=" + std::to_string(latency_ms));
        });
    mutate_scene_service_ = create_service<roomie::srv::MutateScene>(
        "/roomie/mutate_scene",
        [this](
            const std::shared_ptr<roomie::srv::MutateScene::Request> request,
            std::shared_ptr<roomie::srv::MutateScene::Response> response) {
          const auto started = std::chrono::steady_clock::now();
          const SceneMutationJsonResponse result =
              mutation_adapter_->dispatch(request->request_json);
          response->success = result.success;
          response->response_json = result.response_json;
          response->error = result.error;
          const double latency_ms =
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - started)
                  .count();
          RunLogger::logGlobal(
              "scene_mutation",
              "request_completed success=" +
                  std::string(result.success ? "true" : "false") +
                  " latest_scene_revision=" +
                  std::to_string(result.latest_scene_revision) +
                  " durable_scene_revision=" +
                  std::to_string(result.durable_scene_revision) +
                  " request_bytes=" +
                  std::to_string(request->request_json.size()) +
                  " response_bytes=" +
                  std::to_string(result.response_json.size()) +
                  " latency_ms=" + std::to_string(latency_ms));
        });
    RCLCPP_INFO(get_logger(),
                "roomie pipeline started; live services: /roomie/query_scene, "
                "/roomie/mutate_scene");
  }

  ~RoomiePipelineNode() override {
    if (pipeline_) {
      pipeline_->stop();
    }
  }

 private:
  std::unique_ptr<RoomiePipeline> pipeline_;
  std::unique_ptr<SceneQueryJsonAdapter> query_adapter_;
  std::unique_ptr<SceneMutationJsonAdapter> mutation_adapter_;
  rclcpp::Service<roomie::srv::QueryScene>::SharedPtr query_scene_service_;
  rclcpp::Service<roomie::srv::MutateScene>::SharedPtr mutate_scene_service_;
};

}  // namespace roomie

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<roomie::RoomiePipelineNode>();
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
