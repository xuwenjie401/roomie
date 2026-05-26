#include <memory>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "roomie/pipeline/pipeline.hpp"
#include "roomie/pipeline/pipeline_config.hpp"

namespace roomie {

class RoomiePipelineNode : public rclcpp::Node {
 public:
  RoomiePipelineNode() : Node("roomie_pipeline_node") {
    PipelineConfig config = PipelineConfig::declareAndLoad(*this);
    pipeline_ = std::make_unique<RoomiePipeline>(*this, std::move(config));
    pipeline_->start();
    RCLCPP_INFO(get_logger(), "roomie pipeline skeleton started");
  }

  ~RoomiePipelineNode() override {
    if (pipeline_) {
      pipeline_->stop();
    }
  }

 private:
  std::unique_ptr<RoomiePipeline> pipeline_;
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
