cmake_minimum_required(VERSION 3.16)
project(roomie)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_library(
  roomie_pipeline
  src/detection_bridge_thread.cpp
  src/instance_map_thread.cpp
  src/map_thread.cpp
  src/pipeline.cpp
  src/pipeline_config.cpp
  src/publisher_persistence_thread.cpp
  src/python_inference_backend.cpp
  src/ros_io_thread.cpp
  src/worker_thread.cpp
)
target_include_directories(
  roomie_pipeline PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                         "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(roomie_pipeline Eigen3::Eigen)
ament_target_dependencies(
  roomie_pipeline
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
)

add_executable(roomie_pipeline_node src/pipeline_node.cpp)
target_link_libraries(roomie_pipeline_node roomie_pipeline)
ament_target_dependencies(roomie_pipeline_node rclcpp)

install(TARGETS roomie_pipeline
        EXPORT roomie-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
install(TARGETS roomie_pipeline_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_export_targets(roomie-targets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 rclcpp sensor_msgs std_msgs visualization_msgs)
ament_package()
