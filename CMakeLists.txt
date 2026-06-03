cmake_minimum_required(VERSION 3.16)
project(roomie)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(ROOMIE_ENABLE_NVBLOX "Build the nvblox-backed map backend" OFF)
set(ROOMIE_NVBLOX_SOURCE_DIR
    "/home/agxi/RealityLab/map_ws/src/nvblox"
    CACHE PATH
    "Path to the nvblox source tree used when nvblox is not installed")
set(ROOMIE_NVBLOX_LIBRARY_DIR
    "/home/agxi/RealityLab/map_ws/install/nvblox_ros/lib"
    CACHE PATH
    "Runtime library directory for the local nvblox install")

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(
  roomie_pipeline
  src/cpu_point_map_backend.cpp
  src/detection_bridge_thread.cpp
  src/dsg/object_graph.cpp
  src/dsg/object_graph_io.cpp
  src/dsg/object_snapshot_remaker.cpp
  src/image_utils.cpp
  src/instance_map_thread.cpp
  src/map_backend.cpp
  src/map_thread.cpp
  src/pipeline.cpp
  src/pipeline_config.cpp
  src/publisher_persistence_thread.cpp
  src/python_inference_backend.cpp
  src/ros_io_thread.cpp
  src/utils/run_logger.cpp
  src/utils/visualization_utils.cpp
  src/worker_thread.cpp
)
target_include_directories(
  roomie_pipeline PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                         "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(roomie_pipeline Eigen3::Eigen nlohmann_json::nlohmann_json ${OpenCV_LIBS})

if(ROOMIE_ENABLE_NVBLOX)
  find_package(CUDAToolkit REQUIRED)
  find_package(gflags REQUIRED)
  find_package(glog REQUIRED)
  find_package(nvblox QUIET)
  if(NOT nvblox_FOUND)
    if(NOT EXISTS "${ROOMIE_NVBLOX_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR "ROOMIE_ENABLE_NVBLOX is ON but nvblox was not found and "
                          "ROOMIE_NVBLOX_SOURCE_DIR is invalid: ${ROOMIE_NVBLOX_SOURCE_DIR}")
    endif()
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTORCH_WRAPPER OFF CACHE BOOL "" FORCE)
    set(BUILD_RENDERER OFF CACHE BOOL "" FORCE)
    add_subdirectory("${ROOMIE_NVBLOX_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/nvblox_core")
  endif()

  if(TARGET nvblox::nvblox_lib)
    set(ROOMIE_NVBLOX_TARGET nvblox::nvblox_lib)
  elseif(TARGET nvblox_lib)
    set(ROOMIE_NVBLOX_TARGET nvblox_lib)
  else()
    message(FATAL_ERROR "Could not find nvblox_lib target")
  endif()

  target_sources(roomie_pipeline PRIVATE src/nvblox_map_backend.cpp)
  target_compile_definitions(roomie_pipeline PRIVATE ROOMIE_ENABLE_NVBLOX=1)
  target_link_libraries(roomie_pipeline ${ROOMIE_NVBLOX_TARGET})
  if(EXISTS "${ROOMIE_NVBLOX_LIBRARY_DIR}/libnvblox_lib.so")
    set_target_properties(
      roomie_pipeline
      PROPERTIES
        BUILD_RPATH "${ROOMIE_NVBLOX_LIBRARY_DIR}"
        INSTALL_RPATH "${ROOMIE_NVBLOX_LIBRARY_DIR}")
  endif()
endif()

ament_target_dependencies(
  roomie_pipeline
  geometry_msgs
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_msgs
  visualization_msgs
)

add_executable(roomie_pipeline_node src/pipeline_node.cpp)
target_link_libraries(roomie_pipeline_node roomie_pipeline)
ament_target_dependencies(roomie_pipeline_node rclcpp)
if(ROOMIE_ENABLE_NVBLOX AND EXISTS "${ROOMIE_NVBLOX_LIBRARY_DIR}/libnvblox_lib.so")
  set_target_properties(
    roomie_pipeline_node
    PROPERTIES
      BUILD_RPATH "${ROOMIE_NVBLOX_LIBRARY_DIR}"
      INSTALL_RPATH "${ROOMIE_NVBLOX_LIBRARY_DIR}")
endif()

install(TARGETS roomie_pipeline
        EXPORT roomie-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
install(TARGETS roomie_pipeline_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/roomie_python_inference_worker.py
  scripts/roomie_dsg_viewer.py
  scripts/roomie_offline_room_partition_ui.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_object_snapshots test/test_object_snapshots.cpp)
  target_link_libraries(test_object_snapshots roomie_pipeline)
endif()

ament_export_targets(roomie-targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3
  geometry_msgs
  nlohmann_json
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_msgs
  visualization_msgs)
ament_package()
