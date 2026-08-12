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
    "/home/lindenbot/RealityLab/map_ws/src/nvblox"
    CACHE PATH
    "Path to the nvblox source tree used when nvblox is not installed")
set(ROOMIE_NVBLOX_LIBRARY_DIR
    "/home/lindenbot/RealityLab/map_ws/install/nvblox_ros/lib"
    CACHE PATH
    "Runtime library directory for the local nvblox install")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
# ROS Humble's rosidl_generator_py still consumes legacy FindPythonInterp and
# FindPythonLibs variables. Resolve one modern Interpreter+Development tuple,
# then fence every legacy cache input to that same ABI; setting only the
# executable can otherwise pair /usr/bin/python3.10 with Conda's 3.13 headers
# and produce an unloadable service extension.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
    "Python interpreter used by ROSIDL generators" FORCE)
list(GET Python3_INCLUDE_DIRS 0 ROOMIE_PYTHON_INCLUDE_DIR)
list(GET Python3_LIBRARIES 0 ROOMIE_PYTHON_LIBRARY)
set(PYTHON_INCLUDE_DIR "${ROOMIE_PYTHON_INCLUDE_DIR}" CACHE PATH
    "Python include directory used by ROSIDL generators" FORCE)
set(PYTHON_LIBRARY "${ROOMIE_PYTHON_LIBRARY}" CACHE FILEPATH
    "Python library used by ROSIDL generators" FORCE)
set(PYTHON_SOABI "${Python3_SOABI}" CACHE INTERNAL
    "Python extension ABI used by ROSIDL generators" FORCE)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ego_shade REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(PNG REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(yaml-cpp REQUIRED)

if(TARGET SQLite3::SQLite3)
  set(ROOMIE_SQLITE_TARGET SQLite3::SQLite3)
elseif(TARGET SQLite::SQLite3)
  set(ROOMIE_SQLITE_TARGET SQLite::SQLite3)
else()
  message(FATAL_ERROR "SQLite3 was found but did not export a supported CMake target")
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/MutateScene.srv"
  "srv/QueryScene.srv"
)

add_library(
  roomie_pipeline
  src/artifacts/artifact_intent_builder.cpp
  src/artifacts/artifact_runtime_actor.cpp
  src/artifacts/artifact_scheduler.cpp
  src/artifacts/python_dam_worker.cpp
  src/artifacts/embedding_runtime_actor.cpp
  src/artifacts/embedding_task_scheduler.cpp
  src/artifacts/online_snapshot_worker.cpp
  src/artifacts/python_embedding_encoder.cpp
  src/artifacts/semantic_index.cpp
  src/artifacts/semantic_scene_projector.cpp
  src/artifacts/semantic_index_store_adapter.cpp
  src/artifacts/snapshot_bank.cpp
  src/association_engine.cpp
  src/cpu_point_map_backend.cpp
  src/detection_bridge_thread.cpp
  src/ephemeral_run_workspace.cpp
  src/furniture_evidence.cpp
  src/dsg/object_graph.cpp
  src/dsg/object_graph_io.cpp
  src/dsg/object_snapshot_remaker.cpp
  src/image_utils.cpp
  src/instance_map_thread.cpp
  src/label_thresholds.cpp
  src/map_backend.cpp
  src/map_thread.cpp
  src/pipeline.cpp
  src/pipeline_config.cpp
  src/presence_evidence.cpp
  src/publisher_persistence_thread.cpp
  src/python_inference_backend.cpp
  src/robot_mask_generator.cpp
  src/robot_state_estimator.cpp
  src/query/scene_mutation_json_adapter.cpp
  src/query/scene_query_json_adapter.cpp
  src/query/scene_query_gateway.cpp
  src/query/semantic_search_provider.cpp
  src/ros_io_thread.cpp
  src/scene/geometry_scheduler.cpp
  src/scene/geometry_worker.cpp
  src/scene/furniture_graph.cpp
  src/scene/persistence_actor.cpp
  src/scene/scene_reducer.cpp
  src/scene/scene_store.cpp
  src/utils/run_logger.cpp
  src/utils/visualization_utils.cpp
  src/worker_thread.cpp
)
target_include_directories(
  roomie_pipeline PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                         "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>"
)
target_link_libraries(roomie_pipeline Eigen3::Eigen nlohmann_json::nlohmann_json
                      ego_shade::core yaml-cpp ${ROOMIE_SQLITE_TARGET} PNG::PNG
                      ${OpenCV_LIBS})

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
  ament_index_cpp
  geometry_msgs
  nav_msgs
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
ament_target_dependencies(roomie_pipeline_node rclcpp std_srvs)
rosidl_get_typesupport_target(roomie_cpp_typesupport_target
                              ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(roomie_pipeline_node "${roomie_cpp_typesupport_target}")

add_executable(roomie_offline_scene_worker src/offline_scene_worker.cpp)
target_link_libraries(roomie_offline_scene_worker roomie_pipeline)
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
install(TARGETS roomie_pipeline_node roomie_offline_scene_worker
        RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)
install(PROGRAMS
  scripts/roomie_agibot_head_rgbd_adapter.py
  scripts/roomie_agibot_hand_color_adapter.py
  scripts/roomie_bag_replay_harness.py
  scripts/roomie_python_dam_worker.py
  scripts/roomie_python_inference_worker.py
  scripts/roomie_sentence_transformer_worker.py
  scripts/roomie_dsg_viewer.py
  scripts/roomie_offline_scene_editor.py
  scripts/roomie_offline_room_partition_ui.py
  scripts/roomie_scene_qa.py
  scripts/roomie_scene_qa_viewer.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY scripts/scene_qa DESTINATION lib/${PROJECT_NAME}
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_bounded_channel test/test_bounded_channel.cpp)
  target_link_libraries(test_bounded_channel roomie_pipeline)
  ament_add_gtest(test_image_utils test/test_image_utils.cpp)
  target_link_libraries(test_image_utils roomie_pipeline)
  ament_add_gtest(test_version_types test/test_version_types.cpp)
  target_link_libraries(test_version_types roomie_pipeline)
  ament_add_gtest(test_inference_wire test/test_inference_wire.cpp)
  target_link_libraries(test_inference_wire roomie_pipeline)
  ament_add_gtest(test_label_thresholds test/test_label_thresholds.cpp)
  target_link_libraries(test_label_thresholds roomie_pipeline)
  ament_add_gtest(test_association_engine test/test_association_engine.cpp)
  target_link_libraries(test_association_engine roomie_pipeline)
  ament_add_gtest(test_presence_evidence test/test_presence_evidence.cpp)
  target_link_libraries(test_presence_evidence roomie_pipeline)
  ament_add_gtest(test_furniture_evidence test/test_furniture_evidence.cpp)
  target_link_libraries(test_furniture_evidence roomie_pipeline)
  ament_add_gtest(test_robot_mask_generator
                  test/test_robot_mask_generator.cpp)
  target_link_libraries(test_robot_mask_generator roomie_pipeline)
  target_compile_definitions(
    test_robot_mask_generator
    PRIVATE "ROOMIE_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
  ament_add_gtest(test_robot_state_estimator
                  test/test_robot_state_estimator.cpp)
  target_link_libraries(test_robot_state_estimator roomie_pipeline)
  target_compile_definitions(
    test_robot_state_estimator
    PRIVATE "ROOMIE_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
  ament_add_gtest(test_python_inference_backend
                  test/test_python_inference_backend.cpp)
  target_link_libraries(test_python_inference_backend roomie_pipeline)
  target_compile_definitions(
    test_python_inference_backend
    PRIVATE
      "ROOMIE_FAKE_INFERENCE_WORKER_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/fake_inference_worker.py\"")
  ament_add_gtest(test_python_embedding_encoder
                  test/test_python_embedding_encoder.cpp)
  target_link_libraries(test_python_embedding_encoder roomie_pipeline)
  target_compile_definitions(
    test_python_embedding_encoder
    PRIVATE
      "ROOMIE_FAKE_EMBEDDING_WORKER_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/fake_embedding_worker.py\"")
  ament_add_gtest(test_surface_snapshot test/test_surface_snapshot.cpp)
  target_link_libraries(test_surface_snapshot roomie_pipeline)
  ament_add_gtest(test_map_causal_barrier test/test_map_causal_barrier.cpp)
  target_link_libraries(test_map_causal_barrier roomie_pipeline)
  ament_add_gtest(test_scene_reducer test/test_scene_reducer.cpp)
  target_link_libraries(test_scene_reducer roomie_pipeline)
  ament_add_gtest(test_furniture_graph test/test_furniture_graph.cpp)
  target_link_libraries(test_furniture_graph roomie_pipeline)
  target_compile_definitions(
    test_furniture_graph
    PRIVATE "ROOMIE_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
  ament_add_gtest(test_geometry_scheduler test/test_geometry_scheduler.cpp)
  target_link_libraries(test_geometry_scheduler roomie_pipeline)
  ament_add_gtest(test_geometry_worker test/test_geometry_worker.cpp)
  target_link_libraries(test_geometry_worker roomie_pipeline)
  ament_add_gtest(test_snapshot_bank test/test_snapshot_bank.cpp)
  target_link_libraries(test_snapshot_bank roomie_pipeline)
  ament_add_gtest(test_online_snapshot_worker test/test_online_snapshot_worker.cpp)
  target_link_libraries(test_online_snapshot_worker roomie_pipeline)
  ament_add_gtest(test_artifact_scheduler test/test_artifact_scheduler.cpp)
  target_link_libraries(test_artifact_scheduler roomie_pipeline)
  ament_add_gtest(test_artifact_runtime_actor
                  test/test_artifact_runtime_actor.cpp)
  target_link_libraries(test_artifact_runtime_actor roomie_pipeline)
  ament_add_gtest(test_python_dam_worker test/test_python_dam_worker.cpp)
  target_link_libraries(test_python_dam_worker roomie_pipeline)
  target_compile_definitions(
    test_python_dam_worker
    PRIVATE
      "ROOMIE_FAKE_DAM_WORKER_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures/fake_dam_worker.py\"")
  ament_add_pytest_test(test_dam_worker_official_api
                        test/test_dam_worker_official_api.py)
  ament_add_pytest_test(test_query_scene_ros_typesupport
                        test/test_query_scene_ros_typesupport.py
                        APPEND_ENV
                          "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_py")
  ament_add_pytest_test(test_scene_qa test/test_scene_qa.py TIMEOUT 60)
  ament_add_gtest(test_embedding_runtime_actor
                  test/test_embedding_runtime_actor.cpp)
  target_link_libraries(test_embedding_runtime_actor roomie_pipeline)
  ament_add_gtest(test_artifact_intent_builder test/test_artifact_intent_builder.cpp)
  target_link_libraries(test_artifact_intent_builder roomie_pipeline)
  ament_add_gtest(test_semantic_index test/test_semantic_index.cpp)
  target_link_libraries(test_semantic_index roomie_pipeline)
  ament_add_gtest(test_semantic_scene_projector
                  test/test_semantic_scene_projector.cpp)
  target_link_libraries(test_semantic_scene_projector roomie_pipeline)
  ament_add_gtest(test_semantic_index_store_adapter
                  test/test_semantic_index_store_adapter.cpp)
  target_link_libraries(test_semantic_index_store_adapter roomie_pipeline)
  ament_add_gtest(test_scene_query_gateway test/test_scene_query_gateway.cpp)
  target_link_libraries(test_scene_query_gateway roomie_pipeline)
  ament_add_gtest(test_scene_query_json_adapter
                  test/test_scene_query_json_adapter.cpp)
  target_link_libraries(test_scene_query_json_adapter roomie_pipeline)
  ament_add_gtest(test_scene_mutation_json_adapter
                  test/test_scene_mutation_json_adapter.cpp)
  target_link_libraries(test_scene_mutation_json_adapter roomie_pipeline)
  ament_add_gtest(test_semantic_search_provider
                  test/test_semantic_search_provider.cpp)
  target_link_libraries(test_semantic_search_provider roomie_pipeline)
  ament_add_gtest(test_scene_store test/test_scene_store.cpp)
  target_link_libraries(test_scene_store roomie_pipeline)
  ament_add_gtest(test_persistence_actor test/test_persistence_actor.cpp)
  target_link_libraries(test_persistence_actor roomie_pipeline)
  ament_add_gtest(test_ephemeral_run_workspace
                  test/test_ephemeral_run_workspace.cpp)
  target_link_libraries(test_ephemeral_run_workspace roomie_pipeline)
  ament_add_gtest(test_object_snapshots test/test_object_snapshots.cpp)
  target_link_libraries(test_object_snapshots roomie_pipeline)
  ament_add_gtest(test_runtime_bounds test/test_runtime_bounds.cpp)
  target_link_libraries(test_runtime_bounds roomie_pipeline)
  ament_add_pytest_test(
    test_agibot_head_rgbd_adapter
    test/test_agibot_head_rgbd_adapter.py
    TIMEOUT 60)
  ament_add_pytest_test(
    test_agibot_hand_color_adapter
    test/test_agibot_hand_color_adapter.py
    TIMEOUT 60)
  ament_add_pytest_test(
    test_bag_replay_harness
    test/test_bag_replay_harness.py
    TIMEOUT 60)
  ament_add_pytest_test(
    test_inference_protocol
    test/test_inference_protocol.py
    TIMEOUT 60)
  ament_add_pytest_test(
    test_embedding_worker_protocol
    test/test_embedding_worker_protocol.py
    TIMEOUT 60)
  if(ROOMIE_ENABLE_NVBLOX)
    ament_add_gtest(
      test_nvblox_backend_conformance
      test/test_nvblox_backend_conformance.cpp
      TIMEOUT 120)
    if(TARGET test_nvblox_backend_conformance)
      target_compile_definitions(
        test_nvblox_backend_conformance PRIVATE ROOMIE_ENABLE_NVBLOX=1)
      target_link_libraries(
        test_nvblox_backend_conformance roomie_pipeline CUDA::cudart)
      set_tests_properties(
        test_nvblox_backend_conformance
        PROPERTIES LABELS "gtest;gpu;nvblox")
    endif()
  endif()
endif()

ament_export_targets(roomie-targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3
  ament_index_cpp
  ego_shade
  geometry_msgs
  nav_msgs
  nlohmann_json
  OpenCV
  PNG
  rclcpp
  rosidl_default_runtime
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_msgs
  visualization_msgs
  SQLite3
  yaml-cpp)
ament_package()
