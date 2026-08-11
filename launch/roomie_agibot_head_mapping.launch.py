from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def source_package_path(directory, filename):
    launch_path = Path(__file__).resolve()
    for parent in launch_path.parents:
        candidate = parent / "src" / "roomie" / directory / filename
        if candidate.exists():
            return str(candidate)
    for parent in launch_path.parents:
        candidate = parent / directory / filename
        if candidate.exists():
            return str(candidate)
    return str(launch_path.parents[1] / directory / filename)


def generate_launch_description():
    adapter_config = LaunchConfiguration("adapter_config")
    hand_adapter_config = LaunchConfiguration("hand_adapter_config")
    hand_camera_config = LaunchConfiguration("hand_camera_config")
    pipeline_config = LaunchConfiguration("pipeline_config")
    calibration_file = LaunchConfiguration("calibration_file")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    hand_left_color_topic = LaunchConfiguration("hand_left_color_topic")
    hand_right_color_topic = LaunchConfiguration("hand_right_color_topic")
    tf_topic = LaunchConfiguration("tf_topic")
    publish_static_tf = LaunchConfiguration("publish_static_tf")
    enable_boxer = LaunchConfiguration("enable_boxer")
    enable_hand_cameras = LaunchConfiguration("enable_hand_cameras")
    boxer_max_inference_fps = LaunchConfiguration("boxer_max_inference_fps")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    fixed_frame = LaunchConfiguration("fixed_frame")

    default_adapter_config = source_package_path(
        "config",
        "agibot_head_rgbd_adapter.yaml",
    )
    default_pipeline_config = source_package_path(
        "config",
        "pipeline_agibot_head_mapping.yaml",
    )
    default_hand_adapter_config = source_package_path(
        "config",
        "agibot_hand_color_adapter.yaml",
    )
    default_hand_camera_config = source_package_path(
        "config/robots/G2",
        "cameras.yaml",
    )
    default_rviz_config = source_package_path("rviz", "roomie_pipeline.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "adapter_config",
                default_value=default_adapter_config,
                description="AgiBot head RGB-D adapter parameter file.",
            ),
            DeclareLaunchArgument(
                "hand_adapter_config",
                default_value=default_hand_adapter_config,
                description="AgiBot hand-color adapter parameter file.",
            ),
            DeclareLaunchArgument(
                "hand_camera_config",
                default_value=default_hand_camera_config,
                description="Canonical hand camera geometry and raw distortion YAML.",
            ),
            DeclareLaunchArgument(
                "pipeline_config",
                default_value=default_pipeline_config,
                description="Roomie mapping-only parameter file.",
            ),
            DeclareLaunchArgument(
                "calibration_file",
                default_value=(
                    "/home/lindenbot/sensor_base/agibot_genie/"
                    "roomie_base/calib/head_camera_params.yaml"
                ),
                description="Head camera intrinsic and static-extrinsic YAML.",
            ),
            DeclareLaunchArgument(
                "color_topic",
                default_value="/gdk/camera/head_color",
                description="Live RGB image topic consumed by the adapter.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/gdk/camera/head_depth",
                description="Live depth image topic consumed by the adapter.",
            ),
            DeclareLaunchArgument(
                "hand_left_color_topic",
                default_value="/gdk/camera/hand_left_color",
                description="Raw left-hand color image topic.",
            ),
            DeclareLaunchArgument(
                "hand_right_color_topic",
                default_value="/gdk/camera/hand_right_color",
                description="Raw right-hand color image topic.",
            ),
            DeclareLaunchArgument(
                "tf_topic",
                default_value="/tf",
                description=(
                    "Dynamic TF topic consumed by Roomie. Use /tf for "
                    "genie_bag_player or /gdk/tf for a direct GDK source."
                ),
            ),
            DeclareLaunchArgument(
                "publish_static_tf",
                default_value="false",
                description=(
                    "Publish camera static transforms from calibration_file. "
                    "Leave false when genie_bag_player already publishes them."
                ),
            ),
            DeclareLaunchArgument(
                "enable_boxer",
                default_value="false",
                description=(
                    "Enable the OWLv2 + BoxerNet detection and instance-mapping "
                    "pipeline."
                ),
            ),
            DeclareLaunchArgument(
                "enable_hand_cameras",
                default_value="true",
                description=(
                    "Rectify and admit both hand cameras when detection is enabled."
                ),
            ),
            DeclareLaunchArgument(
                "boxer_max_inference_fps",
                default_value="5.0",
                description="Maximum OWLv2 + BoxerNet inference rate.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz with the Roomie mapping displays.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="RViz display configuration.",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="map",
                description="RViz fixed frame.",
            ),
            Node(
                package="roomie",
                executable="roomie_agibot_head_rgbd_adapter.py",
                name="roomie_agibot_head_rgbd_adapter",
                output="screen",
                parameters=[
                    adapter_config,
                    {
                        "calibration_file": calibration_file,
                        "color_topic": color_topic,
                        "depth_topic": depth_topic,
                        "independent_rgb_max_fps": ParameterValue(
                            boxer_max_inference_fps,
                            value_type=float,
                        ),
                        "publish_static_tf": ParameterValue(
                            publish_static_tf,
                            value_type=bool,
                        ),
                    },
                ],
            ),
            Node(
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            enable_boxer,
                            "'.lower() == 'true' and '",
                            enable_hand_cameras,
                            "'.lower() == 'true'",
                        ]
                    )
                ),
                package="roomie",
                executable="roomie_agibot_hand_color_adapter.py",
                name="roomie_agibot_hand_color_adapter",
                output="screen",
                parameters=[
                    hand_adapter_config,
                    {
                        "camera_config": hand_camera_config,
                        "hand_left_input_topic": hand_left_color_topic,
                        "hand_right_input_topic": hand_right_color_topic,
                        "max_fps": ParameterValue(
                            boxer_max_inference_fps,
                            value_type=float,
                        ),
                    },
                ],
            ),
            Node(
                package="roomie",
                executable="roomie_pipeline_node",
                name="roomie_pipeline_node",
                output="screen",
                # Saving the coordinated NVblox checkpoint happens at the end
                # of Roomie's graceful shutdown, after all semantic and
                # persistence queues have drained.  The launch default of five
                # seconds can SIGTERM the node before it publishes the map
                # manifest, leaving a non-empty SceneStore that cannot be
                # recovered safely on the next start.
                sigterm_timeout="120.0",
                sigkill_timeout="10.0",
                parameters=[
                    pipeline_config,
                    {
                        "topics.tf_topic": tf_topic,
                        "detection.enabled": ParameterValue(
                            enable_boxer,
                            value_type=bool,
                        ),
                        "detection.python_backend_enabled": ParameterValue(
                            enable_boxer,
                            value_type=bool,
                        ),
                        "artifacts.snapshot_enabled": ParameterValue(
                            enable_boxer,
                            value_type=bool,
                        ),
                        "artifacts.dam_enabled": ParameterValue(
                            enable_boxer,
                            value_type=bool,
                        ),
                        "artifacts.embedding_enabled": ParameterValue(
                            enable_boxer,
                            value_type=bool,
                        ),
                        "persistence.save_scene_graph": ParameterValue(
                            enable_boxer,
                            value_type=bool,
                        ),
                        "detection.max_inference_fps": ParameterValue(
                            boxer_max_inference_fps,
                            value_type=float,
                        ),
                    },
                ],
            ),
            Node(
                condition=IfCondition(use_rviz),
                package="rviz2",
                executable="rviz2",
                name="roomie_agibot_head_mapping_rviz",
                arguments=[
                    "-d",
                    rviz_config,
                    "-f",
                    fixed_frame,
                ],
                output="screen",
            ),
        ]
    )
