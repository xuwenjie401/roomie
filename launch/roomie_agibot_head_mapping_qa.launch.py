"""Launch AgiBot mapping, RViz, and the live browser Scene QA console."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def roomie_executable(filename: str) -> str:
    """Resolve an installed executable, with a source-tree fallback for development."""

    try:
        installed = Path(get_package_prefix("roomie")) / "lib" / "roomie" / filename
        if installed.exists():
            return str(installed)
    except Exception:
        pass
    launch_path = Path(__file__).resolve()
    for parent in launch_path.parents:
        candidate = parent / "src" / "roomie" / "scripts" / filename
        if candidate.exists():
            return str(candidate)
        candidate = parent / "scripts" / filename
        if candidate.exists():
            return str(candidate)
    return str(launch_path.parents[1] / "scripts" / filename)


def default_qa_python() -> str:
    configured = os.environ.get("ROOMIE_QA_PYTHON")
    if configured:
        return configured
    jarvis_python = Path("/home/lindenbot/miniconda3/envs/jarvis/bin/python")
    return str(jarvis_python) if jarvis_python.exists() else sys.executable


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("roomie"))
    mapping_launch = share / "launch" / "roomie_agibot_head_mapping.launch.py"

    adapter_config = LaunchConfiguration("adapter_config")
    pipeline_config = LaunchConfiguration("pipeline_config")
    calibration_file = LaunchConfiguration("calibration_file")
    tf_topic = LaunchConfiguration("tf_topic")
    publish_static_tf = LaunchConfiguration("publish_static_tf")
    enable_boxer = LaunchConfiguration("enable_boxer")
    boxer_max_inference_fps = LaunchConfiguration("boxer_max_inference_fps")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    fixed_frame = LaunchConfiguration("fixed_frame")

    use_scene_qa = LaunchConfiguration("use_scene_qa")
    qa_python = LaunchConfiguration("qa_python")
    qa_config = LaunchConfiguration("qa_config")
    qa_service = LaunchConfiguration("qa_service")
    qa_service_timeout_sec = LaunchConfiguration("qa_service_timeout_sec")
    qa_host = LaunchConfiguration("qa_host")
    qa_port = LaunchConfiguration("qa_port")
    qa_browser = LaunchConfiguration("qa_browser")
    qa_default_provider = LaunchConfiguration("qa_default_provider")

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(mapping_launch)),
        launch_arguments={
            "adapter_config": adapter_config,
            "pipeline_config": pipeline_config,
            "calibration_file": calibration_file,
            "tf_topic": tf_topic,
            "publish_static_tf": publish_static_tf,
            "enable_boxer": enable_boxer,
            "boxer_max_inference_fps": boxer_max_inference_fps,
            "use_rviz": use_rviz,
            "rviz_config": rviz_config,
            "fixed_frame": fixed_frame,
        }.items(),
    )

    scene_qa = ExecuteProcess(
        condition=IfCondition(use_scene_qa),
        cmd=[
            qa_python,
            roomie_executable("roomie_scene_qa_viewer.py"),
            "--live",
            "--qa-config",
            qa_config,
            "--service",
            qa_service,
            "--service-timeout-sec",
            qa_service_timeout_sec,
            "--host",
            qa_host,
            "--port",
            qa_port,
            "--browser",
            qa_browser,
            "--default-provider",
            qa_default_provider,
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "adapter_config",
                default_value=str(share / "config" / "agibot_head_rgbd_adapter.yaml"),
                description="AgiBot head RGB-D adapter parameter file.",
            ),
            DeclareLaunchArgument(
                "pipeline_config",
                default_value=str(share / "config" / "pipeline_agibot_head_mapping.yaml"),
                description="Roomie pipeline parameter file.",
            ),
            DeclareLaunchArgument(
                "calibration_file",
                default_value=(
                    "/home/lindenbot/sensor_base/agibot_genie/"
                    "roomie_base/calib/head_camera_params.yaml"
                ),
                description="Head camera intrinsic and static-extrinsic YAML.",
            ),
            DeclareLaunchArgument("tf_topic", default_value="/tf"),
            DeclareLaunchArgument("publish_static_tf", default_value="false"),
            DeclareLaunchArgument(
                "enable_boxer",
                default_value="true",
                description="Enable the full Boxer/snapshot/DAM/embedding semantic path.",
            ),
            DeclareLaunchArgument("boxer_max_inference_fps", default_value="5.0"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=str(share / "rviz" / "roomie_pipeline.rviz"),
            ),
            DeclareLaunchArgument("fixed_frame", default_value="map"),
            DeclareLaunchArgument(
                "use_scene_qa",
                default_value="true",
                description="Start the live browser Scene QA console.",
            ),
            DeclareLaunchArgument(
                "qa_python",
                default_value=default_qa_python(),
                description="Python interpreter containing google-genai for Gemini.",
            ),
            DeclareLaunchArgument(
                "qa_config",
                default_value=str(share / "config" / "scene_qa" / "config.json"),
                description="Gemini/Doubao models, prompt, and Scene QA runtime config.",
            ),
            DeclareLaunchArgument("qa_service", default_value="/roomie/query_scene"),
            DeclareLaunchArgument("qa_service_timeout_sec", default_value="10.0"),
            DeclareLaunchArgument("qa_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("qa_port", default_value="8776"),
            DeclareLaunchArgument(
                "qa_default_provider",
                default_value="gemini",
                description="Initial browser provider: gemini or doubao.",
            ),
            DeclareLaunchArgument(
                "qa_browser",
                default_value="auto",
                description="Use auto to open the browser or off to print the URL only.",
            ),
            mapping,
            scene_qa,
        ]
    )
