# Roomie 当前使用说明

更新时间：2026-08-12

适用工作区：`/home/lindenbot/RealityLab/jarvis`

## 1. 环境准备

当前 Roomie 主要使用 ROS 2 Humble 和 NVBLOX：

```bash
source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/map_ws/install/setup.bash
source /home/lindenbot/RealityLab/tools_ws/install/setup.bash
```

若使用 Conda，建议编译时明确指定系统 Python，避免 `ament` Python 环境冲突。

Scene QA 使用独立的 `jarvis` Conda 环境；本机默认路径为
`/home/lindenbot/miniconda3/envs/jarvis/bin/python`。Gemini 需要
`google-genai`，场景搜索/截图需要 `numpy` 和 `Pillow`。Doubao 直接调用火山方舟
Responses HTTP API，使用 Python 标准库，不要求安装 `openai` 或火山 SDK：

```bash
source /home/lindenbot/miniconda3/etc/profile.d/conda.sh
conda activate jarvis

python -c 'import google.genai, numpy, PIL; print("Scene QA dependencies: OK")'
test -n "${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}" && echo "Gemini API key: configured"
test -n "${DOUBAO_API_KEY:-${ARK_API_KEY:-}}" && echo "Doubao API key: configured"
```

Gemini 依次读取 `--api-key`、`GEMINI_API_KEY`、`GOOGLE_API_KEY`；Doubao 依次读取
`--doubao-api-key`、`DOUBAO_API_KEY`、`doubao_api_key`、`ARK_API_KEY`。如果 key
已经在 `.bashrc` 中通过 `export` 设置，打开新终端后 launch 会直接继承，不要再把它
写到 launch 参数或命令历史里。在线问答还需要能够访问相应 API 并具有模型配额。
Doubao 默认模型和地址为 `doubao-seed-2-0-lite-260215`、
`https://ark.cn-beijing.volces.com/api/v3`，默认关闭 thinking；模型、地址和
`doubao_thinking_type` 都可在
`config/scene_qa/config.json` 中修改。ROS 2 Humble
继续使用系统 Python 3.10；QA 程序会自动通过系统 Python 子进程访问
`/roomie/query_scene`，不要求在 Conda 环境中重新安装 ROS。

方舟接口参考：[快速入门](https://www.volcengine.com/docs/82379/1795150)、
[Responses API 函数调用](https://www.volcengine.com/docs/82379/1958524?lang=zh)。

## 2. 编译

```bash
cd /home/lindenbot/RealityLab/jarvis

colcon build \
  --packages-select roomie \
  --symlink-install \
  --cmake-args \
    -DROOMIE_ENABLE_NVBLOX=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3 \
    -Dnvblox_DIR=/home/lindenbot/RealityLab/map_ws/install/nvblox_ros/share/nvblox/cmake

source install/setup.bash
```

仅使用 CPU 地图后端时，可以将 `ROOMIE_ENABLE_NVBLOX` 设置为 `OFF`。

当前 workspace 是 `--symlink-install`：`share/roomie/config`、launch 和 Python
入口通常链接回源码。可用下面的命令确认具体文件：

```bash
readlink -f install/roomie/share/roomie/config/pipeline_agibot_head_mapping.yaml
```

修改已有 YAML/JSON 配置不需要重新编译，但参数只在节点启动时读取，因此必须重启
launch。若命令像本文示例一样直接传入源码 YAML 的绝对路径，则根本不经过 install
副本。修改 C++ 必须重新执行 `colcon build`；新增待安装文件或修改 CMake install
规则也应重新构建。修改软链接指向的既有 launch/Python 文件通常不用重建，但运行中
进程仍需重启。

## 3. 配置

AgiBot 头部相机映射的参考配置：

- `config/pipeline_agibot_head_mapping.yaml`
- `config/agibot_head_rgbd_adapter.yaml`

开始运行前，应检查以下配置项：

```yaml
basic:
  world_frame: map

topics:
  color_topic: /roomie/input/head_color/image_rect
  depth_topic: /roomie/input/head_color/depth_registered
  camera_info_topic: /roomie/input/head_color/camera_info
  odom_topic: /odom
  tf_topic: /tf

robot_mask:
  robot_config: G2/robot.yaml
  camera_config: G2/cameras.yaml
  reuse_translation_epsilon_m: 0.000005
  reuse_rotation_epsilon_rad: 0.000005

robot_state:
  enabled: true
  odom_history_sec: 5.0
  odom_stale_timeout_sec: 0.25
  rotation_window_sec: 0.10
  rotation_enter_rad_s: 0.05
  rotation_exit_rad_s: 0.02
  rotation_exit_hold_sec: 0.30
  drop_frames_when_unknown: true
```

输入图像必须是与相机 profile 完全一致的 rectified pinhole 图像。Roomie 不再订阅
mask topic，而是在每个 RGB 时间戳上用 G2 内部 TF 直接生成 mask；每个相机拥有独立
缓存。精确时刻的内部 TF 不完整时，该帧会等待后续 TF，若期间 RGB 已更新则丢弃旧帧。
后续增加 `hand_left_color` 或 `hand_right_color` 时，应为各自的 rectified 图像建立独立
相机订阅，mask 生成与缓存仍按 camera id 隔离。

当前配置已通过 `detection.additional_camera_ids` 接入两路 detection-only 手相机：

```yaml
detection:
  additional_camera_ids: [hand_left_color, hand_right_color]
```

原始手图不能直接送入 pipeline。`roomie_agibot_hand_color_adapter.py` 会按
`config/robots/G2/cameras.yaml` 中的 source distortion 去畸变，发布 rectified 图像、
zero-distortion `CameraInfo` 和两路静态相机 TF。`roomie_agibot_live.launch.py` 默认将输入
设为 `/live_connect/hand_left_color`、`/live_connect/hand_right_color`；手相机适配器只在
`enable_boxer:=true` 且 `enable_hand_cameras:=true` 时启动。完整 bag 示例和限制见
`HAND_IMG.md`。

机器人状态机使用 `/odom` pose 在图像时间戳上估算 yaw 角速度，不依赖当前 G2 数据中
恒为零的 `twist.angular.z`。角速度达到 `rotation_enter_rad_s` 后，该时间段的 mapping 和
detection 帧都会在入队前丢弃；只有连续 `rotation_exit_hold_sec` 低于退出阈值才恢复。
启动阶段、odom 非法或超过 `odom_stale_timeout_sec` 时状态为 unknown，默认同样丢帧。
可通过 `robot_state.enabled: false` 恢复旧的无状态过滤行为。当前版本不控制 DAM；手部
相机则由 body/arm 相对默认导航构型的偏离状态统一启停，具体见 `HAND_IMG.md`。

还需要检查所有本机路径：

- NVBLOX 地图保存和加载目录
- SceneStore SQLite 路径
- DSG JSON 输出路径
- 日志目录
- Boxer 仓库和模型路径
- DAM 模型、源码及虚拟环境路径
- Sentence-T5 模型路径
- 相机标定文件路径

建议复制一份 YAML，为每次实验使用独立的输出目录。

## 4. 运行模式

### 4.1 通用 Pipeline

```bash
source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/map_ws/install/setup.bash
source /home/lindenbot/RealityLab/tools_ws/install/setup.bash
source /home/lindenbot/RealityLab/jarvis/install/setup.bash

ros2 launch roomie roomie_pipeline.launch.py \
  config_path:=/absolute/path/to/pipeline.yaml \
  use_rviz:=true \
  show_manual_rooms:=true
```

默认使用 `pipeline_genie_live.yaml`。其中仍包含当前机器的模型和输出路径，使用前应先检查。

### 4.2 AgiBot 建图模式

只运行 RGB-D 适配器、NVBLOX 建图和 SceneStore：

```bash
ros2 launch roomie roomie_agibot_head_mapping.launch.py \
  pipeline_config:=/absolute/path/to/pipeline.yaml \
  enable_boxer:=false \
  use_rviz:=true
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `adapter_config` | RGB-D 适配器配置 |
| `pipeline_config` | Roomie Pipeline 配置 |
| `calibration_file` | 相机标定文件 |
| `tf_topic` | TF 输入话题 |
| `publish_static_tf` | 是否发布静态相机 TF |
| `enable_boxer` | 是否启用完整语义感知 |
| `boxer_max_inference_fps` | Boxer 最大推理帧率 |
| `use_rviz` | 是否启动 RViz |
| `fixed_frame` | RViz 固定坐标系 |

使用 Genie Bag Player 时，`tf_topic` 通常为 `/tf`；直接连接 GDK 时可能需要改为 `/gdk/tf`。

使用 `live_connect` 的本地 RGB-D 输出时，先启动连接节点，再使用 live 入口。该
入口除订阅 `/live_connect/head_color` 和 `/live_connect/head_depth` 外，其余行为
与 `roomie_agibot_head_mapping.launch.py` 一致。默认使用
`config/pipeline_genie_live.yaml`，并将地图、SceneStore、资产和日志隔离到
`genie_live` 数据目录：

```bash
ros2 launch live_connect live_connect.launch.py

# 在另一个已加载工作区环境的终端中运行：
ros2 launch roomie roomie_agibot_live.launch.py \
  enable_boxer:=true \
  use_rviz:=true

# 完整语义、RViz 和浏览器问答一体启动：
ros2 launch roomie roomie_agibot_live_with_qa.launch.py \
  boxer_max_inference_fps:=5.0
```

### 4.3 完整语义模式

```bash
ros2 launch roomie roomie_agibot_head_mapping.launch.py \
  pipeline_config:=/absolute/path/to/pipeline.yaml \
  enable_boxer:=true \
  boxer_max_inference_fps:=5.0 \
  use_rviz:=true
```

`enable_boxer:=true` 会同时启用：

- 目标检测
- Python 检测后端
- Snapshot
- DAM
- Embedding
- DSG 持久化

运行前必须确保 Boxer、DAM、Sentence-T5 等模型路径有效。

### 4.4 完整语义、RViz 和实时问答一体启动

推荐使用更上层的组合 launch：

```bash
ros2 launch roomie roomie_agibot_head_mapping_qa.launch.py \
  pipeline_config:=/absolute/path/to/pipeline.yaml \
  boxer_max_inference_fps:=5.0
```

这个 launch 默认同时启动：

- AgiBot RGB-D adapter
- Roomie mapping pipeline
- Boxer、snapshot、DAM 和 embedding 完整语义链
- RViz2
- 连接 `/roomie/query_scene` 的浏览器问答界面

浏览器默认打开 `http://127.0.0.1:8776/`。页面左侧可用按钮逐题切换
Gemini/Doubao 并输入问题，中间显示从工具结果获得的对象几何，右侧实时显示所选模型
的规划状态、场景图工具名、参数、返回值、对象候选及最终回答。页面显示的是模型显式
返回的 reasoning 摘要和可审计工具轨迹，
不会也不能展示模型内部隐藏的 chain-of-thought。

组合 launch 的 QA 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `use_scene_qa` | `true` | 是否启动浏览器问答服务 |
| `qa_python` | `jarvis` 环境 Python | Gemini 需要包含 `google-genai`；Doubao 无额外 SDK |
| `qa_config` | 包内 `config/scene_qa/config.json` | Gemini/Doubao 模型、system prompt 和 QA 参数 |
| `qa_service` | `/roomie/query_scene` | live QueryScene 服务名 |
| `qa_service_timeout_sec` | `10.0` | 服务发现和调用超时 |
| `qa_host` | `127.0.0.1` | Web 服务监听地址 |
| `qa_port` | `8776` | Web 服务端口；设为 `0` 可由系统分配 |
| `qa_default_provider` | `gemini` | 页面首次打开时选中 `gemini` 或 `doubao` |
| `qa_browser` | `auto` | `auto` 自动打开浏览器，`off` 只打印 URL |

例如不自动打开浏览器，或者使用另一套 Python 环境：

```bash
ros2 launch roomie roomie_agibot_head_mapping_qa.launch.py \
  pipeline_config:=/absolute/path/to/pipeline.yaml \
  qa_browser:=off \
  qa_python:=/absolute/path/to/python
```

若希望页面首次打开时默认选中 Doubao，再加 `qa_default_provider:=doubao`；页面打开后
仍可随时用按钮切换下一问的 provider。

`enable_boxer` 和 `use_rviz` 在组合 launch 中默认均为 `true`，仍可显式覆盖。
QA Web 服务与 pipeline 同时启动；如果刚打开页面就提问，服务会在
`qa_service_timeout_sec` 内等待 `/roomie/query_scene`。Boxer/DAM/embedding 尚未完成
首批任务时，早期回答可能为空或不完整。Web 服务当前没有登录认证；除非已经配置
防火墙或反向代理认证，不要把 `qa_host` 从 loopback 改为公网可访问地址。
两个 provider 都不可用时 Web 服务会明确退出；只有一个可用时页面会禁用另一个按钮。
模型会收到问题及其主动请求的场景工具结果；Doubao 的多轮函数调用按方舟 Responses
API 使用存储响应和 `previous_response_id` 续接，不应把不允许发送给外部模型的数据放入
场景或问题中。

## 5. 播放 AgiBot 数据

先启动 Roomie，再在另一个终端播放数据。

终端 1：

```bash
source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/map_ws/install/setup.bash
source /home/lindenbot/RealityLab/tools_ws/install/setup.bash
source /home/lindenbot/RealityLab/jarvis/install/setup.bash

ros2 launch roomie roomie_agibot_head_mapping.launch.py \
  pipeline_config:=/absolute/path/to/pipeline.yaml \
  enable_boxer:=false \
  use_rviz:=true
```

终端 2：

```bash
source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/tools_ws/install/setup.bash

ros2 run genie_bag_player genie_bag_play \
  /home/lindenbot/Datasets/roomie/agibot/test1_0730_1547 \
  --image
```

## 6. 地图保存和恢复

当前推荐：

```yaml
tsdf:
  map_load_mode: coordinated
```

`coordinated` 模式下：

- 如果 SceneStore 内存在有效地图 manifest，启动时会自动恢复对应地图。
- 地图文件、世界坐标系、配置指纹、文件大小和哈希都会校验。
- 语义状态会回退或恢复到与地图 checkpoint 对齐的 SceneStore revision。
- 即使新运行配置中的 `load_map: false`，有效的 durable manifest 仍可能触发恢复。
- `map_load_path` 应保持未设置；实际路径来自同一个 SceneStore 的 durable manifest。

如需明确开始一条新的地图历史，使用：

```yaml
tsdf:
  map_load_mode: seed
```

注意：

- 新实验不要意外复用旧的 SQLite 和地图输出目录。
- 恢复时必须保持相同的 `world.frame_id` 和地图关键参数。
- 配置中的 `map_load_path` 若与 durable manifest 冲突，启动会拒绝加载。
- 当前地图 checkpoint 主要在正常关闭时写入；结束运行请使用一次 `Ctrl-C`，并等待保存完成。
- 当前没有周期性地图 checkpoint。

`load_map: true` 不会从 `map_save_path` 自动猜测某个 `.nvblox` 文件。在
`coordinated` 模式下它依赖 SceneStore manifest；只有导入一份与现有 SceneStore
无关的地图时，才应显式使用 `map_load_mode: seed` 和 `map_load_path`。`seed` 会清除旧的
durable scene lineage，避免旧对象与新地图混用。

加载成功后日志应同时出现类似以下记录，即使尚未收到新的相机帧，RViz 也应立即看到
恢复的 surface：

```text
loaded durable checkpoint path=... aligned_scene_revision=...
refreshed_nvblox_surface_cache mode=full ... surface_points=...
published loaded checkpoint surface
```

若只看到第一行而地图为空，请确认正在运行的是本次重新构建后的 C++ library。可检查：

```bash
readlink -f /home/lindenbot/RealityLab/jarvis/install/roomie/share/roomie/config/pipeline_agibot_head_mapping.yaml
rg -n 'loaded durable|published loaded|refreshed_nvblox' logs/agibot_head_mapping/*/map_checkpoint.log logs/agibot_head_mapping/*/map.log
sqlite3 /home/lindenbot/Datasets/output/roomie_agibot_head/roomie_scene.sqlite3 \
  'select checkpoint_id, checkpoint_path, aligned_scene_revision from map_checkpoint_manifest order by checkpoint_id;'
```

### 6.1 一次无痕运行

需要从同一份 coordinated checkpoint 反复测试在线更新、同时不改写初始
map/scene/assets 基线时，使用：

```bash
ros2 launch roomie roomie_agibot_live_with_qa.launch.py \
  pipeline_config:=/home/lindenbot/RealityLab/jarvis/install/roomie/share/roomie/config/pipeline_genie_ephemeral.yaml \
  enable_boxer:=true
```

该配置保持 TSDF、Boxer、instance、snapshot、DAM、embedding 和 QA 在线运行。
启动时会把 SceneStore 一致性复制到 `/tmp/roomie_ephemeral` 下的唯一工作区，
并为已有 snapshot assets 建立只读叠加；本次产生的 scene revision、任务、图片、
DSG 导出全部写入该工作区。初始 NVBlox checkpoint 只读加载，退出时不保存新地图。
正常退出后临时工作区自动删除，`logging.root_dir` 下的诊断日志仍会保留。

无痕模式只支持 `tsdf.map_load_mode: coordinated`，且要求基线 SceneStore 中存在
有效 map manifest。临时工作区准备失败时 pipeline 会拒绝启动，不会退回原路径写入。
`SIGKILL` 或掉电可能留下临时目录，但不会修改基线；确认没有运行中的 Roomie 后可
手动清理相应的 `roomie_ephemeral_*` 目录。

## 7. 运行状态检查

```bash
ros2 topic hz /roomie/map_surface
ros2 topic echo /roomie/objects --once
ros2 service list | rg roomie
```

主要输出话题：

```text
/roomie/map_surface
/roomie/detections_2d_image
/roomie/detections_2d
/roomie/raw_detections
/roomie/instances
/roomie/objects
```

## 8. 查询场景

查询服务：

```text
/roomie/query_scene
```

服务类型：

```text
roomie/srv/QueryScene
```

查询房间示例：

```bash
ros2 service call /roomie/query_scene roomie/srv/QueryScene \
  "{request_json: '{\"schema_version\":\"roomie.query_scene.v1\",\"calls\":[{\"id\":\"rooms\",\"method\":\"rooms\",\"params\":{}}]}'}"
```

批量查询示例：

```json
{
  "schema_version": "roomie.query_scene.v1",
  "calls": [
    {
      "id": "object",
      "method": "get_object",
      "params": {
        "object_id": 7
      }
    },
    {
      "id": "nearby",
      "method": "get_objects_near",
      "params": {
        "center_world": [1.0, 2.0, 0.5],
        "radius_m": 2.0
      }
    },
    {
      "id": "relations",
      "method": "relations",
      "params": {
        "object_id": 7,
        "direction": "either"
      }
    }
  ]
}
```

支持的方法包括：

```text
get_object
get_objects_near
rooms
relations
inspect_snapshot
search_objects
```

长时间、多轮查询可以先调用 `begin_session`，并在后续请求中携带 `session_id` 和 `expected_scene_revision`。

## 9. 人工修订场景

修订服务：

```text
/roomie/mutate_scene
```

服务类型：

```text
roomie/srv/MutateScene
```

请求示例：

```json
{
  "schema_version": "roomie.mutate_scene.v1",
  "operation": "apply_human_annotation",
  "request_id": "edit-42",
  "base_scene_revision": 42,
  "object_id": 7,
  "dependencies": {
    "identity_revision": 3,
    "annotation_revision": 6
  },
  "patch": {
    "name": "Ada's chair",
    "semantic_id": 91,
    "label": "reading chair",
    "description": "human verified walnut chair",
    "attributes": {
      "owner": "library",
      "tag": "favorite"
    },
    "room_memberships": [10, 12]
  },
  "timeout_ms": 1200
}
```

`base_scene_revision` 是必填项。建议携带 dependency revision，以避免覆盖其他并发修改。

## 10. 保存和查看 DSG

启用 `persistence.save_scene_graph` 后，可以手动保存：

```bash
ros2 service call /roomie/save_dsg std_srvs/srv/Trigger '{}'
```

启动 DSG Viewer：

```bash
ros2 run roomie roomie_dsg_viewer.py \
  /absolute/path/to/pipeline.yaml
```

或者直接查看导出的 JSON 和地图：

```bash
ros2 run roomie roomie_dsg_viewer.py \
  --json /path/to/latest.json \
  --points /path/to/map.nvblox \
  --port 8765
```

### 10.1 离线编辑 SceneStore

`roomie_offline_scene_editor.py` 直接加载已经停止采集的数据目录，不启动 ROS node，
也不执行 detector、tracker、DAM 或 embedding。启动前必须先停止仍在写同一个
SceneStore 的 pipeline；离线编辑器和在线 pipeline 不能同时作为 scene writer。

```bash
source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/jarvis/install/setup.bash

ros2 run roomie roomie_offline_scene_editor.py \
  /home/lindenbot/Datasets/output/genie_live/
```

默认打开 `http://127.0.0.1:8766/`，从数据目录中的
`roomie_scene.sqlite3`、`assets/assets/` 和最新 durable nvblox checkpoint 读取：

- 左栏按 id、name、label 搜索 object；中间显示采样点云和可点击的 3D OBB；
  右栏显示 id、name、label、description、几何、状态和 snapshot。
- `name` 是人工指定的实例名，OWL/Boxer 仍只产生 `label`；未人工指定时
  `name` 为空，界面显示时回退到 `label`。
- name、label 修改和 object 删除先在浏览器暂存；`Save` 才把整批修改写入
  SceneStore。删除会写 tombstone、移除关联 track/furniture/relation，object id
  不会复用。
- 修改 label 不会隐式改变家具分类。`Rebuild All Furniture` 会保存当前暂存修改，
  然后显式重建全部家具 role 和派生 relation。
- 保存会原地覆盖指定 SceneStore，不自动创建备份；写入使用 scene/component CAS，
  并在同一 SQLite transaction 中把现有地图 checkpoint 对齐到新的语义 revision，
  因而后续 coordinated load 不会回退这些离线修改。

默认最多向浏览器发送 300,000 个采样 surface voxel，点坐标和 RGB 合计约
4.5 MB；它只是显示层，不会把点云复制进 SceneStore。显存/内存受限或只需编辑
3D 框时可以关闭点云：

```bash
ros2 run roomie roomie_offline_scene_editor.py \
  /home/lindenbot/Datasets/output/genie_live/ \
  --no-points
```

也可以用 `--max-points 100000` 调低采样数，或用 `--port`、`--no-browser` 调整
服务行为。第一次打开旧数据时，SceneStore 会从 schema v5 原地迁移到 v6；v6
保留 map-invariant 语义编辑所需的 checkpoint alignment 历史。

## 11. 场景问答

### 11.1 默认持续输入 CLI

不传问题时直接进入持续输入模式：

```bash
source /home/lindenbot/miniconda3/etc/profile.d/conda.sh
conda activate jarvis
source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/jarvis/install/setup.bash

ros2 run roomie roomie_scene_qa.py
```

交互示例：

```text
Roomie Scene QA interactive mode. Type /exit or /quit to leave; each question reads the latest scene.
roomie> 厨房里有哪些物体？
{
  "reasoning": "...",
  "answer": "..."
}
roomie> 离黄色瓶子最近的物体是什么？
```

输入 `/exit`、`/quit`、`:q`，或按 `Ctrl-D`/`Ctrl-C` 退出。也可以在命令行提供
第一个问题；回答后仍会继续显示 `roomie>`：

```bash
ros2 run roomie roomie_scene_qa.py "厨房里有哪些物体？"
```

脚本或 CI 只需一次回答时显式使用 `--once`：

```bash
ros2 run roomie roomie_scene_qa.py --once "厨房里有哪些物体？"
```

CLI 默认使用 Gemini；启动时可选择 Doubao：

```bash
ros2 run roomie roomie_scene_qa.py --provider doubao
```

CLI 会复用同一个所选模型 client 和 ROS transport，但每个问题独立推理：每次回答
创建一个新的 read session，并在该回答的所有工具调用中固定同一个 scene revision；
下一问读取最新 revision。当前不会把上一问的自然语言对话历史自动带入下一问。

#### 导航寻物与视野内寻物 task profile

导航寻物使用独立 prompt 和两个有界工具，输出字段与
`roomie_msgs/action/RoomieNavigation.action` 对齐：

```bash
ros2 run roomie roomie_scene_qa.py --once --task navigation \
  "帮我去书桌拿维生素C"

# 不允许反问时，在多个目的地中选择证据最强的一个
ros2 run roomie roomie_scene_qa.py --once --task navigation \
  --no-follow-up-question "帮我拿药"
```

视野内寻物只支持 live 模式，输出字段与
`roomie_msgs/action/FindObjectInView.action` 对齐：

```bash
ros2 run roomie roomie_scene_qa.py --once --task find_object_in_view \
  "视野内有没有奥美拉唑"
```

这里的“视野内”采用几何定义：物体 yaw-OBB 的任一顶点落入采样的
`head_color` 视锥及配置深度范围。主工具读取最新 Image header、CameraInfo 和
`map <- head_color` TF，不读取该帧 RGB，也不做遮挡判断。另一个有界 VLM 工具可读取
`head_color`、`hand_left_color`、`hand_right_color` 各自最后一次发布的 2D detector
结构化结果（label、score、bbox）和带框图。`/roomie/detections_2d[_image]` 及两路
`/.../hand_*_color` 输出使用 transient-local QoS，因此问答进程晚启动时仍能取得上一张；
三路结果各自独立，不保证同一时刻，工具会返回 header、同步状态和相对 head 几何采样的
时间差。Web QA 启动时会立即启动常驻相机 worker；播放流正常时优先等待一张新图，bag
或传感器暂停超过等待时间时则使用进程内保留的最后一个完整 Image/CameraInfo/TF 视图，
并在工具结果中返回 `view_mode=cached_last_valid`、`cached=true` 和
`cache_age_sec`。该缓存没有 wall-time 过期限制，适合暂停后查询暂停画面，但不是磁盘
缓存；如果 QA worker 在 bag 已暂停后才启动且从未收到过图像，仍无法回溯此前画面。
默认深度范围为 `0.1–6.0 m`，传感器等待 1 秒，最新 TF 回退容差为 0.2 秒；
对应参数位于 `config/scene_qa/config.json`，也可用 `--in-view-*` CLI 参数覆盖。
当前数据和最后有效缓存均不可用时任务失败退出，不会伪装成“未找到”。

`roomie_agibot_live_with_qa.launch.py` 默认启动的 Web UI 也提供 `Scene QA`、
`Navigation` 和 `In-view find` 三种模式。Navigation 模式可逐次选择是否允许在多个
目的地间反问；Web 请求会按所选模式切换独立 system prompt 和有界工具集。启动时可用
`qa_default_task:=scene_qa|navigation|find_object_in_view` 指定浏览器初始模式，例如：

```bash
ros2 launch roomie roomie_agibot_live_with_qa.launch.py \
  qa_default_task:=find_object_in_view
```

离线 viewer 中 `In-view find` 会被禁用，因为它必须读取当前 Image、CameraInfo 和 TF。

### 11.2 单独启动实时 Web 问答界面

不使用组合 launch 时，可以在 pipeline 运行期间单独启动：

```bash
ros2 run roomie roomie_scene_qa_viewer.py --live
```

默认连接 `/roomie/query_scene` 并打开 `http://127.0.0.1:8776/`。可用
`--service`、`--port`、`--browser off` 调整。Web 页面会持续接收问题，并展示
实时进度与每轮场景工具调用；Gemini/Doubao 按钮决定下一问使用哪个 provider。
一次只执行一个问题；并发请求会返回 HTTP 409，
避免多个请求共享同一 live read-session transport。

live read session 在 VLM 第一次调用场景工具时才创建，TTL 固定为 300 秒，避免首轮
模型规划时间消耗 session。session、transport 和响应格式等不可恢复错误会立即终止
当前问答，不会继续交给模型反复重试。页面上的 `Reset QA` 可以丢弃当前阻塞问答并
切换到新的 agent/read-session generation；旧请求稍后返回时结果也不会进入新问答。

每次 Web viewer 启动都会在 `logs/scene_qa/` 创建
`conversation_<时间>_pid<进程号>.json`，并原子更新 `latest.json`。单个文件以交替的
`user`/`assistant` messages 保存该进程接收的全部问题；每条 assistant message 包含
provider、model、最终 reasoning/answer/raw text、耗时、progress events 和
`vlm_trace`。`vlm_trace.iterations` 按实际顺序记录每轮可见模型文本，以及工具调用的
名称、call id（provider 提供时）、参数、完整 JSON 返回和媒体摘要。日志在问题开始、
每轮 VLM 输出和每次工具返回后原子 checkpoint；失败或 Reset 的问答分别写为
`status: "error"` 或 `status: "reset"`，并保留此前完成的 iteration。日志不会保存 API key，
也不会包含模型不可见的内部 chain-of-thought。

该 JSON 是同一 viewer 进程的多轮审计记录；当前问答推理仍是逐题独立的，前一问文本
不会自动作为下一问的 VLM 上下文。

### 11.3 离线问答

静态 DSG 必须通过 `--offline-json` 显式指定，不会在 live 服务失败后静默回退：

```bash
ros2 run roomie roomie_scene_qa.py \
  --offline-json /path/to/latest.json
```

离线 Web 界面：

```bash
ros2 run roomie roomie_scene_qa_viewer.py \
  --offline-json /path/to/latest.json \
  --points /path/to/map.nvblox
```

### 11.4 场景无关的物体参考库

物体参考库保存在源码目录的 `assets/object_references/`，整个目录已被 Git 忽略；原图、
人工编辑草稿和发布结果都只留在本机。原图按物体分目录，例如：

```text
assets/object_references/
└── raw/
    └── 维生素C片/
        ├── vc1.png
        ├── vc2.png
        └── ...
```

第一次整理或增加图片后，启动离线编辑器：

```bash
cd /home/lindenbot/RealityLab/jarvis/src/roomie
source /home/lindenbot/miniconda3/etc/profile.d/conda.sh
conda activate jarvis

python scripts/roomie_object_reference_editor.py
```

完成 `colcon build` 后也可以运行
`ros2 run roomie roomie_object_reference_editor.py`。编辑器默认打开
`http://127.0.0.1:8780/`，监听地址、端口和是否打开浏览器可通过
`--host`、`--port`、`--browser` 调整。

在编辑器中需要人工完成以下内容，然后点击“发布只读版本”：

1. 为候选图片拖拽裁剪框，并逐张确认裁剪；全图合适时也必须显式确认。
2. 为一个物体恰好选择三张有区分度的图片，设置唯一的排名 1、2、3。
3. 为三张图填写视角标签，并按需填写显示名和别名。

发布会把三张裁剪图缩放到最长边不超过 768 px，并以 JPEG 质量 85 保存到不可变的
`processed/builds/<build-id>/`；元数据写入同目录的 SQLite，最后原子更新
`processed/CURRENT`。运行时会校验数据库完整性、三图约束、文件长度和 SHA-256。
这里不预存视觉 embedding，也不在本地做 OCR/分类：经过人工筛选后，每个物体只有三张
小图，按需直接交给 Gemini 或 Doubao 判断，避免维护与实际 VLM 不一致的检索模型。

Scene QA CLI 每次启动、Web viewer 每次启动和 `Reset QA` 时都会重新检查参考库：

- 健康时，system prompt 会列出精确的 reference ID，并只注册
  `inspect_object_reference` 与 `compare_scene_objects_to_reference`。
- 不存在、未发布或校验失败时，这两个工具不会注册；普通场景问答仍可使用，prompt 会明确
  参考比较不可用。

`inspect_object_reference` 只查看一个精确 reference ID 的三张参考图；
`compare_scene_objects_to_reference` 把一个 reference ID 与一个或多个明确的场景
object ID 一次交给 VLM。后者不会自动搜索候选物体，也不会代替 VLM 判定匹配。这里的
“视野内”按当前稳定场景图中的对象理解，不保证对象此刻仍位于相机画面中；需要先用已有
场景查询工具得到具体 object ID，再做参考比较。默认每次最多比较 4 个场景对象。

需要把参考库放在别处时，可设置 `ROOMIE_OBJECT_REFERENCES_ROOT`，或给 Scene QA CLI、
viewer 传 `--object-reference-root /path/to/object_references`。最大批量数可在
`config/scene_qa/config.json` 中通过 `object_reference_max_scene_objects` 调整。

## 12. 测试

```bash
cd /home/lindenbot/RealityLab/jarvis

source /opt/ros/humble/setup.bash
source /home/lindenbot/RealityLab/map_ws/install/setup.bash
source install/setup.bash

colcon test \
  --packages-select roomie \
  --event-handlers console_direct+

colcon test-result --verbose
```

如果已有对应构建目录，也可以直接运行：

```bash
cd /home/lindenbot/RealityLab/jarvis/src/roomie
ctest --test-dir build/roomie --output-on-failure
```

当前测试集包含 `test_scene_qa` 的持续 CLI、live read-session 和 Web viewer 模式回归，
以及 `test_object_references` 的人工草稿约束、不可变发布、损坏检测和两项参考工具回归。
