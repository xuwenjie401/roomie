# Roomie for Robots

## Hand Image
当前 Roomie 仍只用 `head_color + head_depth` 更新 TSDF；检测链路则可以从任意已知
内外参的 rectified RGB 相机出发，将当前地图投影成 Boxer 的 depth patch。因此左右手相机
应作为 detection-only camera 接入，不参与 mapping，也不需要手部 depth。

### 示例 bag 的输入事实

示例数据：`/home/lindenbot/Datasets/roomie/bags/manip_0808_2`

| Topic | 类型/格式 | 尺寸 | 帧数 | 约频率 |
| --- | --- | ---: | ---: | ---: |
| `/live_connect/hand_left_color` | `sensor_msgs/Image`, `bgr8` | 1280×1056 | 1262 | 9.9 Hz |
| `/live_connect/hand_right_color` | `sensor_msgs/Image`, `bgr8` | 1280×1056 | 1258 | 9.9 Hz |

这两路消息的 `header.frame_id` 为空，图像仍带原始手相机畸变；bag 的 `/tf_static` 只有
头相机外参，没有 `arm_l_end_link -> hand_left_color` 和
`arm_r_end_link -> hand_right_color`。所以不能直接把原 topic remap 给 detection。

### 接入后的数据流

```text
/live_connect/hand_{left,right}_color
  -> roomie_agibot_hand_color_adapter
       - 默认 disabled；仅在 /roomie/hand_cameras/enabled=true 时处理图像
       - 用 G2 原始畸变参数去畸变
       - 输出 rectified bgr8 + zero-distortion CameraInfo
       - 发布 mount frame -> hand camera 的静态 TF
  -> /roomie/input/hand_{left,right}_color/image_rect
  -> RosIoThread (detection-only, per-camera mask/TF/intrinsics cache)
  -> latest TSDF surface 投影得到 PatchDepth
  -> OWLv2 + BoxerNet
  -> 共享的 instance association / SceneStore
```

对应配置为：

```yaml
detection:
  additional_camera_ids: [hand_left_color, hand_right_color]
```

camera id 会在 `config/robots/G2/cameras.yaml` 中解析出规范输出 topic、frame 和 pinhole
内参。两路手相机设置为 `enable_mapping=false, enable_detection=true`。检测队列容量至少
覆盖当前三路 camera id，并按 camera id 保留每路最新帧；不设置头相机优先级。

### 导航构型状态开关

左右手相机共用一个 `navigation_posture_deviated` 开关。相对于
`config/robots/G2/default_navigation_posture.yaml`，以下任一条件成立都会同时允许两路手图
进入 detection：

- `body_link5` 离开导航基准；
- 任一 `arm_l_link1..7` 离开导航基准；
- 任一 `arm_r_link1..7` 离开导航基准。

默认构型集中写在上述单个 YAML 中，明确不包含任何 `head_link*`，所以导航时转头不会开启
手图。判断依据是相对默认构型的位姿差而非速度：手臂移动后即使静止在操作姿态，手图仍
保持开启。进入偏离状态需持续 `0.10 s`；全部 link 回到退出阈值内并持续 `0.50 s` 后关闭，
避免临界值附近频繁开关。头相机继续使用原有策略，不受该开关影响。

pipeline 通过 transient-local topic `/roomie/hand_cameras/enabled` 发布这一状态。手相机
adapter 默认关闭，disabled 时在 `cv2.remap` 前直接返回，不再发布 rectified 图像；C++
对 detection-only 相机另有一次 decode/TF/mask 之前的早期 admission，防止其他发布源绕过
adapter 后重新占用 head pipeline 的处理时间。手图 enable 后三路相机共享串行 detection
backend，允许 head 的待处理旧帧被更新或丢弃。

### 用示例 bag 运行

先重新编译并加载 workspace：

```bash
cd /home/lindenbot/RealityLab/jarvis
colcon build --packages-select roomie --symlink-install
source install/setup.bash
```

启动 pipeline（`roomie_agibot_live.launch.py` 已把左右手输入指向 `/live_connect/*`）：

```bash
ros2 launch roomie roomie_agibot_live.launch.py \
  enable_boxer:=true \
  enable_hand_cameras:=true \
  boxer_max_inference_fps:=5.0
```

另一个终端播放：

```bash
ros2 bag play /home/lindenbot/Datasets/roomie/bags/manip_0808_2
```

可用以下命令确认接入：

```bash
ros2 topic hz /roomie/input/hand_left_color/image_rect
ros2 topic hz /roomie/input/hand_right_color/image_rect
ros2 run tf2_ros tf2_echo map hand_left_color
ros2 run tf2_ros tf2_echo map hand_right_color
```

### 限制与调参

- 这里的 hand camera 3D 依据仍是头部 RGB-D 建出的 TSDF。如果手相机看到的是地图未覆盖、
  刚被拿起或正在运动的物体，可能因为 `min_patch_coverage_ratio` 不足而跳过，或得到不可靠
  的 3D box；这不是增加手部 depth。
- `detection.max_inference_fps` 当前是每相机入口上限，但 Python worker 仍是单路串行 GPU
  事务。三路相机同时启用时，建议从 2–5 FPS 起测，并观察 detection/inference queue 日志。
- robot-state gate 仍会在底盘旋转或相关状态未知时丢弃感知帧。姿态开关只控制手相机进入
  detection；ROS subscription 保持存在，但 disabled 时不做整流和 rectified 图像发布，也
  不会让手相机参与 TSDF mapping。
