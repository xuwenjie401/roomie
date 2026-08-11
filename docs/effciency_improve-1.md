结论：除了 batching，最值得做的不是“多开几个 worker”，而是：

1. 隔离 DAM 对 detection GPU 的干扰。
2. 用多相机 freshness-aware 调度，避免处理已经过时或信息重复的帧。
3. 把 OWL 和 BoxerNet 的双视觉 backbone 合并或做级联。
4. 再做低分辨率、GPU NMS、单次 H2D、`torch.compile`/TensorRT 等单请求优化。

这几项叠加，收益可能明显大于 batching。

## 1. 当前最大嫌疑：DAM 正在抢 detection 的 GPU

日志中稳态 worker 平均是 122.53 ms：

- OWL：62.52 ms
- BoxerNet：38.07 ms
- preprocess：15.49 ms

但我按时间窗口重新统计后发现：

- 首段 130 个稳态请求平均只有约 73.1 ms。
- 相近检测数量的 10 秒窗口，worker 平均可以从 66.6 ms 变成 215.8 ms。
- 例如平均每帧约 2.9/3.1 个检测，耗时却相差 3.2 倍，无法用候选框数量解释。
- 有些平均只有约 1.1 个框的窗口，worker 仍从 76.7 ms 跳到 154.8 ms。

同时，有 37 个 DAM-3B 任务在有效传感器窗口内完成。当前 DAM 的 admission 只在任务启动前检查：

```cpp
detection_queue_.empty() &&
python_backend_.idle() &&
inference_response_queue_.empty()
```

见 [pipeline.cpp](/home/lindenbot/RealityLab/jarvis/src/roomie/src/pipeline.cpp:1193)。但一旦 DAM 启动，后面到达的 detection 无法抢占它；而 DAM 默认也使用 CUDA。

所以这里有一个很强的推断：DAM 在短暂 detection 空档中启动，然后与后续 OWL/BoxerNet 并发争用 GPU，制造了当前的大量长尾。需要一次 `DAM off/on` A/B 来最终确认，但现有证据已经很明显。

优先方案：

- 最佳：DAM 放第二块 GPU、远端机器，或者录制/操作结束后再跑。
- 同一 GPU 时：引入统一 GPU arbiter，实时 detection 拥有保留预算；持续相机输入期间不启动长 DAM。
- DAM 任务只为稳定、确认后的 track 生成，并合并同一对象频繁的 appearance 更新。
- CUDA MPS 可以限制低优先级进程可用的执行资源，但它更适合软隔离，不能替代真正的实时优先级和设备隔离。[NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/latest/index.html)

如果把平均 worker 从 122.5 ms 恢复到日志中已经出现过的约 73 ms，单 worker 的理论服务率会从约 8.2 提升到约 13.7 req/s，约 1.68 倍。这可能是目前投入产出最高的一项。

## 2. 三相机不能继续使用现在的普通 latest-by-key 队列

当前 detection queue 容量是 2，而未来有 `head/left/right` 三个 key。满队列时，新 key 会淘汰最旧的其他 key，[pipeline.cpp](/home/lindenbot/RealityLab/jarvis/src/roomie/src/pipeline.cpp:507)。这不保证三路公平，某一路可能长期被另外两路挤掉。

更合理的结构是：

- 每个 camera 一个容量为 1 的 latest mailbox，只保留最新帧。
- 全局调度器从三个 mailbox 中选择下一帧。
- 使用“加权公平 + deadline/age boost”，而不是 FIFO。
- 每路有最低保障频率，活动手相机获得临时高权重。
- 根据每路 EWMA 推理耗时预测完成时间；预计无法在 freshness deadline 内完成的帧直接丢弃。
- 当前 `perception.deadline_ms=10000` 对 10 Hz 实时图像太宽松，应改成相机级 freshness budget。

建议调度策略：

- 非操作状态：head 保持主要份额，左右手低频巡检。
- 操作状态：正在工作的手相机进入 burst 模式，另一只手和 head 保留最低配额。
- 不抢占已经运行的 CUDA kernel，但允许在 OWL 与 BoxerNet 之间取消已过时请求。
- 记录每个 camera 的完成 FPS、结果图像年龄、P95 延迟、supersede 数量和最长无服务时间。

这不会直接缩短一次模型推理，但会显著提高“有效结果/单位 GPU 时间”。

## 3. 不要让 10 Hz 输入等于 10 Hz 全链路推理

手部相机特别适合 temporal cascade：

- 10 Hz 接收图像并运行廉价的运动检测/跟踪。
- OWL 只在新物体出现、跟踪置信度下降、画面明显变化或抓取事件附近运行。
- 已确认物体使用 2D tracker、地图投影和 instance map 更新。
- BoxerNet 只处理新物体、几何不确定物体或需要刷新 OBB 的 track。
- 稳定场景下左右手可以交替检测，而不是同时满速。

例如每路仍输出 10 Hz 跟踪结果，但完整 OWL+BoxerNet 只运行 3 Hz，重推理请求数可以直接减少约 70%。这通常比把单次推理优化 20% 更有价值。

还可以利用机器人状态触发：

- 手臂或夹爪静止：低频。
- 接近目标、抓取、放置：活动手相机短时升到最高优先级。
- 新目标进入工作空间或跟踪丢失：立即重检测。
- head 与 hand 看到同一已确认对象时，选择预计信息增益更高的一路。

## 4. 长期最大单请求优化：去掉双 backbone

当前每帧实际上跑了两个完整的 960×960 视觉编码器：

- OWLv2 vision backbone 做 2D detection。
- BoxerNet 内部又跑一次 DINOv3 backbone 做 3D OBB。

OWL 已经占 worker 平均时间约 51%。而当前 prompt 文件是固定的 104 类，不是真正每帧动态变化的开放词表。

长期可以考虑：

- 在 BoxerNet 的 DINO 特征上训练一个轻量 2D detection head。
- 先跑一次 DINO，得到 2D boxes，再直接交给 Boxer query head。
- OWL 变成低频开放集 fallback，只用于未知物体、类别扩展或周期性校正。
- 或为手部相机训练较小的固定类别 detector，head 继续保留 OWL。

这会把“OWL backbone + DINO backbone”变成“一次 backbone + 两个 head”。开发和数据成本较高，但它是最可能接近砍掉几十毫秒的架构级优化。

## 5. 单请求执行路径仍有不少空间

### 5.1 手部相机降低输入分辨率

两套模型现在都处理 960×960。手部相机通常目标更近、更大，可以单独尝试 768 或 640，head 保持 960。

从 960 降到 768：

- token 数量约变为原来的 64%。
- attention 部分的理论计算量下降更明显。
- 实际模型延迟可能降低约 30%–50%，但 OWL 位置编码和 BoxerNet 训练分辨率需要适配，并验证小物体 recall 和 3D OBB。

还可以做负载自适应：GPU 空闲用 960，拥塞时 hand 降到 768。

### 5.2 GPU NMS 和候选上限

当前 OWL 会较早把 boxes/scores 搬回 CPU，并使用 Python 循环执行逐类别 NMS，[owl_wrapper.py](/home/lindenbot/RealityLab/boxer/owl/owl_wrapper.py:390)。这是明显的长尾来源。

建议：

- threshold、top-K、batched NMS 全部留在 GPU。
- NMS 后只把最终几十个结果传回 CPU。
- BoxerNet 前设置 per-class/global top-K。
- 记录 pre-NMS 数量；当前日志只有 post-NMS 平均 6.43、最大 32，无法看出 NMS 的真实输入规模。

这主要优化 clutter 场景和 P95，不一定大幅降低纯模型核心时间。

### 5.3 图像只上传 GPU 一次

目前路径存在：

- pipe 解码后 `numpy.copy()`。
- CPU 上 uint8→float、除法和乘法。
- OWL 将图像搬到 GPU。
- BoxerNet 又把同一图像搬到 GPU。

见 [roomie_python_inference_worker.py](/home/lindenbot/RealityLab/jarvis/src/roomie/scripts/roomie_python_inference_worker.py:490)。

可以改为：

- 固定的 pinned-memory 输入槽。
- uint8 图像一次异步 H2D。
- OWL 和 DINO 的 normalization 都在 GPU 上从同一张 resident tensor 生成。
- mean/std、intrinsics 等相机常量常驻 GPU。
- 960→960 时跳过无意义的 interpolate、clone 和全图 max/min 检查。

PyTorch 官方也建议用 pinned memory 和 `non_blocking=True` 重叠 H2D 与计算。[PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)

### 5.4 编译 BoxerNet，而不只是 OWL

目前 OWL vision detector 已经 `torch.compile`，文本 embedding 也已缓存；BoxerNet/DINO 主路径没有整体 compile。BoxerNet 虽然已经 BF16 autocast，但仍可尝试：

- `torch.compile(mode="max-autotune")`
- 固定 top-K/padding，让 Boxer query 形状稳定。
- CUDA Graph replay。
- Torch-TensorRT/TensorRT BF16/FP16 engine，之后再评估 INT8/FP8。

`torch.compile` 的 `reduce-overhead`/`max-autotune` 模式支持通过 CUDA Graph 降低小 batch 的 launch overhead，[PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)；TensorRT 官方也建议通过 per-layer profiling、CUDA Graph 和多 stream 判断实际收益。[TensorRT best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)

### 5.5 非 batching 的流水线并行

可以把 worker 拆成有界阶段：

1. Decode/H2D
2. OWL
3. filter/track association
4. optional BoxerNet
5. response

让 frame N 的 BoxerNet 与 frame N+1 的 H2D/预处理重叠；如果 Nsight 显示单个模型没有占满 GPU，也可以测试 OWL(N+1) 与 Boxer(N) 的不同 CUDA stream 并行。

当前多处 `torch.cuda.synchronize()` 是 device-wide barrier，[worker](/home/lindenbot/RealityLab/jarvis/src/roomie/scripts/roomie_python_inference_worker.py:502)。串行计时时没问题，但会阻止流水线重叠。应改成 CUDA events 做阶段计时，只在 CPU 真正读取结果时同步。

多 stream 不一定必然加速：如果两个 transformer 已经占满 Tensor Core，反而会增加长尾，所以必须先用 Nsight Systems 看 kernel occupancy。

## 6. 不建议优先投入的方向

- 只增加 queue 深度：会增加陈旧结果，不增加算力。
- 同一 GPU 直接启动三个 Python worker：很可能复现 DAM 导致的争用和长尾，并重复占用模型显存。
- 先重写 IPC：稳态 serialization 约 0.63 ms、pipe write 约 1.93 ms，IPC 与 worker 的总差约 4.6 ms，不是主瓶颈。
- 优化 map projection/resize：平均分别只有 1.39 ms 和 0.98 ms。
- 继续优化文本 embedding：OWL 已在初始化时缓存，运行时不是瓶颈。
- 单纯切 BF16：当前 RTX 5090 上 OWL 和 Boxer 已经走 BF16。

## 推荐落地顺序

1. 同一个 bag 做 `DAM off/on` 三轮 A/B，同时记录 GPU utilization、clock、power 和每阶段 CUDA event。
2. 将 DAM 移出实时 GPU，重新测纯 detection 饱和吞吐；不要再用 5.39 FPS 这个输入受限结果估算上限。
3. 改为三路 latest mailbox + 加权 deadline 调度，做 3×10 Hz 压测。
4. 加 temporal tracking、稳定 track 跳过 Boxer、活动手 burst 调度。
5. 做单次 H2D、GPU NMS、BoxerNet compile。
6. 测试 hand 768/640。
7. 长期评估共享 DINO backbone 的 2D head，OWL 退为 fallback。

按照当前证据，我预计“DAM 隔离 + freshness 调度 + temporal cascade”会比单独 batching 更先带来明显收益；而要稳定做到三路各约 5 FPS，长期还需要降低每帧双 backbone 的成本。