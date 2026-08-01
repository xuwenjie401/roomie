# Roomie 架构分析与改进建议

> 基于代码精读（2026-08-01）。本文只讨论架构层面的大改，不涉及参数调优。

---

## 1. 当前架构事实

### 1.1 线程拓扑

单 ROS 2 节点（`pipeline_node.cpp`，MultiThreadedExecutor）+ 6 个自管理线程 + 1 个 Python 子进程：

```
RosIoThread            被动。run() 只是空等 CV，实际工作全在 ROS 回调里做
                       （handleRgb/handleDepth/handleMask/handleTf 各自加同一把 mutex_）
      │ mapping_queue_(30)          │ detection_queue_(8)
      ▼                             ▼
MapThread                     DetectionBridgeThread  ← 事实上的主驱动
  integrateFrame                  1) projectPatchDepth（同步，抢 backend mutex）
  ++map_version_                  2) resize 到 960×960
  surface_cache_dirty_=true       3) enqueueRequest
                                  4) 轮询 tryPopResponse → 转发
                                          │
                                  PythonInferenceBackend（1 线程 / 1 进程 / 1 in-flight）
                                    stdin/stdout 二进制协议，一次一帧同步事务
                                          │ inference_response_queue_(8)
                                          ▼
                                  InstanceMapThread
                                    applyDetections 全程持 mutex_
                                    几何维护内联在同一线程
                                          │
                                  PublisherPersistenceThread
                                    固定 5 s sleep，每轮 debugSnapshot() 全量拷贝地图
```

三个队列全部是 `pushDropOldest`，容量 ≤ 30，**静默丢帧且无丢帧计数**。

### 1.2 数据流与时钟

系统里存在 4 个互不相关的时钟：

| 时钟 | 来源 | 周期 |
| --- | --- | --- |
| `map_version_` | 每次 `integrateFrame` 自增 | 每帧 |
| 表面缓存重建 | `surface_cache_rebuild_period_sec` | 5 s |
| 几何维护 | `instance_geometry_check_period_sec` | 2 s |
| 发布/持久化 | `publish_period_sec` | 5 s |

它们之间没有因果关系，只有"版本号不等就算脏"这一种极粗的耦合。

### 1.3 两轮工作流（重要事实）

`ObjectSnapshotRemaker::submitFrame` 的**唯一调用点**在 `applyFrozenInstanceSnapshotRemake`（`instance_map_thread.cpp:1679`），即只有 `freeze_instances=true` 才会执行。这意味着：

1. 第一轮：正常建图，产出 OBB 和 label，**snapshot 字段全空**
2. 第二轮：`freeze_instances=true` 重跑，只做 snapshot 挑选
3. 第三步：离线跑 `roomie_dsg_dam_describe.py` 生成 description
4. 第四步：离线 `roomie_offline_room_partition_ui.py` 手工画房间

`relations` 在 C++ 侧**从不生成**，只在 `snapshotFromJson` 里读入（`object_graph_io.cpp:523`）。房间和关系纯粹是手工离线产物。

这条流水线目前是"人在环上、四趟离线"，而非在线增量。

---

## 2. 五个架构层面的根因

### 2.1 几何与语义没有共同的"变更区域"抽象

`applyDetections` 里语义融合与几何维护顺序执行、共用一把锁：

```
applyDetections()                                    [instance_map_thread.cpp:1416]
  ├─ 关联 + updateTrack/createTrack                  [持 mutex_]
  ├─ mergeDuplicateStableTracks  O(T²)，changed 后重启外循环  [持 mutex_]
  ├─ 放锁 → geometrySurfaceCache()（shared_ptr，这步无锁，设计正确）
  └─ 重新加锁 → applyGeometryMaintenance             [持 mutex_]
        └─ 每个 track 遍历全部 surface_points        O(T × P)
```

`evaluateGeometryAgainstSurface`（`:414`）对每个 track 线性扫描整个点云，只有一次 AABB 粗筛，**没有任何空间索引**。P ≈ 10⁶、T ≈ 10² 时单轮约 10⁸ 次操作，且全程持锁，publisher 的 `snapshotInstances()` 只能干等。

后果不只是慢：`shouldEvaluateTrackGeometry`（`:1114`）在 `kGood` 且 OBB 没大变时直接跳过重算，于是**第一次勉强通过的 OBB 会被永久冻结**。OBB 本身只由观测加权平均得到（`updateTrack:1877`），地图几何从来没有反过来修正过 box，几何只是一个"打分/否决"的旁路，不是联合优化的一方。

### 2.2 表面缓存是全量重建，没有增量

`rebuildSurfaceCacheLocked`（`nvblox_map_backend.cpp:454`）每 5 s 扫描**全部** TSDF block 的**全部**体素，把从未变过的历史方块和刚更新的少数方块一起重算一遍。`integrateFrame` 只写了一个 `surface_cache_dirty_` 布尔，丢掉了"哪些 block 变了"这一关键信息（nvblox 本身是能给出 updated block indices 的）。

### 2.3 调度是轮询 + 挂钟限速，不是事件驱动

- 5 个线程各自 `waitPopFor(50ms)` 或 `sleep_for` 轮询
- `readyForNextRequest`（`detection_bridge_thread.cpp:285`）用挂钟对 10 fps 硬限速，超出的帧直接扔掉，既无反压也不做 coalesce
- Python 后端单 in-flight：`writeMessage` 后同步阻塞在 `readMessage`，**CPU 做预处理时 GPU 全空闲**，每帧还要通过 pipe 拷 ~2.7 MB
- 三次 IPC 失败后 `backend_disabled_after_failure_ = true` 永久禁用，无自愈
- publisher 每 5 s 的 `debugSnapshot()` 会以 `allow_periodic_rebuild=true` 触发全量重建并持 backend mutex，**直接阻塞检测线程的 `projectPatchDepth`**

这形成一条隐式优先级倒置链：可视化 → 地图 → 检测 → 几何 → 发布。

### 2.4 Snapshot 选择信息量太少，且只能离线补

质量分 = `bbox_quality × (position_score, size_score 加权)`，其中 `position_score` 是框中心的信息熵（`object_snapshot_remaker.cpp:40`）。缺失的判据包括：清晰度/模糊、曝光、**遮挡（明明有 patch-depth 投影图可用）**、视角多样性、目标完整度（虽然 `edgeCompletenessWeight` 在 observation 侧算过，但没有传进 snapshot 打分）。

每次改进写一个新 BMP（~2.7 MB），staging 目录**永不清理**；每个对象只留 1 张 `StoredBest`，DAM 因此只能看到单一视角。

### 2.5 DAM / Embedding 完全脱离在线管道

- description 由离线脚本串行生成，写到 `latest.dam_described.json`；运行时读的是 `latest.json`；房间在 `latest_manual_scene_graph.json`。**三份文件靠人工保持一致。**
- `nodeFromTrack`（`object_graph.cpp:162`）新建节点时 `description = track.label`，把描述位当 label 用；只有 `updateNodeFromTrack` 有 `if (description.empty())` 这一层弱保护。
- `ObjectSearchIndex`（`embeddings.py:34`）在首次 search 时一次性 encode 全部对象并只缓存在内存，**无磁盘持久化、无增量失效**，每次重启全量重算。
- 没有任何 provenance：不知道某条描述来自哪张 snapshot、哪个模型、哪版 prompt，因此无法判断"该不该重算"。

---

## 3. 架构改进方案

以下 6 项按依赖顺序排列，A 和 B 是地基。

### A. 统一世界状态层 + 事件总线

用一个带版本的不可变 `WorldState` 取代"三份 JSON + 多把锁"，读者拿 `shared_ptr<const>` 零拷贝读，写者发布新版本。

```cpp
struct WorldRevision {
  uint64_t geometry_version = 0;   // 来自 map_version
  uint64_t semantic_version = 0;   // 来自 frame_index_
  uint64_t snapshot_version = 0;   // submitFrame 递增
  uint64_t annotation_version = 0; // 房间/关系/描述
};

class WorldStateStore {
 public:
  std::shared_ptr<const WorldState> latest() const;      // 无锁读
  void commit(WorldStateDelta delta);                    // 单写者
  Subscription subscribe(EventMask mask, Callback cb);
};
```

事件类型取代各线程的独立定时器：

```
MapDirty        { dirty_blocks, new_map_version }
ObbChanged      { object_id, old_obb, new_obb }
GeometryVerdict { object_id, status, score }
SnapshotPicked  { object_id, image_hash, quality }
DescriptionReady{ object_id, source_snapshot_hash, model_id, text }
EmbeddingReady  { object_id, text_hash, vector }
```

要点：把"房间/关系/描述"提升为 `WorldState` 的一等公民，而不是现在这种"C++ 保管一段不透明的 `scene_graph_json` 字符串再原样回写"（`object_graph_io.cpp:491`）。离线 UI 改成向 store 提交 `annotation` delta。

收益：消掉 5 处 50 ms 轮询；publisher 变成事件订阅者不再定时全量拷贝；QA 可以直接订阅 store 而不是读静态文件；DAM/embedding 有了明确的触发信号。

### B. 空间局部化的几何–语义联合更新

**B1. 脏块传播。** 让 `integrateFrame` 返回 nvblox 已有的 updated block indices，而不是只置一个布尔：

```cpp
struct MapDirtyInfo {
  std::vector<nvblox::Index3D> dirty_blocks;
  uint64_t new_map_version;
};
MapDirtyInfo integrateFrame(const MappingFrame& frame) override;
```

**B2. 增量表面缓存。** `SurfaceBlockCache` 已经是按 block 组织的（`nvblox_map_backend.cpp:322`），只需把 `rebuildSurfaceCacheLocked` 改成"只重建 dirty_blocks 对应的条目"，其余复用。这是收益最高、改动最局部的一步。

**B3. 对象–方块覆盖索引。** 在 `InstanceMapThread` 维护双向映射：

```cpp
std::unordered_map<int, std::vector<Index3D>> blocks_by_object_;
std::unordered_map<Index3D, std::vector<int>, Index3DHash> objects_by_block_;
```

几何维护从"遍历所有 track × 所有点"变成"由 `MapDirty.dirty_blocks` 反查受影响的 object_id，只重算这些"。复杂度从 O(T × P) 降到 O(受影响对象 × 局部点数)。

**B4. 让几何真正参与 OBB 估计。** 目前几何只能给 `kGood/kBad/kEmpty` 三态否决。建议在 `evaluateGeometryAgainstSurface` 之上加一个轻量 refine：用框内近表面点做一次 PCA 或 2D 主方向拟合，产出 `suggested_obb` 与置信度，再与观测加权融合（而不是直接替换）。这才是"几何语义联合"，也顺带解决 2.1 里"OBB 被永久冻结"的问题——refine 结果本身会驱动 `obb_revision` 变化，重新打开几何复检。

**B5. 拆锁。** 把 `InstanceMapThread::mutex_` 一把大锁拆成：
- `tracks_mutex_`（关联/融合，短临界区）
- `graph_mutex_`（对外快照）

并把几何维护移出检测响应路径，做成独立的 `GeometryMaintenanceThread`，订阅 `MapDirty` + `ObbChanged` 事件。检测线程不再为几何计算买单。

另外 `mergeDuplicateStableTracks`（`:1989`）是 O(T²) 且每次 `changed` 后重启外循环（最坏 O(T³)）。有了 `objects_by_block_` 后，去重候选只需在共享 block 的对象间找。

### C. 推理调度：流水线化 + 反压

**C1. 多 in-flight + 双缓冲。** 让 Python worker 支持 N 个在途请求（协议里已有 `time_ns` + `camera_id` 可做关联键，无需改 wire format 的语义），CPU 预处理与 GPU 推理重叠。目标是 GPU 利用率而非固定 fps。

**C2. 共享内存传图。** 960×960 RGB + mask 走 `/dev/shm` 或 POSIX shm，pipe 里只传 handle。省掉每帧 ~2.7 MB 的双向拷贝。

**C3. 用反压取代挂钟限速。** `readyForNextRequest` 改为"在途请求数 < 上限"即可发送；队列满时**选择性丢帧**（保留视角变化大的、丢冗余的），并记录 `dropped_frames` 计数到 `RunLogger`。当前的静默丢帧是可观测性上的黑洞。

**C4. worker 自愈。** 去掉 `backend_disabled_after_failure_` 的终态，改成指数退避重启 + 健康探针。

**C5. 解开 publisher 与 backend 的锁竞争。** publisher 不再调用会触发重建的 `snapshot()`；改为订阅事件、消费 B2 产出的增量缓存。可视化路径永远不应该阻塞感知路径。

### D. Snapshot 选择：多视图 + 在线化

**D1. 打通在线路径。** 把 `submitFrame` 从 `freeze_instances` 分支里解耦，正常建图时也提交候选。消除"必须冻结重跑第二轮"的强制约束。

**D2. 丰富质量判据。** 现有 `bbox_quality × position × size` 之外补上：
- 清晰度：框内 Laplacian 方差
- 曝光：直方图饱和比例
- **遮挡：用已有的 patch-depth 投影图比对 OBB 前表面深度**（这个信息现在白白丢掉了）
- 完整度：把 observation 侧已算好的 `edgeCompletenessWeight` 传进来

**D3. 每对象保留 Top-K 多视图。** `StoredBest` → `StoredBestSet{K}`，并加视角多样性约束（相机位姿间夹角阈值）。DAM 拿多视图能显著改善描述质量，也为将来的多视图一致性检查留出空间。

**D4. 生命周期管理。** staging 目录做引用计数 + LRU 淘汰；只存裁剪后的 ROI 而非整帧；BMP 换 PNG/JPEG。

### E. DAM 与 Embedding：内容寻址 + 异步管道

核心是给每条派生产物一个**内容哈希**，让"是否需要重算"变成可判定的：

```cpp
struct DescriptionRecord {
  std::string text;
  std::string source_snapshot_hash;  // 输入 ROI 的哈希
  std::string model_id;              // "DAM-3B"
  std::string prompt_hash;
  uint64_t obb_revision;             // 生成时的几何版本
  TimeNanoseconds generated_ns;
};
struct EmbeddingRecord {
  std::vector<float> vector;
  std::string text_hash;             // 对应 description 的哈希
  std::string model_id;
};
```

失效规则明确：`snapshot_hash` 变了 → 重算描述；`text_hash` 变了 → 重算 embedding；`model_id`/`prompt_hash` 变了 → 全量重算。

**E1. 描述服务化。** 把 DAM 从"离线批脚本"改成常驻服务（ROS action 或独立进程 + 事件订阅），订阅 `SnapshotPicked`，按对象优先级（新对象 > 高质量 snapshot 更新 > 陈旧描述）排队。批量凑够 batch 再进 GPU。

**E2. Embedding 持久化 + 增量。** 落盘为 `object_id → (vector, text_hash, model_id)`，启动时只对 `text_hash` 不匹配的对象重编码。当前每次重启全量 encode 是纯浪费。

**E3. 描述位与 label 分离。** `nodeFromTrack` 里 `node.description = track.label` 应删掉，`description` 为空就是空，语义查询侧做 fallback。现在这个赋值让"有没有真描述"变得无法判断。

**E4. 三份 JSON 合一。** 由 A 的 `WorldState` 统一持有 objects / relations / rooms / descriptions / embeddings，各自带版本。离线 UI 和 DAM 都改为提交 delta。

### F. 可观测性

当前 `RunLogger` 记了大量 `*_ms` 计时，但缺三类关键指标：

1. **丢帧**：每个队列的 drop 计数与原因（rate limit / queue full / low coverage）
2. **锁等待**：几何维护持锁时长、publisher 等锁时长
3. **陈旧度**：`geometry_version` 与 `map_version` 的差、description 相对 snapshot 的滞后

建议按 `WorldRevision` 打一条结构化的 per-cycle 记录，使"某对象的描述比它的几何落后多少个版本"可以直接查询。

---

## 4. 建议实施顺序

| 阶段 | 内容 | 依赖 | 主要收益 |
| --- | --- | --- | --- |
| 1 | B1 + B2 增量表面缓存 | 无 | 立竿见影的延迟下降，改动局部 |
| 2 | C5 解开 publisher/backend 锁竞争 | 无 | 消除可视化阻塞感知 |
| 3 | F 可观测性（丢帧/锁/陈旧度） | 无 | 后续所有优化的度量基础 |
| 4 | B3 + B5 覆盖索引 + 拆锁 + 独立几何线程 | 1 | 几何维护退出关键路径 |
| 5 | A 世界状态层 + 事件总线 | 3, 4 | 为 D/E 提供触发机制 |
| 6 | C1–C4 推理流水线化 | 3 | GPU 利用率 |
| 7 | D1–D4 snapshot 在线化 + 多视图 | 5 | 消除两轮工作流 |
| 8 | E1–E4 DAM/embedding 服务化 | 5, 7 | 描述与检索自动跟随 |
| 9 | B4 几何参与 OBB refine | 4 | 真正的几何语义联合 |

前 3 项互相独立、都不需要大改接口，适合先做以建立基线。

---

## 5. 需要留意的取舍

- **B4（几何 refine OBB）有回归风险。** 地图本身有噪声，让几何直接改 box 可能引入抖动。建议先只输出 `suggested_obb` 并记录与观测融合结果的差异，观察一段时间再启用。
- **A 的改动面最大。** `PipelineConfig` 已有 145 个字段，`WorldState` 若设计不当会变成又一个巨型结构。建议按 geometry / semantic / annotation / derived 四段切分，各自独立版本号。
- **C1 多 in-flight 会改变检测时序语义。** `InstanceMapThread` 目前隐含假设响应大致按时间有序（`ageUnmatchedTracks` 依赖这一点）。乱序到达需要显式的时间戳重排缓冲。
- **D3 多视图会放大存储压力。** 必须与 D4 的生命周期管理同时上线。

---

## 6. 附：关键代码位置索引

| 关注点 | 位置 |
| --- | --- |
| 语义融合主流程 | `src/instance_map_thread.cpp:1416` |
| 几何评估（O(T×P) 热点） | `src/instance_map_thread.cpp:414` |
| 几何复检门控（OBB 冻结来源） | `src/instance_map_thread.cpp:1114` |
| OBB 加权更新 | `src/instance_map_thread.cpp:1877` |
| 重复轨迹合并 O(T²) | `src/instance_map_thread.cpp:1989` |
| 表面缓存全量重建 | `src/nvblox_map_backend.cpp:454` |
| map_version 自增点 | `src/nvblox_map_backend.cpp:205` |
| 挂钟限速 | `src/detection_bridge_thread.cpp:285` |
| 同步 IPC 事务 | `src/python_inference_backend.cpp:352` |
| publisher 触发全量重建 | `src/publisher_persistence_thread.cpp:93` |
| snapshot 质量评分 | `src/dsg/object_snapshot_remaker.cpp:369` |
| snapshot 唯一调用点（冻结分支） | `src/instance_map_thread.cpp:1679` |
| description = label 赋值 | `src/dsg/object_graph.cpp:162` |
| scene_graph 信封不透明回写 | `src/dsg/object_graph_io.cpp:491` |
| embedding 惰性全量编码 | `scripts/scene_qa/embeddings.py:138` |

