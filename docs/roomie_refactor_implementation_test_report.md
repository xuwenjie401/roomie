# Roomie 重构实现与测试交接报告

日期：2026-08-04
工作分支：`develop/g2_alpha`
重构前基线提交：`f95c2c6b4c8533247c435e94aa795ca363c6f42a`
输入文档：`docs/roomie_arch_discussion.md`、`docs/roomie_arch_coding_plan.md`

## 1. 本轮结论

本轮已完成 Roomie 主干的重构级实现，并继续完成了上一版报告列出的两个 P0 恢复/SLO 问题。最终 NVBLOX 配置构建成功，仓库 CTest 36/36 通过，独立 Python 测试 32/32 通过，RTX 5090 上的 NVBLOX conformance 测试通过；mapping-only 真实 bag、graceful checkpoint 和实际 checkpoint 恢复 smoke test 也已执行。后续补充的 Doubao/Gemini Web 切换和“checkpoint 已加载但启动 surface 未发布”问题也已实现、安装和验证。

本轮实现覆盖 Coding Plan 的 Gate A～E 主链路：可观测因果链、确定性的含当前帧地图、单写者 SceneState、持久化语义制品，以及 live query/mutation。代码当前保留为未提交工作树，供下一位 agent 审查、拆分提交和继续真实数据验收。

这里的“本轮完成”指代码实现、自动化测试、mapping-only 真实数据主链和单图 DAM-3B 真模型 smoke 收口，不等同于完整语义生产验收。开启 Boxer、snapshot、DAM、embedding 的完整语义 bag 仍未执行。

## 2. 已实现内容

### 2.1 Gate A：provenance、IPC 和有业务语义的背压

- 为 frame、map、surface、scene、artifact 引入显式版本和来源标识，避免只靠时间戳推断因果关系。
- Python inference 使用带 request id、边界检查、超时和明确错误分类的 framed IPC；worker 异常、关闭和超时均有有界处理。
- `BoundedChannel` 支持按用途选择 reject、replace、supersede、deadline 等策略，并暴露统计量。
- 检测链路从同步调用改为请求/响应异步协调，支持旧请求 supersede，且关闭时能中断挂起 worker。

主要入口：

- `include/roomie/pipeline/types.hpp`
- `include/roomie/pipeline/thread_safe_queue.hpp`
- `src/detection_bridge_thread.cpp`
- `src/python_inference_backend.cpp`
- `scripts/roomie_python_inference_worker.py`

### 2.2 Gate B：不可变 SurfaceSnapshot 和精确 include-current barrier

- map actor 独占地图变更和 surface refresh；publisher 只读取已发布的不可变 snapshot。
- CPU 与 NVBLOX backend 统一到版本化 map/surface 契约。
- NVBLOX 增加 dirty-block 增量 surface cache 更新，同时保留必要的 full rebuild 路径。
- `FrameBundle` 与 `MapCommit` 通过精确版本关联，不再用“当前最新地图”近似代替“包含当前帧的地图”。
- barrier 明确处理等待、deadline、eviction、not-joined 和 supersede，避免无限等待。

主要入口：

- `include/roomie/pipeline/surface_snapshot.hpp`
- `include/roomie/pipeline/map_backend.hpp`
- `src/map_thread.cpp`
- `src/cpu_point_map_backend.cpp`
- `src/nvblox_map_backend.cpp`

### 2.3 Gate C：单写者 SceneState、异步 geometry 与 SceneStore

- 新建 typed `SceneCommand`、`SceneEvent`、`SceneState`、`SceneSnapshot` 和 `SceneReducer`。
- reducer 是持久 scene 内容的唯一写者；geometry worker 只计算结果，通过 dependency token 和 CAS 再提交。
- geometry scheduler 支持去重、失效、重试和 supersede；旧结果不会覆盖新 revision。
- SQLite `SceneStore` 保存 scene checkpoint/delta、artifact task/outbox、mutation ledger、durability watermark 和 map checkpoint manifest。
- persistence actor 从实时 reducer 解耦，具有 soft/hard lag、terminal failure 和有界 shutdown 行为。
- `restoreAt(revision)` 已实现，可从有界 checkpoint/delta 链恢复指定 revision。
- SceneStore schema 升级到 v5，新增 durable artifact origin、map checkpoint frame/config 指纹，以及 startup-only 原子 `rewindTo(revision)`。
- coordinated map 恢复会把 durable scene/task/embedding/origin 一起回退到 manifest 的 aligned revision；显式 seed mode 则清除不相容的 durable scene-derived state，不再混合任意 map 与 semantic suffix。
- 对象 observation 明细 history 默认保留最近 256 条，同时保留 all-time 聚合；SemanticIndex record history 默认保留最近 1024 条。
- snapshot control 使用容量 64 的 bounded/coalescing queue。同 owner/kind 的新 revision 合并；容量拒绝触发 terminal fault，关闭后的拒绝不误触发 fault。
- `waitUntilIdle(timeout)` 和 pipeline 分阶段 drain 均有超时，避免关闭无限等待。

主要入口：

- `include/roomie/scene/`、`src/scene/`
- `include/roomie/dsg/observation_history.hpp`
- `include/roomie/pipeline/snapshot_control_queue.hpp`
- `src/instance_map_thread.cpp`
- `src/pipeline.cpp`

### 2.4 Gate D：在线 snapshot、DAM、embedding 和语义索引

- `AssetStore` 使用内容寻址的不可变资产，提供引用计数、pin、校验和 GC。
- `SnapshotBank` 在线选择 top-k snapshot，记录视角/尺度多样性和质量，避免只保留近重复帧。
- DAM 和 embedding 均使用 durable task/outbox、lease/retry/backoff 和幂等 task id。
- 联合 artifact 调度采用一次原子多类型 lease，优先级为：已逾期任务、interactive 任务、普通到期任务，再按 scene revision/task id 稳定排序；DAM 和 embedding 使用相同公平性语义。
- DAM 输出支持 schema-valid、schema-repaired 和 unstructured fallback 三种明确解析路径。
- embedding 采用 versioned index；查询可过滤陈旧 generation，scene 变更会使旧描述/embedding 失效。
- SLO payload 同时记录 durable Unix 时间和进程 epoch 隔离的 steady 时间。当前进程内优先使用 steady clock，跨进程恢复时退回 Unix clock，并记录 measurement clock、elapsed 和 violation。
- stable-object SLO origin/priority 与 owning scene revision 原子持久化；首次 snapshot 前重启不会重置 Unix 起点，跨进程也不会复用旧进程 steady tuple。
- live object origin 不再受 replay memo 容量淘汰；burst 分类改用 steady clock；memo 淘汰后的 replay 会查询 durable task，保持 payload 字节级幂等。
- DAM/SceneStore durable JSON 整数统一做类型、负数和溢出检查；embedding SLO 已计数 task 历史改为可配置有界集合。

主要入口：

- `include/roomie/artifacts/`、`src/artifacts/`
- `scripts/roomie_python_dam_worker.py`
- `scripts/roomie_sentence_transformer_worker.py`

### 2.5 直接使用 NVIDIA 官方 DAM 源码

实现不依赖 DAAAM。Roomie worker 直接把官方 `NVlabs/describe-anything` checkout 加入 Python import path，并调用官方 `dam.describe_anything_model.DescribeAnythingModel` API。

本机配置如下：

| 项目 | 值 |
| --- | --- |
| 官方源码 | `/home/lindenbot/RealityLab/describe-anything` |
| Git remote | `https://github.com/NVlabs/describe-anything.git` |
| 源码 commit | `153ad3d33c29324e9197f565547c6bc8500da02d` |
| 本地权重 | `/home/lindenbot/hugging_face/DAM-3B`（6.7G） |
| Python 环境 | `/home/lindenbot/RealityLab/.venvs/roomie-dam/bin/python` |
| Roomie worker | `scripts/roomie_python_dam_worker.py` |

配置已同步到 `config/pipeline.yaml`、`config/pipeline_nvblox.yaml` 和 `config/pipeline_agibot_head_mapping.yaml`。仓库源码（排除构建产物和 pycache）中 `DAAAM` 引用数为 0。

对应官方资料：

- [NVlabs/describe-anything](https://github.com/NVlabs/describe-anything)
- [官方最小 demo](https://github.com/NVlabs/describe-anything/blob/main/demo_simple.py)
- [DescribeAnythingModel 实现](https://github.com/NVlabs/describe-anything/blob/main/dam/describe_anything_model.py)
- [官方依赖声明](https://github.com/NVlabs/describe-anything/blob/main/pyproject.toml)

### 2.6 Gate E：live query、mutation 和 schema v3

- 新增 `QueryScene.srv`、`MutateScene.srv` 及 JSON adapter。
- query server 提供容量 128、固定 TTL 的 bounded read session；session id 不透明，同一 session 固定 scene revision。
- unknown、expired、stale 和 capacity 状态均显式返回，不静默切换 revision。
- Scene QA 默认访问 live `/roomie/query_scene`；静态 JSON 仅通过 `--offline-json` 显式启用。
- 一次模型 answer 只创建一次 read session，所有 tool call 复用同一 session/revision。
- 针对 ROS Humble Python 3.10 与 Gemini 环境 Python 3.11/3.13 ABI 不一致，增加系统 Python subprocess adapter。
- mutation gateway 支持 expected revision、幂等 ledger、durability 等待和 terminal durability fuse。
- room、object、relation 使用 canonical schema v3；room containment 由 canonical room/geometry 派生，旧 schema 可兼容读取。

主要入口：

- `include/roomie/query/`、`src/query/`
- `srv/QueryScene.srv`、`srv/MutateScene.srv`
- `scripts/scene_qa/live_query.py`
- `scripts/scene_qa/_ros_query_worker.py`
- `scripts/scene_qa/gemini_agent.py`
- `scripts/roomie_scene_qa.py`

### 2.7 持续 CLI 与 mapping/RViz/live QA 组合启动

- `roomie_scene_qa.py` 默认进入持续输入模式，可选位置参数作为第一问；`/exit`、`/quit`、`:q`、Ctrl-D 和 Ctrl-C 均可退出。
- 同一 CLI 进程复用所选 provider client 和 ROS transport；每一问仍创建新的固定 revision read session。脚本调用通过 `--once` 保留单次、机器可读 JSON 输出语义。
- `roomie_scene_qa_viewer.py` 默认改为 live `/roomie/query_scene`，静态 DSG 只能通过 `--offline-json` 显式启用。
- live Web UI 实时轮询所选 provider 的规划状态和工具进度；回答完成后展示工具名、参数、完整响应、对象候选和显式 reasoning 摘要。模型内部隐藏 chain-of-thought 不暴露。
- 新增 `roomie_agibot_head_mapping_qa.launch.py`，组合原 mapping launch、RViz 和 Web QA；默认启用完整 Boxer/snapshot/DAM/embedding 链，并用独立 `qa_python` 运行 Gemini 环境。
- Web 后端串行化 QA 请求；同一 transport 上存在未完成请求时返回 HTTP 409，避免跨请求 read-session 状态互相覆盖。
- Web 页面新增 Gemini/Doubao 按钮，选择随每个 `/ask` 请求发送；两个 agent 使用各自独立的工具 registry/read session，缺 key 或依赖时只禁用对应 provider。
- 新增 `DoubaoSceneQaAgent`，按火山方舟 Responses API 的 `function_call`、`call_id`、`function_call_output` 和 `previous_response_id` 协议执行本地场景工具；截图证据以多模态 data URL 回传。HTTP 使用 Python 标准库，不新增 OpenAI/Volcengine SDK 依赖。
- CLI 增加 `--provider doubao`，Web/launch 增加 `qa_default_provider`；配置新增 `doubao_model` 和 `doubao_base_url`。

主要入口：

- `launch/roomie_agibot_head_mapping_qa.launch.py`
- `scripts/roomie_scene_qa.py`
- `scripts/roomie_scene_qa_viewer.py`
- `scripts/scene_qa/doubao_agent.py`
- `docs/roomie_usage.md`

## 3. 最终测试结果

### 3.1 NVBLOX 配置全量构建和 CTest

构建缓存确认 `ROOMIE_ENABLE_NVBLOX:BOOL=ON`。最终命令：

```bash
cd /home/lindenbot/RealityLab/jarvis
colcon build --packages-select roomie --symlink-install \
  --cmake-args -DROOMIE_ENABLE_NVBLOX=ON
source /opt/ros/humble/setup.bash
ctest --test-dir build/roomie --output-on-failure
```

结果：

| 项目 | 结果 |
| --- | --- |
| CMake build | PASS，`roomie_pipeline_node` 及全部测试目标构建成功 |
| CTest | 36/36 PASS，0 failed |
| GTest 目标 | 29 个 |
| Pytest 目标 | 7 个 |
| GPU/NVBLOX | 1 个，PASS，1.65 s |
| CTest 总耗时 | 10.78 s |

覆盖的关键类别包括 bounded channel、版本类型、IPC wire/protocol、Python worker 超时/崩溃、surface snapshot、map causal barrier、scene reducer、geometry scheduler/worker、SnapshotBank、artifact/embedding actor、semantic index、query/mutation、SceneStore、persistence、长期运行边界、AgiBot adapter、bag harness 和 NVBLOX backend conformance。

全量测试第一次运行时只有 `test_query_scene_ros_typesupport` 因 CTest 未注入 build-tree `PYTHONPATH` 而失败；已在 `CMakeLists.txt` 为该测试加入 `rosidl_generator_py` 路径。加入 Scene QA 持续交互/UI 回归后，全量 36/36 通过。

### 3.2 独立 Python 测试

最终命令：

```bash
source /opt/ros/humble/setup.bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/build/roomie/rosidl_generator_py${PYTHONPATH:+:$PYTHONPATH}" \
/usr/bin/python3 -m pytest -q \
  test/test_agibot_head_rgbd_adapter.py \
  test/test_bag_replay_harness.py \
  test/test_dam_worker_official_api.py \
  test/test_embedding_worker_protocol.py \
  test/test_inference_protocol.py \
  test/test_query_scene_ros_typesupport.py \
  test/test_scene_qa.py
```

结果：32/32 PASS，0.44 s。该结果与 CTest 的 Python 项有重叠；Scene QA 现已正式加入 CTest，并新增默认持续输入、`--once`、viewer 默认 live 模式、live-session、Gemini fake client、Doubao Responses/function-call fake HTTP 和 provider UI 行为覆盖。replay harness 用例继续验证 graceful stop 只向 supervisor/session leader 发送一次 SIGINT。

注意：不可把 `test/fixtures/*.py` 作为 pytest 输入；它们是会在 import 时进入 worker loop 的协议测试进程，不是测试模块。

### 3.3 官方 DAM-3B 真模型 smoke test

使用 `/home/lindenbot/RealityLab/.venvs/roomie-dam/bin/python`、官方源码和本地 6.7G `DAM-3B` 权重，对 `assets/snapshot_and_description.png` 的一个 bbox region 执行 Roomie worker 同一条 `prepare_dam_agent()` / `get_description()` / normalize 路径。

| 项目 | 结果 |
| --- | --- |
| CUDA | 12.8，RTX 5090，available |
| checkpoint shards | 2/2 loaded |
| 模型加载 | 3.047 s |
| 单图推理 | 0.630 s |
| raw output | 非空 |
| durable parse path | `schema_repaired`，保留 raw text/evidence/mask provenance |

该 smoke 验证真实官方模型加载和一次推理，不等价于 run3 的多对象 durable scheduler、SLO、公平性和跨重启端到端验收。`test_dam_worker_official_api.py` 仍负责轻量、确定性的 API 参数契约回归。

### 3.4 静态检查

```bash
git diff --check
```

结果：PASS，无 whitespace error。

### 3.5 Scene QA 安装与 Web smoke

- workspace install target 成功安装/软链接 `roomie_agibot_head_mapping_qa.launch.py`。
- `ros2 launch roomie roomie_agibot_head_mapping_qa.launch.py --show-args` 成功，确认 mapping、RViz、完整语义和 QA 参数均可从安装树解析；`enable_boxer`、`use_rviz`、`use_scene_qa` 默认均为 `true`，`qa_python` 默认指向 `jarvis` 环境。
- 使用 `jarvis` Python 启动 viewer 后，`/` 和 `/graph.json` 返回 HTTP 200；HTML 同时包含 Gemini/Doubao 按钮，provider metadata 显示两者均可用。
- 使用 `.bashrc` 中的真实 `DOUBAO_API_KEY` 调用当前默认 `doubao-seed-2-0-lite-260215`：最小回答成功；强制工具用例实际完成 1 次 function call；经 Web `/ask` 对真实离线 DSG 提问时完成 5 次场景图工具调用并返回答案与 4 个高亮组。
- 在没有运行 pipeline 的负向 smoke 中，`POST /ask` 在配置的 0.2 s discovery timeout 后返回 HTTP 500，并在 `/events` 中保留 `service_unavailable` 轨迹，没有挂死或静默切换离线 JSON。
- CLI 安装入口在管道输入 `/exit` 时进入默认交互提示并正常退出。

该 smoke 没有调用真实 Gemini API，也没有替代第 5.4 节仍未执行的完整语义 bag；Gemini function-calling 由 fake client/live transport 自动化用例覆盖。Doubao 函数调用既有 fake HTTP 确定性测试，也有上述真实 API smoke。

## 4. 真实 bag 与恢复结果

已用 `scripts/roomie_bag_replay_harness.py` 执行 run2 mapping-only 回放。配置显式关闭 detection、snapshot、DAM 和 embedding，因此本节只验收 RGB-D adapter、NVBLOX map/surface、SceneStore 对齐、graceful shutdown checkpoint 和恢复。

运行目录与产物：

- 配置：`/tmp/roomie_refactor_agibot_run2/config/pipeline.yaml`
- 指标：`/tmp/roomie_refactor_agibot_run2/metrics.json`
- 进程输出：`/tmp/roomie_refactor_agibot_run2/process/`
- RunLogger：`/tmp/roomie_refactor_agibot_run2/logs/run_20260804_175241_pid3055928/`
- checkpoint：`/tmp/roomie_refactor_agibot_run2/state/nvblox/color_tsdf_map_roomie.264735986c51cb330000000000000001.436.nvblox`
- SceneStore：`/tmp/roomie_refactor_agibot_run2/state/roomie_scene.sqlite3`

结果：

| 项目 | 结果 |
| --- | --- |
| pipeline | exit code 0，graceful stop |
| player | 到 EOF 后保留事件循环；230 s 上限触发托管收尾，最终 SIGTERM |
| map/surface revision | 436 / 436 |
| integrated frames | 434 |
| frame admission drop | 0 |
| incremental/full blocks | 639623 / 13，增量比例 0.9999797 |
| checkpoint | PASS，160,514,048 bytes |
| manifest alignment | map revision 436，scene revision 0 |

首次启动暴露了 `roomie_agibot_head_rgbd_adapter.py` 源码没有 executable bit、`--symlink-install` 后无法启动的问题，已把版本控制模式从 `100644` 修正为 `100755`。首次 graceful 收尾还暴露出 harness 对整个 `ros2 launch` 进程组发 SIGINT、launch 再转发导致节点收到两次 SIGINT 的问题；已改为先只通知 supervisor/session leader，超时才升级进程组 SIGTERM/SIGKILL，并补回归测试。

随后使用同一 SceneStore 启动实际恢复 smoke test：保持 run2 配置原有的 `tsdf.load_map=false`，只用 `tsdf.save_map=false` 禁止测试过程生成新文件。durable manifest 自动触发 coordinated recovery，NVBLOX 成功加载上述 160 MB checkpoint，日志确认：

```text
loaded durable checkpoint ... aligned_scene_revision=0
durable_scene_revision=0 abandoned_semantic_suffix_revisions=0
```

恢复进程收到单次 SIGINT 后记录 `stop requested` 和 `stopped`，没有残留 pipeline/player 进程。

后续在用户当前 `roomie_agibot_head` SceneStore 上诊断到一个恢复后的发布缺口：日志已记录 `loaded durable checkpoint`，SQLite manifest、文件大小和哈希均有效，但在没有新 RGB-D 帧时，reader-facing `latest_published_surface_` 尚未建立，因此 RViz 持续显示 `tsdf_blocks=0 / surface_points=0`。修复后 `MapThread` 会在成功加载 checkpoint 后、处理首帧前强制 full surface rebuild 并发布 frame-0 startup snapshot；新增 CPU actor 测试和 GPU NVBLOX 保存→加载→surface extraction conformance 测试均通过。为避免在用户真实 SceneStore 上生成新的 shutdown checkpoint，本轮没有用生产 YAML 再做一次会写状态的 post-fix launch。

`/tmp/roomie_refactor_agibot_run1/metrics.json` 仍只是较早中间版本的负向诊断材料，不能与本次 run2 结果混用。

## 5. 已关闭问题与真实遗留

### 5.1 已关闭：map checkpoint / scene revision 协调恢复

- 默认 `tsdf.map_load_mode=coordinated`：durable manifest 会自动触发恢复，即使 fresh-run 配置仍是 `tsdf.load_map=false`；系统先校验 manifest 指向的 scene prefix 和 checkpoint 的 backend/path/frame/config fingerprint/size/hash，再原子回退 SceneStore 的 scene journal、current/history、task/outbox、embedding 和 artifact origin 到 aligned revision。
- `tsdf.map_load_mode=seed` 是显式选择：加载配置地图或从空地图重新开始时，在同一 rewind 事务中清除既有 durable scene-derived state 和旧 map manifest，防止任意地图与旧语义状态混合。
- durable manifest 与显式 `map_load_path` 冲突时拒绝启动，不再静默忽略配置路径。
- v4 legacy manifest 迁移为 `legacy.unknown` frame/fingerprint；数据库仍可打开，但 coordinated load 会安全地因不兼容而停止。
- 成功恢复后会立即重建并发布 startup surface，不再依赖第一帧新相机数据触发 cache publication；日志增加 `published loaded checkpoint surface`。

恢复测试覆盖 63/64/65 checkpoint/delta 边界、durable semantic suffix、suffix task/embedding/origin 清理、checkpoint 保留、回退后新 revision 分支，以及 NVBLOX 真文件的 frame/config/path compatibility fence。

### 5.2 已关闭：首次 snapshot 前重启的 stable-object SLO

- `ObjectCreated` 的 Unix origin 和 priority 与 owning scene revision 同事务持久化。
- 重启只恢复 durable Unix/priority；旧进程 steady epoch/timestamp 不会跨进程复用。
- live object origin 不再因 4096 memo 容量被淘汰；对象 merge/tombstone 时才清理。
- replay memo 淘汰后查询 durable task，复用原 payload/timestamp，保持相同 task id 的字节级幂等。
- new-object burst 用 steady clock；`ArtifactSloContext` untracked 只接受 `(0, 0)`，负 due 被拒绝。

测试覆盖首次 snapshot 前真实 SceneStore close/reopen、steady fallback、origin 容量压力、wall-clock 回拨和 memo 淘汰 replay。

### 5.3 已关闭：其余有界性与 durable parser 问题

- DAM task/dependency/SLO JSON 拒绝负数转 unsigned、uint64 到 int64/int 溢出和错误数值类型。
- SceneStore object/track/provenance/graph/checkpoint/delta 的整数恢复统一使用显式类型、符号和范围检查；损坏 legacy graph revision 用例验证会拒绝 reopen。
- embedding `slo_accounted_task_ids_` 改为配置容量的 deque+set 历史，统计量暴露当前大小。

### 5.4 仍未完成

- SQLite journal、历史 checkpoint/delta 和 terminal task 记录还没有长期 compaction/retention 策略。
- map checkpoint 仍只在 graceful shutdown 发布，没有周期性 map+scene 协调 checkpoint/barrier。
- 尚未运行开启 Boxer、snapshot、官方 DAM-3B 和 embedding 的 run3 完整语义 bag；当前只完成单图真模型 smoke。
- CMake 在当前 Conda PATH 下仍提示 `libsqlite3`/`libz` 可能遮蔽系统库；真实运行应继续使用干净 ROS shell 并检查 `LD_LIBRARY_PATH`。
- Coding Plan 中明确延期的多 in-flight/batch、shared memory、PCA/OBB refinement 仍保持延期。

## 6. 后续建议顺序

1. 新建独立 run3，开启 Boxer、snapshot、官方 DAM-3B、embedding，测 artifact SLO、公平性、query/mutation 和跨重启恢复。
2. 设计周期性 map+scene checkpoint barrier；不能简单在 map actor 定时保存后再读取一个不对齐的 scene watermark。
3. 为 SQLite journal/outbox/embedding 历史制定可审计 retention/compaction 策略，并补长周期恢复测试。
4. 根据职责边界拆分提交；不要把 `build/`、`install/`、`log/`、真实 bag 产物或 `__pycache__` 提交进仓库。

## 7. 交接边界

- 本轮没有提交 commit、没有 push、没有删除用户的架构文档改动。
- 本轮执行了 mapping-only 真实 bag、实际 checkpoint 恢复、单图真实 DAM-3B 推理和真实 Doubao function-calling/Web smoke，但没有执行完整语义 bag，也没有调用真实 Gemini API。
- 当前存在两个构建树：包内 `build/roomie` 用于开发期定向测试，workspace 根 `build/roomie`/`install/roomie` 用于最终 36 项 CTest 和 ROS launch；均为构建产物，不应提交。
- 自动化测试、mapping-only 主链、graceful checkpoint 和 checkpoint restore 为全绿；第 5.4 节保留的项目没有写成 PASS。
