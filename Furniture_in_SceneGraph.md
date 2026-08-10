# Roomie with Furniture

当前 roomie的场景图 仅有 room和Object，缺少中间的furniture层级
现在补充关键设计:
- furniture 从Object创建，具备object的属性;
- 目前使用label的白名单进行创建，建议在config/scene_qa/下新增配置文件，初版应该要包括 desk, table, sofa, chair, shelf, bed, cabinet, nightstand, drawer.
- furniture 与 object 有 in / on关系，这部分我们需要思考下，怎么判定 和 更新时机
- furniture 可以嵌套 on / in 另一个furniture. 举例 toy -->(in) drawer -->(in) cabinet
- furniture 的创建为触发式，比如可以在load后，发送service，非实时地一次性触发, 更新所有家具 及 in/on关系。这个接口我们保留，后续想改为在线的 轮询式触发，或 按label的object创建的 事件式触发，都可以，但初版，我们仅考虑 一次性触发
- furniture 的创建未触发时，在线运行 新的object出现时，其应该可以触发和已有furniture的绑定 (in / on)

## 已实现约定

- canonical scene graph schema 为 v4。Furniture 是 Object 的同 ID role 投影，属性仍只从 Object 读取，不复制 geometry/semantic/annotation。
- 显式调用 `/roomie/rebuild_furniture_graph`（`std_srvs/Trigger`）时，根据 `config/scene_qa/furniture.json` 对当前 publishable Object 做完整 role 重建，并原子更新派生关系；状态未变化时为 no-op。
- role 建立以后，Object 的几何、生命周期、merge/tombstone 以及 Room bounds 变化会自动维护关系；新 label 是否成为 Furniture 仍只在下一次显式 rebuild 时判断。
- `in`/`on` 使用 yaw-OBB 的确定性几何规则。每个 child 最多选择一个 Furniture parent，`on` 优先于 `in`；Furniture 也可作为 child，关系端点类型会保留为 `furniture`。
- Room 同时保留 `room_contains_object`，并为 Furniture 增加 `room_contains_furniture`。
- `/roomie/query_scene` 支持 `furniture` 方法；`get_object` 返回可选 `furniture_role`，`rooms` 返回 `furniture_ids`，`relations` 可按同一 Object/Furniture ID 查询。
- Scene QA 离线和在线工具均提供 `list_furniture` 与 `get_relations`。
