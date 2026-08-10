# Roomie with Furniture

当前 roomie的场景图 仅有 room和Object，缺少中间的furniture层级
现在补充关键设计:
- furniture 从Object创建，具备object的属性;
- 目前使用label的白名单进行创建，建议在config/scene_qa/下新增配置文件，初版应该要包括 desk, table, sofa, chair, shelf, bed, cabinet, nightstand, drawer.
- furniture 与 object 有 in / on关系，这部分我们需要思考下，怎么判定 和 更新时机
- furniture 可以嵌套 on / in 另一个furniture. 举例 toy -->(in) drawer -->(in) cabinet
- furniture 的创建为触发式，比如可以在load后，发送service，非实时地一次性触发, 更新所有家具 及 in/on关系。这个接口我们保留，后续想改为在线的 轮询式触发，或 按label的object创建的 事件式触发，都可以，但初版，我们仅考虑 一次性触发
- furniture 的创建未触发时，在线运行 新的object出现时，其应该可以触发和已有furniture的绑定 (in / on)

