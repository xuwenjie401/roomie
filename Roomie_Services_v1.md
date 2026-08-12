# Roomie Services for realtime robot mobile manipulation

# goal
本文档描述 roomie 作为场景图组件，如何通过 制定接口规范 参与到 robot实时长程任务的执行中,
以下将 发起调用的主流程记为 main_pipe, 场景图简记为 SG, 这里的services是广义，在ros2下可能为action

# 调用场景 简单示例
## 导航 -- 指定名称拿物 或 指定地点名称拿物
比如 "帮我拿 维生素C / 维生素D / 奥美拉唑肠溶胶囊 / 康复新液盒子 / 抽屉药 / 老花镜 / 怡宝 / 纸巾"
或 "帮我去书桌拿维生素C"
此两种情况下，调用逻辑相同

main_pipe 发送任务的 完整语言输入(string)， 以及 是否允许 roomie 进一步追问
roomie query成功时,
返回 家具级的id, 位置 和 vlm检索结果的描述(string);
roomie 找到多个完全符合要求的物体时,
返回 vlm输出的反问(string);
roomie query失败时, (SG中没找到相关物体)
返回 推荐前去探索的 家具信息的 list，以及 vlm检索结果描述和推荐的理由(string)

## 视野内寻物
比如 在导航去探索的家具后，机器人弯腰，main_pipe调用roomie以确认当前 有无 要查找的物体 -- "视野内有没有奥美拉䂳"

main_pipe 发送原任务的 完整语言输入(string)
roomie 返回 是否确认找到目标，以及:
1) 成功时, 目标物体的 ObjectInfo，以及目标物体在当前head_color上的 ObjectInHeadCam 信息;
2) 失败时, 若有值得尝试拿起仔细查看的 物体，输出怀疑物体信息; 若无任何值得尝试，该字段为空;

和 vlm检索结果的描述(string)


## 场景记忆问答
比如 "沙发上有什么?"、"家里有哪些能喝的?"、"家里有几个椅子，分别描述一下是怎样的椅子"

和当前scene_qa保持一致

main_pipe 发送任务的 完整语言输入(string)，
roomie 返回 vlm调用SG-tools后的最终输出(string)


# 接口 详细说明

其中 `FurnitureInfo`、`ObjectInfo` 和 `ObjectInHeadCam` 的定义见 [roomie_msgs.md](./roomie_msgs.md)。

## 导航
```
# Goal
string task_description
bool allow_follow_up_question
float32 max_duration_s

---
# Result
uint8 STATUS_FOUND=0
uint8 STATUS_NOT_FOUND=1
uint8 STATUS_NEED_CONFIRM=2

uint8 status
string reason

# STATUS_FOUND 时有效
roomie_msgs/FurnitureInfo target_furniture

# STATUS_NOT_FOUND 时有效，顺序即推荐顺序
roomie_msgs/FurnitureInfo[] exploration_furniture

# STATUS_NEED_CONFIRM 时有效
string follow_up_question

string vlm_description

---
# Feedback
uint8 PHASE_STARTING=0
uint8 PHASE_EXPLORING=1
uint8 PHASE_PAUSED=2
uint8 PHASE_SAVING_RESULT=3
uint8 phase # 当前处于什么阶段
```

## 视野内寻物
```
# Goal
string task_description
float32 max_duration_s

---
# Result
uint8 STATUS_FOUND=0
uint8 STATUS_NOT_FOUND=1

uint8 status
string reason

# STATUS_FOUND 时有效
roomie_msgs/ObjectInfo target_object
roomie_msgs/ObjectInHeadCam target_object_in_head_cam

# STATUS_NOT_FOUND 时有效；两个数组按下标一一对应，没有怀疑物体时均为空
roomie_msgs/ObjectInfo[] suspected_objects
roomie_msgs/ObjectInHeadCam[] suspected_objects_in_head_cam

string vlm_description

---
# Feedback
uint8 PHASE_STARTING=0
uint8 PHASE_EXPLORING=1
uint8 PHASE_PAUSED=2
uint8 PHASE_SAVING_RESULT=3
uint8 phase # 当前处于什么阶段
```

## 场景记忆问答
```
# Goal
string task_description
float32 max_duration_s

---
# Result
uint8 STATUS_SUCCESS=0
uint8 STATUS_FAILED=1

uint8 status
string reason
string answer

---
# Feedback
uint8 PHASE_STARTING=0
uint8 PHASE_EXPLORING=1
uint8 PHASE_PAUSED=2
uint8 PHASE_SAVING_RESULT=3
uint8 phase # 当前处于什么阶段
```
