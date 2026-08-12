# Roomie Messages v1

本文档定义 [Roomie_Services_v1.md](./Roomie_Services_v1.md) 中接口引用的基础消息。

## `FurnitureInfo.msg`

实现：`../roomie_msgs/msg/FurnitureInfo.msg`

```ros
int32 furniture_id
string name
string label
string description
geometry_msgs/PointStamped position
geometry_msgs/Vector3 size
```

- `furniture_id`：家具在 SG 中的 ID。
- `name`：家具实例名称；没有时为空。
- `label`：家具类别。
- `description`：家具描述；没有时为空。
- `position`：家具在 SG 中的位置。
- `size`：家具的三维尺寸。

## `ObjectInfo.msg`

实现：`../roomie_msgs/msg/ObjectInfo.msg`

```ros
int32 object_id
string label
string name
string description
geometry_msgs/PoseStamped pose
geometry_msgs/Vector3 size
```

- `object_id`：物体在 SG 中的 ID。
- `label`：物体类别。
- `name`：物体名称；没有时为空。
- `description`：物体描述；没有时为空。
- `pose`：物体的三维位姿。
- `size`：物体的三维尺寸。

## `ObjectInHeadCam.msg`

实现：`../roomie_msgs/msg/ObjectInHeadCam.msg`

```ros
std_msgs/Header header
sensor_msgs/RegionOfInterest bbox
```

- `header`：当前 `head_color` 图像的 header。
- `bbox`：物体在当前 `head_color` 图像中的 2D bounding box。
