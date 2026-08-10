# Roomie for Robots

## Hand Image
当前roomie的mapping链路 需要配套的rgb-d输入，然后detection链路仅需 rgb图像内外参已知，即可使用地图投影得到depth patch 进行后续流程，现在我们需要加入手部图像的处理，大体可以复用头部图像的流程，但需要做以下方面的改动。

- 示例数据: 
/home/lindenbot/Datasets/roomie/bags/manip_0808_2

- 
