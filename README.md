# Adjusted-HRNet-for-Semantic-Segmentation
Keras——适用于语义分割的改编的HRNet！

# 1. pose_hrnet

*原样复制的HRNet，不同之处是相比于原作者写的代码，每类情况单独写一个函数，没有在一个函数中为了考虑各种情况写的比较复杂难懂。*

总的来说，HRNet还是存在像inception一样的stem模块，产生四倍下采样的特征图，进而逐步增加分支，每个分支完成之后接用resnet的block模块进行特征提取，完了多个分支之间进行类似于全连接之间的加法特征融合（将每个三维特征图当作全连接网络的一个节点），最后一次融合的时候只产生单个最高分辨率的分支。

## 网络结构

![pose_hrnet](/pose_hrnet.png)

# 2. seg_hrnet

pose_hrnet最终产生的分辨率为原来的四分之一，seg_hrnet相比于pose_hrnet只是在四分之一特征图上做了四倍最近邻上采样插值，最后接了softmax分类。

## 网络结构

![seg_hrnet](/seg_hrnet.png)

# 3. seg_se_hrnet

seg_se_hrnet是在seg_hrnet的基础上添加了SE block模块，也就是在每次做加法融合的之前，对各个分支的结果做通道重定标，类似于全连接网络的`w1*x1 + w2*x2`。

## 网络结构

![seg_se_hrnet](/seg_se_hrnet.png)

# 4. seg_fc_hrnet

HRNet更一般的版本，分支更多，最小分辨率为4x4，并且去掉了stem模块，从头就开始产生8个分支，不是逐渐增加分支，最高分辨率通道数为32，逐次通道增加32个。我认为这样多尺度之间的信息交换更加频繁。

## 网络结构

![seg_fc_hrnet](/seg_fc_hrnet.png)

# 4. seg_fc_hrnet_avgpooling

与seg_fc_hrnet相比，将transform_layer和每次融合时的下采样方法都从HRNet的二倍二倍步长卷积下采样转换为一步平均池化，其对于fc这种超多分支的结构能减少不少参数。