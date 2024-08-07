# by_particle_number_training
用于数量反向传播的代码将主要在train_colour_and_shape_by_stack.py中。
首先对于用于处理数据集的TripleDataset类，改动为：先由for循环，将每个扣出来的颗粒图像的路径存放在img_path_pos中，然后新建了attribute_list，用于存放每种组合所对应的粒子真实数目，然后通过一个嵌套循环生成新的op_list,op_list中也加入了属性组合的操作，且将每种组合对应的真实数量作为answer放入op_list所在的字典中。由于一共8中组合（颜色4个概念，形状2个概念），故每轮训练中for循环进行了8次，每次的y_pred得到的都为满足每种组合的数量。

  
reasoning_out_and_in_and.py中在attributes_classify中添加了pic_tens = torch.stack([pic for pic in pic_ten])，使每颗颗粒的小图像全部添加到pic_tens中，一起送入网络，得到所有颗粒的小图片预测结果，然后根据属性组合操作attribute_combin进行将两种属性特定概念的组合，在组合前通过torch.where函数实现大于等于0.5的置信度变为1，小于0.5变为0，组合完毕后通过sum函数统计满足组合的数量，然后与真实值计算loss，进行训练。


计算图为and_graph2.pdf

