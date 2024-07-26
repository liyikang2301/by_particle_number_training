# by_particle_number_training
用于数量反向传播的代码主要在train_colour_and_shape_by_stack.py中
reasoning_out_and_in.py中在attributes_classify中添加了pic_tens = torch.stack([pic for pic in pic_ten])，使每颗颗粒的小图像全部添加到pic_tens中，一起送入网络
