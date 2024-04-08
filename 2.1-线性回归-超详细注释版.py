#需要安装pip install scikit-learn,matplotlib

#1）设计模型（输入和输出的大小，不同层的前向传播）
#2）构造损失-loss和优化器-optimizer
#3) 训练循环
# - forward pass：计算预测
# - backward psss：梯度
# - update weights 


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#【生成训练数据的函数-sklearn.datasets】
#准备数据,使用sklearn.datasets.make_regression 生成一个简单的回归数据集
#sklearn.datasets.make_regression 是 scikit-learn库中的一个函数，用于生成一个回归问题的模拟数据集
#除了线性回归，sklearn的模块sklearn.datasets还提供其他的模拟数据集：
#如make_classification-分类问题
#make_blobs-聚类
#非线性分类问题-make_moons（半圆形状，make_circles（环形状
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

#【转变numpy到tensor的函数-torch.from_numpy】
#使用sklearn.datasets函数生成的数组是numpy数组而不是张量↑，所以需要把numpy转变为tensor-张量
#使用torch.from_numpy函数将numpy转变为tensor-张量，
#并发动了连招astype() 函数把数据类型转变为32位浮点数，在深度学习中常用的是np.float64-64位浮点数，np.float32-32位浮点数
#但除此之外也可以转变为np.int64-64位整数，np.bool-布尔类型，表示True或False，np.str-字符串
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

#【重塑张量维度的函数-shape/size】
#view函数，将目标变量y 1维张量重塑为一个2维张量
#y.shape表示的是y的形状，y.shape[0]表示的是第一个维度，就是行数，1表示的是第二个维度，就是列数。
#0表示现在这个行数保持不变，所以只有一列，也叫做列向量。
#1维变3维：y = y.view(1, y.size(0), 1)
#1维变4维：y = y.view(1, 1, y.size(0), 1)
#size和shape的区别有但不大，可以忽略
y = y.view(y.shape[0],1)

#【元组解包赋值-X.shape】
#[元组]是指数组，例如my_tuple = (1, 2, 3)，n_samples,n_features也是元组
#其他的元组：
# 空元组-empty_tuple = ()
# 包含整数和字符串的元组-my_tuple = (1, 'hello', 3.14)
# 嵌套元组-nested_tuple = ((1, 2), ('a', 'b', 'c'))
# 单元素元组，注意要在元素后面加上逗号-single_element_tuple = (42,)
# 省略括号的元组-tuple_without_parentheses = 1, 2, 3
#[解包赋值]
#将元组中的元素解包并分别赋值给变量
#a, b, c = my_tuple
# 输出赋值后的变量
#print(a)  # 输出：1
#print(b)  # 输出：2
#print(c)  # 输出：3
#下面的代码是把包含两个元素的n_samples,n_features元组解包，并通过 = 赋值给 X.shape
n_samples,n_features = X.shape

#1）设计模型
input_size = n_features
#神经网络的输出
#神经网络的输出大小通常是根据任务的需求和数据的特点来确定的。
#回归任务中，输出大小通常是 1（预测单个连续值）或者是多个连续值的数量（例如，多变量回归）。
#图像处理中，输出大小通常是图像的像素数量或特征数量。
output_size = 1
#通过下面的nn.linear是构造函数，创造了一个具有input_size输入特征和output_size的输出特征的线性层对象model。
#nn.函数还可以构造其他的层
#卷积层 (nn.Conv1d, nn.Conv2d, nn.Conv3d)：用于处理具有空间结构的数据，如图像或视频数据
#池化层 (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)：用于减小输入的空间维度，提取输入中的重要特征。
#循环层 (nn.RNN, nn.LSTM, nn.GRU)：用于处理序列数据，例如文本或时间序列数据。
#规范化层 (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)：用于加速收敛并提高稳定性，通过对输入进行归一化和标准化。
#Dropout 层 (nn.Dropout, nn.Dropout2d, nn.Dropout3d)：用于防止过拟合，通过随机丢弃一部分神经元来减少神经网络的复杂性。
#激活函数 (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax)：用于在网络的各层之间引入非线性变换，增加网络的表达能力。
#损失函数 (nn.MSELoss, nn.CrossEntropyLoss, nn.BCELoss)：用于计算模型输出与目标值之间的差异，是训练过程中优化的目标。
#转置卷积层 (nn.ConvTranspose2d)：用于进行上采样操作，将低分辨率特征图转换为高分辨率特征图，常用于图像生成和语义分割等任务。
#模型的层数是自定义的，一般增加神经网络的层数可以增加网络的表达能力，使其能够更好地学习数据中的复杂模式和结构。
#增加层数也会增加网络的复杂度和计算量，可能会导致过拟合或训练时间过长的问题。
model = nn.Linear(input_size,output_size)
#2）定义损失loss和优化器optimizer
#[学习率learning rate]，决定了模型在每一次参数更新时所移动的距离。
#较大的学习率会导致参数更新幅度过大，可能会使模型无法收敛；
#而较小的学习率会导致参数更新缓慢，可能会使训练过程变得缓慢或陷入局部最优解。
learning_rate = 0.01
#这行代码创建了一个均方误差损失函数的实例，并将其赋值给变量 criterion
criterion = nn.MSELoss()
#【构造函数-torch.optim.SGD】使用构造函数torch.optim.SGD()用于创建随机梯度下降优化器。
#model.parameters()用于获取模型中所有需要学习的参数，然后将这些参数传递给优化器，以便优化器可以更新它们。
#这里将模型 model 中的所有参数传递给了优化器
#lr=learning_rate 参数指定了优化器的学习率，即在每次参数更新时的步长大小
#构造函数torch.optim.SGD()中都可以放哪些参数
#params：必需参数，用于指定要优化的模型参数。可以通过 model.parameters() 获取模型中需要更新的参数。
#lr:学习率，即每次参数更新的步长大小。控制了优化过程中参数更新的速度。默认值为 0.01。
#momentum:动量，用于加速 SGD 在相关方向上前进，并减少振荡。默认值为 0。
#weight_decay:权重衰减（L2 正则化），在每次更新时对参数进行惩罚，以避免过拟合。默认值为 0。
#dampening：动量的阻尼，用于减少动量的影响。默认值为 0。
#nesterov：是否使用 Nesterov 动量。默认值为 False。
#params_group：一个可选的参数组列表，用于分组不同的参数以应用不同的优化策略。
#lr_decay：学习率衰减，控制学习率在训练过程中的递减。
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#3）训练循环
#这行代码定义了训练的轮数，也称为迭代次数或周期数。
#num_epochs 表示训练循环将遍历整个训练数据集的次数。
#在这个例子中，训练循环将重复执行 100 次。
num_epochs = 100
#这是一个 for 循环语句，用于迭代训练过程中的每个周期-epoch。
#在每个epoch-周期中，模型将使用整个训练数据集进行一次前向传播和反向传播，并更新模型的参数，以使损失函数最小化。
#range(num_epochs) 创建了一个从 0 到 num_epochs-1 的整数序列，（从0开始计数为1，训练100次）
#这个序列被用作迭代器，表示训练循环将遍历的周期数。
#直接使用 range(100)也是等效的，但是就不是变量了。
for epoch in range(num_epochs):
    #前向传播-forward pass和损失-loss
    #前向传播，直接进行函数运算
    y_predicted = model(X)
    #loss值

    #类是用户自定义的数据类型，函数是可重用的代码块
    #前文代码：criterion = nn.MSELoss()创建了一个实例。
    #loss调用这个名为criterion的类，调用nn.MSE()的函数来计算y_predicted，y的损失函数。
    loss = criterion(y_predicted, y)
    #backward pass
    #在 PyTorch 中，通过调用损失张量的 backward() 方法，可以自动计算损失函数关于模型参数的梯度。
    loss.backward()

    #update
    #optimizer.step() 方法会根据计算得到的梯度更新模型中的参数，使得损失函数尽可能地减小。
    optimizer.step()
    #这行代码是用于清除梯度信息的步骤。在每个周期结束时，通常需要清除之前计算的梯度信息，以防止梯度累积。
    #调用 optimizer.zero_grad() 方法会将模型中所有参数的梯度信息归零，准备好接收下一个周期的新的梯度信息。
    optimizer.zero_grad()


    #这行代码是用于控制训练过程中输出日志信息的条件语句。在每个周期结束后，它会判断当前周期是否是 10 的倍数。
    #如果当前周期是 10 的倍数，则会执行下面的代码块，打印当前周期数和损失值。
    if (epoch+1) % 10 == 0:
        #字符串前缀 f 表示这是一个 f-string，它可以在字符串中插入变量和表达式的值。
        #在这个 f-string 中，{} 中的部分会被替换为相应的变量或表达式的值。
        #{epoch+1} 表示将变量 epoch 的值加 1（因为 Python 中索引从 0 开始）插入到字符串中。
        #loss.item() 表示从损失张量中提取出单个数值，因为损失张量可能是一个具有多个元素的张量，但只需要其中的一个数值来表示损失值。
        #{loss.item():.4f} 表示将损失张量 loss 的值提取出来，并保留四位小数进行格式化。
        #如果要保留五位小数，可以将格式化字符串修改为 {loss.item():.5f}
        print(f"epoch:{epoch+1}, loss = {loss.item():.4f}")

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted,"b")
plt.show()







