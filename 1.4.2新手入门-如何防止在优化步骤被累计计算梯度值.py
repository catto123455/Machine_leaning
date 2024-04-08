
#如何防止在优化步骤被累计计算梯度值
import torch
weight = torch.pnes(4, requires_grad=True)


#——————————正确的梯度值计算

for epoch in range(1):#进行一个假设的训练周期
    model_output = (weights * 3).sum() #模拟模型输出，这里是将权重乘以3并求和

    model_output.backward()#使用反向传播计算模型输出相对于权重的梯度

    print(weights.grad)#打印权重的梯度值

    weight.grad.zero_()#将权重的梯度值清零
#weights.grad.zero_() 将权重的梯度值清零，以便下一次计算梯度。
#这个步骤是必要的，因为 PyTorch 默认会累积梯度值，如果不清零，
#下一次计算的梯度值会与上一次计算的结果相加。

#——————————优化器
optimizer = torch.optim.SGD(weights, lr=0.01)   #创建了一个优化器，weights是被优化的参数，lr是学习率
optimizer.step()#根据设置的学习率以及参数的梯度值来更新参数
optimizer.zero_grad() #调用zero.grad方法，将之前计算得到的参数梯度值清零，避免梯度值累计

#——————————调用后向函数
z.backward()

weights.grad.zero_()  #清零梯度累加

