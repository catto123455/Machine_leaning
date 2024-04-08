
import torch

#====================如何生成梯度
x = torch.randn(3,requires_grad=True)#比如添加这个，因为默认情况是False
print(x)

y = x+2     #计算输出Y这个过程被称作反向传播

#会自动生成一个函数，这个函数可以用于反向传播
print(y)

z = y*y*2
z = z.mean()
print(z)#不同的运算有不同的梯度，+就是add，*就是mul，mean就是mean

x = torch.tensor([0.1,1.0,0.001]),dtype=torch,float32)
z.backward() #dz/dx
print(x.grad)#计算一个张量 z 相对于另一个张量 x 的梯度，并打印出梯度值

#====================阻止pytorch梯度的三种方法

#————method1

x.requires_grad_(False)   #它会修改函数变量，生成的结果就没有requires_grad=Ture了

#————method2

y = z.detach()            #x.detach()这将创建一个具有相同值的新向量，它不需要梯度
print(y)

#————method3
with torch.no_grad():
    y = x + 2
    print(y)              # with torch.no_grad():，此时输出y可以看到没有梯度

