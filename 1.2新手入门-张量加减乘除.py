from os import X_OK
import torch
from torch._C import device

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)


#——————张量的加法
z = x + y                    #两个向量相加
z = torch.add(x,y)
print(z)
y.add_(x)               #单独的y+x

#——————张量的减法
z = x - y
z = torch.sub(x,y)
print(z)

#——————张量的乘法
z = x * y
z = torch.mul(x,y)
print(z)
y.mul_(x)

#——————张量的除法
z = x / y
z = torch.div(x, y)

#------------切片功能，
x = torch.rand(5,3)
print(x)
print(x[:, 0])     #只输出第0列
print(x[1,1])      #只输出第1行第1列
print(x[1,1].item())      #也可以这样写item方法，如果他是点

#_____________张量展平
#view 方法是 PyTorch 中的一个张量操作方法，用于调整张量的形状（尺寸）而不改变其元素数量或数据本身。
#view 方法将一个张量从一个形状调整为另一个形状，[只要新形状的元素数量与原张量相同即可]。例如，从一个 3x3 的矩阵调整为一个大小为 9 的向量。
#view 方法也可以用来将多维张量展平为一维张量。这在神经网络中特别常见，因为很多神经网络层（如全连接层）的输入都是一维的
x = torch.rand(4,4)
print(x)
y = x.view(16)   #原始的 4x4 矩阵被展平成一个长度为 16 的一维张量
print(y)
#如果不知道如何正确的调整张量的数据，可以先随便填一个
x = torch.rand(4,4)
print(x)
y = x.view([-1,8])#这个值是错误的，但输出会输出正确的torch.Size([2,8])
print(y.size)

#不用gpu从tensor到numpy
import torch
import numpy as np
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))      #此时可以看到输出的type(b)是numpy.ndarray
#此时要注意如果张量是在cpu而不是gpu上，张量和numpy将会共享一个内存，改变一个，另一个也会改变
a.add_(1)
print(a)#此时a相加1，tensor变为[2,2,2,2,2]，b也会变为同样的
print(b)

#不用gpu从numpy到tensor
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)

#有gpu调用cuda的情况下
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.one(5)
    y = y.to(device)
    z = x + y
    #此时不能再调用z.numpy(),因为此时numpy在cpu上
    z = z.to("cpu")#需要将tensor再挪回cpu，再能继续调用numpy


#参数要求
x = torch.ones(5,requires_grad=True)
print(x)
