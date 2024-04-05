from os import X_OK
import torch
from torch._C import device

x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)


#�����������������ļӷ�
z = x + y                    #�����������
z = torch.add(x,y)
print(z)
y.add_(x)               #������y+x

#�����������������ļ���
z = x - y
z = torch.sub(x,y)
print(z)

#�����������������ĳ˷�
z = x * y
z = torch.mul(x,y)
print(z)
y.mul_(x)

#�����������������ĳ���
z = x / y
z = torch.div(x, y)

#------------��Ƭ���ܣ�
x = torch.rand(5,3)
print(x)
print(x[:, 0])     #ֻ�����0��
print(x[1,1])      #ֻ�����1�е�1��
print(x[1,1].item())      #Ҳ��������дitem������������ǵ�

#_____________����չƽ
#view ������ PyTorch �е�һ�������������������ڵ�����������״���ߴ磩�����ı���Ԫ�����������ݱ���
#view ������һ��������һ����״����Ϊ��һ����״��[ֻҪ����״��Ԫ��������ԭ������ͬ����]�����磬��һ�� 3x3 �ľ������Ϊһ����СΪ 9 ��������
#view ����Ҳ������������ά����չƽΪһά�������������������ر𳣼�����Ϊ�ܶ�������㣨��ȫ���Ӳ㣩�����붼��һά��
x = torch.rand(4,4)
print(x)
y = x.view(16)   #ԭʼ�� 4x4 ����չƽ��һ������Ϊ 16 ��һά����
print(y)
#�����֪�������ȷ�ĵ������������ݣ������������һ��
x = torch.rand(4,4)
print(x)
y = x.view([-1,8])#���ֵ�Ǵ���ģ�������������ȷ��torch.Size([2,8])
print(y.size)

#����gpu��tensor��numpy
import torch
import numpy as np
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))      #��ʱ���Կ��������type(b)��numpy.ndarray
#��ʱҪע�������������cpu������gpu�ϣ�������numpy���Ṳ��һ���ڴ棬�ı�һ������һ��Ҳ��ı�
a.add_(1)
print(a)#��ʱa���1��tensor��Ϊ[2,2,2,2,2]��bҲ���Ϊͬ����
print(b)

#����gpu��numpy��tensor
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)

#��gpu����cuda�������
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.one(5)
    y = y.to(device)
    z = x + y
    #��ʱ�����ٵ���z.numpy(),��Ϊ��ʱnumpy��cpu��
    z = z.to("cpu")#��Ҫ��tensor��Ų��cpu�����ܼ�������numpy


#����Ҫ��
x = torch.ones(5,requires_grad=True)
print(x)