
import torch

#====================��������ݶ�
x = torch.randn(3,requires_grad=True)#��������������ΪĬ�������False
print(x)

y = x+2     #�������Y������̱��������򴫲�

#���Զ�����һ����������������������ڷ��򴫲�
print(y)

z = y*y*2
z = z.mean()
print(z)#��ͬ�������в�ͬ���ݶȣ�+����add��*����mul��mean����mean

x = torch.tensor([0.1,1.0,0.001]),dtype=torch,float32)
z.backward() #dz/dx
print(x.grad)#����һ������ z �������һ������ x ���ݶȣ�����ӡ���ݶ�ֵ

#====================��ֹpytorch�ݶȵ����ַ���

#��������method1

x.requires_grad_(False)   #�����޸ĺ������������ɵĽ����û��requires_grad=Ture��

#��������method2

y = z.detach()            #x.detach()�⽫����һ��������ֵͬ����������������Ҫ�ݶ�
print(y)

#��������method3
with torch.no_grad():
    y = x + 2
    print(y)              # with torch.no_grad():����ʱ���y���Կ���û���ݶ�

