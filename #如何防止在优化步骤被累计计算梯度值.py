
#��η�ֹ���Ż����豻�ۼƼ����ݶ�ֵ
import torch
weight = torch.pnes(4, requires_grad=True)


#����������������������ȷ���ݶ�ֵ����

for epoch in range(1):#����һ�������ѵ������
    model_output = (weights * 3).sum() #ģ��ģ������������ǽ�Ȩ�س���3�����

    model_output.backward()#ʹ�÷��򴫲�����ģ����������Ȩ�ص��ݶ�

    print(weights.grad)#��ӡȨ�ص��ݶ�ֵ

    weight.grad.zero_()#��Ȩ�ص��ݶ�ֵ����
#weights.grad.zero_() ��Ȩ�ص��ݶ�ֵ���㣬�Ա���һ�μ����ݶȡ�
#��������Ǳ�Ҫ�ģ���Ϊ PyTorch Ĭ�ϻ��ۻ��ݶ�ֵ����������㣬
#��һ�μ�����ݶ�ֵ������һ�μ���Ľ����ӡ�

#���������������������Ż���
optimizer = torch.optim.SGD(weights, lr=0.01)   #������һ���Ż�����weights�Ǳ��Ż��Ĳ�����lr��ѧϰ��
optimizer.step()#�������õ�ѧϰ���Լ��������ݶ�ֵ�����²���
optimizer.zero_grad() #����zero.grad��������֮ǰ����õ��Ĳ����ݶ�ֵ���㣬�����ݶ�ֵ�ۼ�

#�����������������������ú�����
z.backward()

weights.grad.zero_()  #�����ݶ��ۼ�

