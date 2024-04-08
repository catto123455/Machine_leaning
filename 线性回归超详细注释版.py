#��Ҫ��װpip install scikit-learn,matplotlib

#1�����ģ�ͣ����������Ĵ�С����ͬ���ǰ�򴫲���
#2��������ʧ-loss���Ż���-optimizer
#3) ѵ��ѭ��
# - forward pass������Ԥ��
# - backward psss���ݶ�
# - update weights 


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#������ѵ�����ݵĺ���-sklearn.datasets��
#׼������,ʹ��sklearn.datasets.make_regression ����һ���򵥵Ļع����ݼ�
#sklearn.datasets.make_regression �� scikit-learn���е�һ����������������һ���ع������ģ�����ݼ�
#�������Իع飬sklearn��ģ��sklearn.datasets���ṩ������ģ�����ݼ���
#��make_classification-��������
#make_blobs-����
#�����Է�������-make_moons����Բ��״��make_circles������״
X_numpy, y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

#��ת��numpy��tensor�ĺ���-torch.from_numpy��
#ʹ��sklearn.datasets�������ɵ�������numpy�����������������������Ҫ��numpyת��Ϊtensor-����
#ʹ��torch.from_numpy������numpyת��Ϊtensor-������
#������������astype() ��������������ת��Ϊ32λ�������������ѧϰ�г��õ���np.float64-64λ��������np.float32-32λ������
#������֮��Ҳ����ת��Ϊnp.int64-64λ������np.bool-�������ͣ���ʾTrue��False��np.str-�ַ���
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

#����������ά�ȵĺ���-shape/size��
#view��������Ŀ�����y 1ά��������Ϊһ��2ά����
#y.shape��ʾ����y����״��y.shape[0]��ʾ���ǵ�һ��ά�ȣ�����������1��ʾ���ǵڶ���ά�ȣ�����������
#0��ʾ��������������ֲ��䣬����ֻ��һ�У�Ҳ������������
#1ά��3ά��y = y.view(1, y.size(0), 1)
#1ά��4ά��y = y.view(1, 1, y.size(0), 1)
#size��shape�������е����󣬿��Ժ���
y = y.view(y.shape[0],1)

#��Ԫ������ֵ-X.shape��
#[Ԫ��]��ָ���飬����my_tuple = (1, 2, 3)��n_samples,n_featuresҲ��Ԫ��
#������Ԫ�飺
# ��Ԫ��-empty_tuple = ()
# �����������ַ�����Ԫ��-my_tuple = (1, 'hello', 3.14)
# Ƕ��Ԫ��-nested_tuple = ((1, 2), ('a', 'b', 'c'))
# ��Ԫ��Ԫ�飬ע��Ҫ��Ԫ�غ�����϶���-single_element_tuple = (42,)
# ʡ�����ŵ�Ԫ��-tuple_without_parentheses = 1, 2, 3
#[�����ֵ]
#��Ԫ���е�Ԫ�ؽ�����ֱ�ֵ������
#a, b, c = my_tuple
# �����ֵ��ı���
#print(a)  # �����1
#print(b)  # �����2
#print(c)  # �����3
#����Ĵ����ǰѰ�������Ԫ�ص�n_samples,n_featuresԪ��������ͨ�� = ��ֵ�� X.shape
n_samples,n_features = X.shape

#1�����ģ��
input_size = n_features
#����������
#������������Сͨ���Ǹ����������������ݵ��ص���ȷ���ġ�
#�ع������У������Сͨ���� 1��Ԥ�ⵥ������ֵ�������Ƕ������ֵ�����������磬������ع飩��
#ͼ�����У������Сͨ����ͼ�����������������������
output_size = 1
#ͨ�������nn.linear�ǹ��캯����������һ������input_size����������output_size��������������Բ����model��
#nn.���������Թ��������Ĳ�
#����� (nn.Conv1d, nn.Conv2d, nn.Conv3d)�����ڴ�����пռ�ṹ�����ݣ���ͼ�����Ƶ����
#�ػ��� (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)�����ڼ�С����Ŀռ�ά�ȣ���ȡ�����е���Ҫ������
#ѭ���� (nn.RNN, nn.LSTM, nn.GRU)�����ڴ����������ݣ������ı���ʱ���������ݡ�
#�淶���� (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)�����ڼ�������������ȶ��ԣ�ͨ����������й�һ���ͱ�׼����
#Dropout �� (nn.Dropout, nn.Dropout2d, nn.Dropout3d)�����ڷ�ֹ����ϣ�ͨ���������һ������Ԫ������������ĸ����ԡ�
#����� (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax)������������ĸ���֮����������Ա任����������ı��������
#��ʧ���� (nn.MSELoss, nn.CrossEntropyLoss, nn.BCELoss)�����ڼ���ģ�������Ŀ��ֵ֮��Ĳ��죬��ѵ���������Ż���Ŀ�ꡣ
#ת�þ���� (nn.ConvTranspose2d)�����ڽ����ϲ������������ͷֱ�������ͼת��Ϊ�߷ֱ�������ͼ��������ͼ�����ɺ�����ָ������
#ģ�͵Ĳ������Զ���ģ�һ������������Ĳ���������������ı��������ʹ���ܹ����õ�ѧϰ�����еĸ���ģʽ�ͽṹ��
#���Ӳ���Ҳ����������ĸ��ӶȺͼ����������ܻᵼ�¹���ϻ�ѵ��ʱ����������⡣
model = nn.Linear(input_size,output_size)
#2��������ʧloss���Ż���optimizer
#[ѧϰ��learning rate]��������ģ����ÿһ�β�������ʱ���ƶ��ľ��롣
#�ϴ��ѧϰ�ʻᵼ�²������·��ȹ��󣬿��ܻ�ʹģ���޷�������
#����С��ѧϰ�ʻᵼ�²������»��������ܻ�ʹѵ�����̱�û���������ֲ����Ž⡣
learning_rate = 0.01
#���д��봴����һ�����������ʧ������ʵ���������丳ֵ������ criterion
criterion = nn.MSELoss()
#�����캯��-torch.optim.SGD��ʹ�ù��캯��torch.optim.SGD()���ڴ�������ݶ��½��Ż�����
#model.parameters()���ڻ�ȡģ����������Ҫѧϰ�Ĳ�����Ȼ����Щ�������ݸ��Ż������Ա��Ż������Ը������ǡ�
#���ｫģ�� model �е����в������ݸ����Ż���
#lr=learning_rate ����ָ�����Ż�����ѧϰ�ʣ�����ÿ�β�������ʱ�Ĳ�����С
#���캯��torch.optim.SGD()�ж����Է���Щ����
#params���������������ָ��Ҫ�Ż���ģ�Ͳ���������ͨ�� model.parameters() ��ȡģ������Ҫ���µĲ�����
#lr:ѧϰ�ʣ���ÿ�β������µĲ�����С���������Ż������в������µ��ٶȡ�Ĭ��ֵΪ 0.01��
#momentum:���������ڼ��� SGD ����ط�����ǰ�����������񵴡�Ĭ��ֵΪ 0��
#weight_decay:Ȩ��˥����L2 ���򻯣�����ÿ�θ���ʱ�Բ������гͷ����Ա������ϡ�Ĭ��ֵΪ 0��
#dampening�����������ᣬ���ڼ��ٶ�����Ӱ�졣Ĭ��ֵΪ 0��
#nesterov���Ƿ�ʹ�� Nesterov ������Ĭ��ֵΪ False��
#params_group��һ����ѡ�Ĳ������б����ڷ��鲻ͬ�Ĳ�����Ӧ�ò�ͬ���Ż����ԡ�
#lr_decay��ѧϰ��˥��������ѧϰ����ѵ�������еĵݼ���
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

#3��ѵ��ѭ��
#���д��붨����ѵ����������Ҳ��Ϊ������������������
#num_epochs ��ʾѵ��ѭ������������ѵ�����ݼ��Ĵ�����
#����������У�ѵ��ѭ�����ظ�ִ�� 100 �Ρ�
num_epochs = 100
#����һ�� for ѭ����䣬���ڵ���ѵ�������е�ÿ������-epoch��
#��ÿ��epoch-�����У�ģ�ͽ�ʹ������ѵ�����ݼ�����һ��ǰ�򴫲��ͷ��򴫲���������ģ�͵Ĳ�������ʹ��ʧ������С����
#range(num_epochs) ������һ���� 0 �� num_epochs-1 ���������У�����0��ʼ����Ϊ1��ѵ��100�Σ�
#������б���������������ʾѵ��ѭ������������������
#ֱ��ʹ�� range(100)Ҳ�ǵ�Ч�ģ����ǾͲ��Ǳ����ˡ�
for epoch in range(num_epochs):
    #ǰ�򴫲�-forward pass����ʧ-loss
    #ǰ�򴫲���ֱ�ӽ��к�������
    y_predicted = model(X)
    #lossֵ

    #�����û��Զ�����������ͣ������ǿ����õĴ����
    #ǰ�Ĵ��룺criterion = nn.MSELoss()������һ��ʵ����
    #loss���������Ϊcriterion���࣬����nn.MSE()�ĺ���������y_predicted��y����ʧ������
    loss = criterion(y_predicted, y)
    #backward pass
    #�� PyTorch �У�ͨ��������ʧ������ backward() �����������Զ�������ʧ��������ģ�Ͳ������ݶȡ�
    loss.backward()

    #update
    #optimizer.step() ��������ݼ���õ����ݶȸ���ģ���еĲ�����ʹ����ʧ���������ܵؼ�С��
    optimizer.step()
    #���д�������������ݶ���Ϣ�Ĳ��衣��ÿ�����ڽ���ʱ��ͨ����Ҫ���֮ǰ������ݶ���Ϣ���Է�ֹ�ݶ��ۻ���
    #���� optimizer.zero_grad() �����Ὣģ�������в������ݶ���Ϣ���㣬׼���ý�����һ�����ڵ��µ��ݶ���Ϣ��
    optimizer.zero_grad()


    #���д��������ڿ���ѵ�������������־��Ϣ��������䡣��ÿ�����ڽ����������жϵ�ǰ�����Ƿ��� 10 �ı�����
    #�����ǰ������ 10 �ı��������ִ������Ĵ���飬��ӡ��ǰ����������ʧֵ��
    if (epoch+1) % 10 == 0:
        #�ַ���ǰ׺ f ��ʾ����һ�� f-string�����������ַ����в�������ͱ��ʽ��ֵ��
        #����� f-string �У�{} �еĲ��ֻᱻ�滻Ϊ��Ӧ�ı�������ʽ��ֵ��
        #{epoch+1} ��ʾ������ epoch ��ֵ�� 1����Ϊ Python �������� 0 ��ʼ�����뵽�ַ����С�
        #loss.item() ��ʾ����ʧ��������ȡ��������ֵ����Ϊ��ʧ����������һ�����ж��Ԫ�ص���������ֻ��Ҫ���е�һ����ֵ����ʾ��ʧֵ��
        #{loss.item():.4f} ��ʾ����ʧ���� loss ��ֵ��ȡ��������������λС�����и�ʽ����
        #���Ҫ������λС�������Խ���ʽ���ַ����޸�Ϊ {loss.item():.5f}
        print(f"epoch:{epoch+1}, loss = {loss.item():.4f}")

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted,"b")
plt.show()







