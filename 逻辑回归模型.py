#1�����ģ�ͣ����������Ĵ�С����ͬ���ǰ�򴫲���
#2��������ʧ-loss���Ż���-optimizer
#3) ѵ��ѭ��
# - forward pass������Ԥ��
# - backward psss���ݶ�
# - update weights 

from typing_extensions import _AnnotatedAlias
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0��׼������
bc = datasets.load_breast_cancer()
# �������ٰ����ݼ��������������ݸ��� X��Ŀ���ǩ���� y��
#��������X��bc.date��ʾ������569,30����Ŀ������y��bc.target����״��569����
X, y = bc.data,bc.target

#Ԫ������n_samples, n_features��ֵ
n_samples, n_features = X.shape
print(n_samples, n_features)

#�������ݼ�����Ϊѵ�����Ͳ��Լ�-train_test_split ���������������ڽ����ݼ�����Ϊѵ�����Ͳ��Լ��Ĺ��ߡ�
#X: �������󣬰��������е�����������
#y: Ŀ�����������������е�Ŀ��ֵ��
#test_size: ���Լ��Ĵ�С����������Ϊ����������ʾ���Լ�ռ���������ı���������������ʾ���Լ���������������
#           �����test_size=0.2 ��ʾ�� 20% ����������Ϊ���Լ���
#random_state: ������ӣ�����ȷ��ÿ�λ��ֵĽ��������ͬ�ġ�
#              ��������øò�����ÿ�����д���ʱ����õ���ͬ�Ļ��ֽ����
#���ص�ֵ��
#X_train: ��һ������ֵ���ڽ���ѵ��������������
#X_test: �ڶ�������ֵ���ڽ���ѵ������Ŀ������
#y_train: ����������ֵ���ڽ���ѵ������Ŀ������
#y_test: ���ĸ�����ֵ���ڽ��ղ��Լ���Ŀ������
#���۱�������ʲô����a��b��c��d�ȣ����ص�ֵ���ᰴ����Ӧ��˳����л��֡�
X_train, X_test,y_train,y_test = train_test_split(X ,y,test_size=0.2,random_state=1234)

#scale
#sc������һ�����ʵ����
#StandardScaler() �� scikit-learn ���е�һ���࣬���ڶ����ݽ��б�׼�������������·�����
#fit(X[, y])�����ڼ������ݵľ�ֵ�ͱ�׼������䱣���� StandardScaler ʵ���С�
#fit_transform(X)����������ݣ�Ȼ������ݽ��б�׼�����������ر�׼��������ݡ�
        #fit_transform(X)����ôʹ�õģ�
        # ������һ���������� X��������ѵ�����ݵ�����
        #X = [[1, 2],
        #     [3, 4],
        #     [5, 6]]
        ## ����һ�� StandardScaler ����
        #scaler = StandardScaler()

        ## ʹ�� fit_transform() ���������ݽ�����Ϻͱ�׼�����������ر�׼���������
        #X_scaled = scaler.fit_transform(X)

        #print("��׼��������ݣ�")
        #print(X_scaled)
        ##��׼��������ݣ�
        #[[-1.22474487 -1.22474487]
        # [ 0.          0.        ]
        # [ 1.22474487  1.22474487]]
#transform(X)��ʹ���ѱ���ľ�ֵ�ͱ�׼������ݽ��б�׼���������ر�׼��������ݡ�

#����	            ���������	     ��׼������	        ����ֵ               ���ý׶�
#fit_transform(X)    	��	            ��	           ��׼���������        ѵ���׶�
#transform(X)	   ������ϣ�	        ��	           ��׼���������        ���Խ׶�
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#��numpyת��Ϊ����
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#��1ά����ת��Ϊ2ά����
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#1��ģ��
# f = wx + b, sigmoid at the end
# ����һ���߼��ع�ģ�ͣ��̳��� nn.Module ��
class LogisticRegression(nn.Module):
    # ��ʼ����������������������������Ϊ����
    def __init__(self,n_input_features):
         # ���ø���ĳ�ʼ������
        super(LogisticRegression, self).__init__()
        #������һ�����Բ㣬������������Ϊ n_input_features�������������Ϊ 1
        self.linear = nn.Linear(n_input_features,1)

    def forward(self, x):
        # ������������� sigmoid �������ת��Ϊ�������
        y_predicted = torch.sigmoid(self.linear(x))
        # ���ؾ��� sigmoid ���������
        return y_predicted

# ʵ����һ���߼��ع�ģ�ͣ������������������� n_features
model = LogisticRegression(n_features)

#2����ʧ���Ż���
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#3��ѵ��ѭ��

num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    #backward pass 
    loss.backward()
    #updates
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if(epoch+1) %10 == 0:
        print(f"{epoch+1},loss= {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")