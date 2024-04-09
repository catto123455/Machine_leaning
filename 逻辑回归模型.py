#1）设计模型（输入和输出的大小，不同层的前向传播）
#2）构造损失-loss和优化器-optimizer
#3) 训练循环
# - forward pass：计算预测
# - backward psss：梯度
# - update weights 

from typing_extensions import _AnnotatedAlias
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0）准备数据
bc = datasets.load_breast_cancer()
# 加载乳腺癌数据集，并将特征数据赋给 X，目标标签赋给 y。
#特征矩阵X的bc.date是示例：（569,30），目标向量y的bc.target的形状（569，）
X, y = bc.data,bc.target

#元组解包给n_samples, n_features赋值
n_samples, n_features = X.shape
print(n_samples, n_features)

#【将数据集划分为训练集和测试集-train_test_split 函数】，它是用于将数据集划分为训练集和测试集的工具。
#X: 特征矩阵，包含了所有的输入特征。
#y: 目标向量，包含了所有的目标值。
#test_size: 测试集的大小，可以设置为浮点数（表示测试集占总样本数的比例）或整数（表示测试集样本的数量）。
#           在这里，test_size=0.2 表示将 20% 的样本划分为测试集。
#random_state: 随机种子，用于确保每次划分的结果都是相同的。
#              如果不设置该参数，每次运行代码时将会得到不同的划分结果。
#返回的值：
#X_train: 第一个返回值用于接收训练集的特征矩阵
#X_test: 第二个返回值用于接收训练集的目标向量
#y_train: 第三个返回值用于接收训练集的目标向量
#y_test: 第四个返回值用于接收测试集的目标向量
#无论变量名是什么，如a，b，c，d等，返回的值都会按照相应的顺序进行划分。
X_train, X_test,y_train,y_test = train_test_split(X ,y,test_size=0.2,random_state=1234)

#scale
#sc创建了一个类的实例，
#StandardScaler() 是 scikit-learn 库中的一个类，用于对数据进行标准化处理。具有以下方法：
#fit(X[, y])：用于计算数据的均值和标准差，并将其保存在 StandardScaler 实例中。
#fit_transform(X)：先拟合数据，然后对数据进行标准化处理，并返回标准化后的数据。
        #fit_transform(X)是怎么使用的：
        # 假设有一个特征矩阵 X，包含了训练数据的特征
        #X = [[1, 2],
        #     [3, 4],
        #     [5, 6]]
        ## 创建一个 StandardScaler 对象
        #scaler = StandardScaler()

        ## 使用 fit_transform() 方法对数据进行拟合和标准化处理，并返回标准化后的数据
        #X_scaled = scaler.fit_transform(X)

        #print("标准化后的数据：")
        #print(X_scaled)
        ##标准化后的数据：
        #[[-1.22474487 -1.22474487]
        # [ 0.          0.        ]
        # [ 1.22474487  1.22474487]]
#transform(X)：使用已保存的均值和标准差对数据进行标准化处理，返回标准化后的数据。

#方法	            对数据拟合	     标准化处理	        返回值               适用阶段
#fit_transform(X)    	是	            是	           标准化后的数据        训练阶段
#transform(X)	   否（已拟合）	        是	           标准化后的数据        测试阶段
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#从numpy转变为张量
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#从1维数组转变为2维数组
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#1）模型
# f = wx + b, sigmoid at the end
# 定义一个逻辑回归模型，继承自 nn.Module 类
class LogisticRegression(nn.Module):
    # 初始化方法，接受输入特征的数量作为参数
    def __init__(self,n_input_features):
         # 调用父类的初始化方法
        super(LogisticRegression, self).__init__()
        #定义了一个线性层，输入特征数量为 n_input_features，输出特征数量为 1
        self.linear = nn.Linear(n_input_features,1)

    def forward(self, x):
        # 对线性输出进行 sigmoid 激活，将其转换为概率输出
        y_predicted = torch.sigmoid(self.linear(x))
        # 返回经过 sigmoid 激活后的输出
        return y_predicted

# 实例化一个逻辑回归模型，传入输入特征的数量 n_features
model = LogisticRegression(n_features)

#2）损失和优化器
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#3）训练循环

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