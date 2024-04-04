# Machine_leaning
startup
学习参考：①gatekeeper.ai alpha  ②BiliBili五分钟机器学习

# 1.线性回归-liner regression
1.1定义：线性回归是一种统计学方法，用于建立输入变量（自变量）和输出变量（因变量）之间的线性关系。
1.2通俗概念理解：
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/107f3170-98fd-4fb8-b24e-7a1c76643eec)
股票的未来走向估算
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/72088859-1f30-494b-b4a2-74a88582af90)
根据已知站位推测金发男孩的站位
1.3作用及缺陷
作用：分析预测数值
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/7623e59e-ef77-430b-a1f2-a2e7822006b6)
缺陷：没有对非线性数据求解的能力
1.4数学表达式
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/a6541224-bb93-4494-ba43-2dde4e6390ed)
y=Ax+b
也可以构建更加复杂的线性回归：
y=β0+β1x1+β2x2+...+βnxn+ε
包含更多的属性x（也叫特征）
例如，一个预测股价Y的任务
属性X可能包含[时间，股票类别，上市时间]等等
1.5训练用代码，待补充
# 2.逻辑回归-logistic regression
2.1定义：
2.2通俗概念理解：
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/2cf8a865-3c0c-4ef4-a45e-f60ea15faac5)
根据一个学生的基本信息判断他是否是三好学生
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/6d2c32c4-cd97-41a1-b685-cc2e8d10d1ec)
判断股票是涨还跌，买还是不买
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/336e853e-8631-44b5-9da3-245d84890a5e)
判断病患是不是有病
2.3作用及缺陷
作用：
①classification-分类算法，
②表示概率p只能在0~1之间表示概率，比如0,0.3,0.99但不可以是-1，1.2等
缺陷：只适合线性分布
2.4数学表达式：
特征向量 x=(x1,x2,...,xn)x=(x1,x2,...,xn)
权重向量 w=(w1,w2,...,wn)w=(w1,w2,...,wn)
有一个偏置项b
逻辑回归将输入特征通过一个 sigmoid 函数映射到一个在 0 到 1 之间的概率值。
sigmoid 函数的数学表达式如下：
σ(z)=1/1+e^−z
其中，z 是输入的线性组合，表示为：
z=b+w1x1+w2x2+...+wnxn
综合起来，逻辑回归模型的数学表达式可以写为：
P(y=1∣x)=σ(z)=1/1+e^−z
其中 P(y=1∣x) 表示给定输入特征 x 下输出为类别 y=1 的概率。
2.5训练用代码，待补充
# 3.K近邻算法-KNN-K-Nearest Neighbors
3.1定义：
3.2通俗概念理解：
3.3作用及缺陷
3.4数学表达式
3.5训练用代码，待补充
# 4.决策树-decision tree
4.1定义：
4.2通俗概念理解：
4.3作用及缺陷
4.4数学表达式
4.5训练用代码，待补充
# 5.聚类-decision tree
5.1定义：
5.2通俗概念理解：
5.3作用及缺陷
5.4数学表达式
5.5训练用代码，待补充
# 6.向量支持机-SVM
6.1定义：
6.2通俗概念理解：
6.3作用及缺陷
6.4数学表达式
6.5训练用代码，待补充
# 7.随机森林-random forest
7.1定义：
7.2通俗概念理解：
7.3作用及缺陷
7.4数学表达式
7.5训练用代码，待补充
# 8.自适应增强算法-adaboost
8.1定义：
8.2通俗概念理解：
8.3作用及缺陷
8.4数学表达式
8.5训练用代码，待补充
# 9.生成对抗网络-GAN
9.1定义：
9.2通俗概念理解：
9.3作用及缺陷
9.4数学表达式
9.5训练用代码，待补充
# 10.稳定扩散模型-diffusion modle
10.1定义：
10.2通俗概念理解：
10.3作用及缺陷
10.4数学表达式
10.5训练用代码，待补充
# 11.VAE
11.1定义：
11.2通俗概念理解：
11.3作用及缺陷
11.4数学表达式
11.5训练用代码，待补充
