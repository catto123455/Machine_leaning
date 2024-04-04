# Machine_leaning
startup
学习参考：①gatekeeper.ai alpha  ②BiliBili五分钟机器学习

# 1.线性回归-liner regression
### 1.1定义：线性回归是一种统计学方法，用于建立输入变量（自变量）和输出变量（因变量）之间的线性关系。
### 1.2通俗概念理解：
<br> ![image](https://github.com/catto123455/Machine_leaning/assets/140484656/107f3170-98fd-4fb8-b24e-7a1c76643eec)
<br>股票的未来走向估算
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/72088859-1f30-494b-b4a2-74a88582af90)
<br>根据已知站位推测金发男孩的站位
### 1.3作用及缺陷
<br>作用：分析预测数值
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/7623e59e-ef77-430b-a1f2-a2e7822006b6)
<br>缺陷：没有对非线性数据求解的能力
### 1.4数学表达式
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/a6541224-bb93-4494-ba43-2dde4e6390ed)
<br>y=Ax+b
<br>也可以构建更加复杂的线性回归：
<br>y=β0+β1x1+β2x2+...+βnxn+ε
<br>包含更多的属性x（也叫特征）
<br>例如，一个预测股价Y的任务
<br>属性X可能包含[时间，股票类别，上市时间]等等
### 1.5训练用代码，待补充
# 2.逻辑回归-logistic regression
### 2.1定义：逻辑回归是一种用于处理分类问题的统计学习方法
### 2.2通俗概念理解：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/2cf8a865-3c0c-4ef4-a45e-f60ea15faac5)
<br>根据一个学生的基本信息判断他是否是三好学生
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/6d2c32c4-cd97-41a1-b685-cc2e8d10d1ec)
<br>判断股票是涨还跌，买还是不买
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/336e853e-8631-44b5-9da3-245d84890a5e)
<br>判断病患是不是有病
### 2.3作用及缺陷
<br>作用：
<br>①classification-分类算法，
<br>②表示概率p只能在0~1之间表示概率，比如0,0.3,0.99但不可以是-1，1.2等
<br>缺陷：只适合线性分布
### 2.4数学表达式：
<br>特征向量 x=(x1,x2,...,xn)x=(x1,x2,...,xn)
<br>权重向量 w=(w1,w2,...,wn)w=(w1,w2,...,wn)
<br>有一个偏置项b
<br>逻辑回归将输入特征通过一个 sigmoid 函数映射到一个在 0 到 1 之间的概率值。
<br>sigmoid 函数的数学表达式如下：
<br>σ(z)=1/1+e^−z
<br>其中，z 是输入的线性组合，表示为：
<br>z=b+w1x1+w2x2+...+wnxn
<br>综合起来，逻辑回归模型的数学表达式可以写为：
<br>P(y=1∣x)=σ(z)=1/1+e^−z
<br>其中 P(y=1∣x) 表示给定输入特征 x 下输出为类别 y=1 的概率。
### 2.5训练用代码，待补充
# 3.K近邻算法-KNN-K-Nearest Neighbors
### 3.1定义：K最近邻（K-Nearest Neighbors，简称KNN）算法是一种基本的分类和回归方法，常用于模式识别和数据挖掘领域。它是一种非参数化的、懒惰学习（lazy learning）的方法，意味着它不会对训练数据进行显式的建模，而是通过存储训练数据的方式进行预测。
### 3.2通俗概念理解：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/b06f6677-6bce-4a49-90ad-12d37576caed)
<br>做决策的时候，会先看看周围的朋友是怎么选择的
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/072ee7c8-2085-4e9c-b935-7365bda4eff1)
<br>电影网站的推荐系统，根据已有用户的喜好，推荐新用户的喜好
### 3.3作用及缺陷
<br>作用：用于分类的评估矩阵，计算准确性，混淆矩阵，f统计等
<br>优点：
<br>  ①直观，好理解算法的逻辑
<br>  ②只观测待测样本的局部分布，不需要估算整体
<br>缺点：
<br>  ①局部估算可能不符合全局的分布
<br>  ②不能计算概率
<br>  ③对k的取值非常敏感，不同的k会得到不同的结果
<br>  ![image](https://github.com/catto123455/Machine_leaning/assets/140484656/363aa4ae-ba98-4c82-aaae-c850c81cba67)

### 3.4数学表达式
<br>假设有一个包含 m 个训练样本的数据集，每个样本包含 n 个特征，表示为 X={(x1,y1),(x2,y2),...,(xm,ym)}，其中 xi 是特征向量，yi 是对应的类别标签。
<br>给定一个新的样本 Xtest，KNN算法的基本思想是：
<br>1. 计算新样本 xtest 与所有训练样本 xi 之间的距离，通常使用欧氏距离或其他距离度量方法。
<br>2. 选择与新样本距离最近的 k 个训练样本。
<br>3. 根据这 k 个样本的类别标签，通过多数表决（majority voting）或加权投票的方式，确定新样本的类别。
<br>  KNN的数学表达式可以用以下伪代码表示：
<br>1. 计算距离：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/7a9a8ed3-8524-4453-990c-f4331edd30e0)
<br>其中 xtest,j 表示新样本的第 j 个特征，xi,j 表示第 i 个训练样本的第 j 个特征。
<br>1. 选择距离最近的 k 个样本：
<br>nearest_neighbors=argminxi distance(xtest,xi)
<br>1. 根据最近的 k 个样本的类别，确定新样本的类别：
<br>ypred=majority_vote({yi∣xi∈nearest_neighbors})
<br>其中 yi 表示训练样本xi 的类别标签。
### 3.5训练用代码，待补充
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
