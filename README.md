# Machine_leaning
startup
学习参考：①gatekeeper.ai alpha  ②BiliBili：五分钟机器学习,③csdn：K同学啊

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

### 4.1定义：它是一种基于树结构的模型，可以通过一系列的规则对数据进行分类或预测。
<br>决策树的过程：
<br>步骤1：过滤所有可能得决策条件
<br>步骤2：选择使子节点熵最小的决策条件（即最大熵增益）
<br>步骤3：重复1,2，直到
<br>1.到达了预先设置的最大树的深度
<br>2.每个子节点的样本都属于一类
### 4.2通俗概念理解：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/91325edd-5d58-4e8f-ad0c-41d5499d4a1b)
<br>根据原有动物训练集的判定标准，判定待测样本羊的类别信息
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/e93cf02b-10bc-45b6-bc31-7375dd0adebd)
<br>医疗判断有误时，可以像树一样追溯之前判断错误的信息
### 4.3作用及缺陷
<br>作用：迭代二分器3（ID3 ALRITHM）分类，分类和回归树[CART（classfication and regression tree）]回归
<br>优点：
<br>①直观，可视化，好理解
<br>②易于追溯和倒推
<br>缺点：
<br>对于树的最大深度这个预制参数很敏感
<br>深度太大，可能overfit
<br>深度太小，可能underfit    
### 4.4数学表达式
<br>4.4.1如何选择最优的决策条件
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/1db32e0a-294d-462f-9363-cb2af7bc2504)
<br>熵-entropy：衡量一个节点内的不确定性。  不确定性越高，熵越高。
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/68d16cff-57c9-4ac5-b017-2a1880930fa1)
<br>熵的增益-entropy gain=上一层的熵 - 当前一层的熵的总和
<br>以上为ID3 ALRITHM模型，只适用于分类问题
### 4.5训练用代码，待补充
# 5.聚类-decision tree
### 5.1定义：聚类是针对给定的样本，依据他们特征的相似度或距离，将其归并到若干个”类“或”簇“的数据分析问题。
### 5.2通俗概念理解：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/cca75bab-9002-4ee3-a900-6e7273953804)
<br>初入大学时学生进入不同的社团
<br>如何定义相似？
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/04331922-e51a-477c-8c2c-b424f37ee9d1)
<br>用两个点表示在欧式空间中的距离，距离越近，表示这两个点越相似，距离越远越不相似。
<br>cluster，属于同一个cluster的都是一类。
<br>Centroid for the current cluster，当前簇的中心。-训练时每增加一个样本就要变动一次簇的中心。
<br>聚类kmeans的训练过程
<br>1.随机从数据集中选取k个样本当做centroid
<br>2.对于数据集中的每个点，计算它距离每个centroid的距离，并把它归为距离最近的那个cluster
<br>3.更新centroid位置
<br>4.重复2,3，直到centroid的位置不再改变。
### 5.3作用及缺陷
<br>作用：分类
<br>优点：非监督学习类的算法不需要样本的标注信息
<br>缺点：  
<br>①不能利用到数据的标注信息，意味着模型的性能不如其他监督学习
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/cb84b354-b8b2-482f-8f08-f4e256e4bb24)
<br>②对于k的取值，也就是你认为数据集中的样本应该分为几类，这个参数的设置极为敏感
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/20d965e6-d833-4a26-8eba-fb97ba4b5695)
<br>③对于数据集本身的样本分布也很敏感
### 5.4数学表达式
<br>K均值（K-Means）：
<br>假设有 n 个数据样本 x1,x2,...,xn，每个样本具有 m 个特征。
<br>将数据集分为 k 个簇（簇的数目事先设定）。
<br>目标是最小化每个样本与其所属簇中心点之间的距离的总和。
<br>簇中心点用c1,c2,...,ck 表示，通过以下方式计算：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/0580aa47-4a82-4635-9640-36cda08cc374)
<br>其中 Si 是第 i 个簇的样本集合。
<br>然后，将每个样本分配给距离其最近的簇中心点所在的簇。
### 5.5训练用代码，待补充
# 6.向量支持机-SVM
### 6.1定义：支持向量机（SVM）是一种监督学习算法，主要用于解决分类问题
### 6.2作用及缺陷
<br>作用：分类
![image](https://github.com/catto123455/Machine_leaning/assets/140484656/2501c5f5-478d-4dae-93bb-1b059c6f37dc)
<br>优点：强分类器，能保证最大化区分两个类别，所以模型的性能优异
<br>缺点：
<br>①hard-margin SVM是线性分类器，不能处理不同类别相互交融的情况
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/d0ec1524-7927-4932-811c-50b1d4267495)
<br>对于这种情况，可以引入容错率,保证大局上的正确性。这种模型也叫做soft-margin SVM
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/7ec44b81-3979-41e6-8e33-e21a0f220a6f)
<br>②线性分类器，不可用于求解线性完全不可分的情况。但可以引入kernel，在更高维度解决数据不可分的情况
### 6.3数学表达式:
<br>假设给定一个训练数据集，包含了一组输入特征X={x1,x2,...,xn} 和对应的输出标签 y，其中 xi 表示第 i 个特征，y 表示类别标签，可以是正类（+1）或负类（-1）。
<br>SVM 的目标是找到一个超平面（或超平面组合），将不同类别的样本尽可能地分开，并且使得距离最近的样本点到超平面的距离最大化。
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/c067dc19-abc9-4d3e-a682-6bfe77709f1e)
<br>在线性可分的情况下，SVM 的数学表达式为：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/8a0629f9-e981-494b-a892-7b190587860b)
<br>subject to     yi(w⋅xi+b)≥1,i=1,2,...,n
<br>其中：
<br>w 是超平面的法向量。
<br>b 是超平面的截距。
<br>∥w∥^2 表示超平面的法向量 w 的模的平方。
<br>xi 表示训练集中的第 i 个样本特征。
<br>yi 是训练集中第 i 个样本的类别标签，取值为 +1 或 -1。
<br>(w⋅xi+b) 表示样本 xi 到超平面的距离。
<br>约束条件 yi(w⋅xi+b)≥1 表示所有样本都距离超平面至少为 11，保证了样本的分类正确性。
### 6.4训练用代码，待补充
# 7.随机森林-random forest
### 7.1定义：随机森林（Random Forest）是一种基于集成学习的分类和回归算法。它通过构建多个决策树并组合它们的预测结果来提高整体模型的性能。随机森林的基本思想是：通过随机选择特征和样本进行训练，构建多个决策树，并通过投票或平均等方式来整合这些决策树的预测结果。
### 7.2通俗概念理解：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/eb38f726-926b-4724-8098-19c8b9c02005)
<br>三个学渣各取优点打败学霸。
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/9c82ff43-029f-433a-b516-d05777203e1d)
<br>集成学习，集成弱监督学习模型，形成一个全科学霸模型
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/ca8b0339-8037-427a-879a-a6651d67ef8a)
### 7.3作用及缺陷
<br>作用：分类，回归
<br>优点：  
<br>①模型随机性强，不易overfit
<br>kangzao对异常点outlier不敏感
<br>②处理高维数据相对更快
<br>③树状结构，模型可解释度高，可以告诉你每个特征的重要性
<br>缺点：
<br>①模型往往过于general，不具备正确处理过于困难的样本的能力。
<br>②模型起点高，但是天花板低
### 7.4数学表达式：是对决策树模型的集成。
### 7.5训练用代码，待补充
# 8.自适应增强算法-adaboost
### 8.1定义：自适应增强（Adaptive Boosting，简称 AdaBoost）是一种集成学习方法，通过组合多个弱学习器来构建一个更强大的分类器。它是一种迭代算法，每一轮迭代中都会根据前一轮分类器的性能对数据进行调整，以便更好地拟合数据。
### 8.2通俗概念理解：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/42caaade-b644-401e-8e43-5914f6b6bb1d)
<br>弱化模型处理好的，集中处理模型处理不好的，优化模型，最终使模型既能解决简单问题，也能解决复杂问题。
### 8.3作用及缺陷
<br>作用：
<br>优点：顺序，级联的结果+利用权重组合结果，使模型擅长解决困难问题，模型性能的天花板高
<br>缺点：
<br>①容易过拟合-overfit
<br>②对于异常点outlier过于敏感
<br>③模型的性能起点低，速度慢
### 8.4数学表达式：
<br>假设给定一个训练数据集D={(x1,y1),(x2,y2),...,(xn,yn)}，其中 xi 表示第 i 个样本的特征， yi 表示第 i 个样本的类别标签，通常为 +1 或 -1。
<br>初始化样本权重：
<br>初始化样本权重为相等的值，即对于所有样本 i，初始权重为wi=n1。
<br>迭代训练基本分类器：
<br>对于每一轮迭代 t=1,2,...,T：
<br>a. 使用当前样本权重训练一个基本分类器 ht(x)，该分类器在第t 轮中应对样本的分类效果较好。
<br>b. 计算基本分类器的误差率（错误分类的样本权重之和）：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/86c82ae6-db2e-45de-b5f9-eeb997419273)
<br>1. 其中，Wi^(t)表示第 t 轮迭代中样本 i 的权重，1(⋅) 是指示函数，表示分类是否正确。
<br>2. c. 计算基本分类器的权重系数：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/24e55903-50b3-4acc-9b3f-3de98de6be88)
<br>3. d. 更新样本权重，以便下一轮迭代时更关注分类错误的样本：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/748ea559-3c08-4063-a7da-6251e88212f7)
<br>其中，Zt 是规范化因子，确保权重之和为 1：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/8a594e4b-ac5c-4836-90c4-121eeefbe629)
<br>组合基本分类器：
<br>将每个基本分类器按其权重系数进行加权组合，得到最终的强分类器：
<br>![image](https://github.com/catto123455/Machine_leaning/assets/140484656/dbd860ae-232e-40cb-99df-8ee8fa97819f)
<br>其中，H(x) 表示最终的强分类器，ht(x) 表示第 t 个基本分类器的预测结果，sign(⋅) 表示符号函数，即大于 0 返回 +1，小于 0 返回 -1，等于 0 返回 0。
### 8.5训练用代码，待补充
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
