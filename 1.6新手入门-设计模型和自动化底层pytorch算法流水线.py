#1）设计模型（输入和输出的大小，不同层的前向传播）
#2）构造损失-loss和优化器-optimizer
#3) 训练循环
# - forward pass：计算预测
# - backward psss：梯度
# - update weights 
import torch
import torch.nn as nn

# f = w * x
# f = 2 * x

X = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)
n_sample, n_features = X.shape
print(n_sample, n_features)

input_size = n_features
output_size = n_features

#model = nn.Linear(input_size, output_size)

class LinearRegration(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegration, self).__init__()
        #difine layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

model = LinearRegration(input_size, output_size)

print(f"Prediction before training:f(5)={model(X_test).item():.3f}")

learning_rate = 0.01
n_iters = 100   #相当于epoch总轮次

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradient = backward pass
    l.backward()   
    # dl/dw  pytorch自己会完成所有的计算，调用即可

    # update weights
    optimizer.step()


    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:      #每多少步输出一次
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction after training:f(5) = {model(X_test).item():.3f}")


