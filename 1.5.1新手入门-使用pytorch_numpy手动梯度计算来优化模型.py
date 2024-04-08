#使用梯度下降算法优化函数
import numpy as np

# f = w * x

# f = 2 * x

X = torch.tensor([1,2,3,4],dtype=torch.float32)
Y = torch.tensor([2,4,6,8],dtype=torch.float32)


w = torch.tensor(0.0,dtype=torch.float32)

# model prediction
def forward(x):
    return w * x
# loss
def loss(y, y_predicted):
    return((y_predicted - y)**2).mean()      #mean()是均值运算

# gradient
# MSE = 1/N * (w*x-y)**2
# dj/dw =1/N 2x(w*x-y)

#def gradient(x,y,y_predicted):        #手动计算的梯度
#    return np.dot(2*x, y_predicted - y).mean()

print(f"Prediction before training:f(5)={forward(5):.3f}")

learning_rate = 0.01
n_iters = 20   #相当于epoch

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradient = backward pass
    l.backward() # dl/dw  pytorch自己会完成所有的计算，调用即可

    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients

    if epoch % 2 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training:f(5)={forward(5):.3f}")
