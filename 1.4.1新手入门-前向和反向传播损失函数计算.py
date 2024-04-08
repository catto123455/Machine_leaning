#Backpropagation-反向传播


#————————————链式法则-chain rule

#                                     x→ a（x） → y →b（y）→z

#想要知道c相对于x的导数
#引入链式法则-chain rule公式：
#                                    dz   dz   dy
#                                    — = — * —
#                                    dx   dy   dx
#b（y）的输出相当于dz/dy
#a（x）的输出相当于dy/dx
#将b（y）*a（x）就会得到最终的梯度


#————————————计算图-computational Graph

#                                    dz      dx·y
#                                    ---  =  -----  = y
#                                    dx       dx   


#                                    x\
#                                      \
#                                        f=x*y → z   →0→0→Loss            
#                                      /
#                                    y/
#在xy相乘时可以计算局部梯度

#                                    dz      dx·y
#                                    ---  =  -----  = x
#                                    dy       dy
#我们必须在最开始时计算我们的最小化的损失函数   
#这个损失相对于我们的参数x的梯度

#最终的梯度
#                                    dLoss     dLoss      dz
#                                    -----  =  -----  *  -----
#                                     dx        dz        dx

#前向传播，应用所有函数并计算损失函数。
#在每个节点我们计算局部梯度
#然后进行反向传播，使用链式法则计算相对于权重或参数的损失梯度

#        △y=w·x              loss= (△y-y)^2=(wx-y)^2


#——————————前向传播-forward pass的过程
#x\
#  ↘          △y               s                loss
#     ( * )  -----→  ( - )  ------→  ( ^2 )   -------→
#  ↗           y^2↗
#w/

#局部    ↑            ↑                ↑
#梯度    d△y         ds               dloss
#       ----- -----→ -----  ------→  ------
#        dw           d△y                ds
#
#
                             
#——————————反向传播-backward pass的过程
#x\
#  \           △y                s                loss
#     ( * )  ←-----  ( - )  ←------  ( ^2 )   ←-------
#   /           y
#w↙

#局部   ↑              ↑               ↑
#梯度  dloss          dloss             dloss
#     ------   ←---  ------  ←-----  ------
#       dw             d△y               ds
#


#                  dLoss
#minimize Loss → --------
#                    dw


#——————————前向传播-forward pass代入具体数值 x=1 y=2 w=1
#x\
#  ↘          △y               s                loss
#     ( * )  -----→  ( - )  ------→  ( ^2 )   -------→
#  ↗           y↗
#w/

#-------------先计算loss值
# △y = x * w = 1 * 1 =1

# s = △y - y^2 =  1 - 2 = -1

# loss = s^2 = (-1)^2 = 1
#-------------再计算反向传播的局部梯度·关于s，loss相对于^2的梯度

#  dloss    ds^2
#  ----- = ------ = 2s    #2s为什么这个公式最后的结果是2s而不是s，微积分运算公式，d( x^n )/ dx =nx^(n-1)即2s^(2-1)=2s
#    ds      ds

#-------------再计算下一个局部梯度·s相对于^2的梯度

#   ds      d△y-y       
#  ------ = ---------- = 1      #为什么结果是1而不是-1，由于平方，(-1)^2 = 1，问题又来了，哪里来的平方？？？
#   d△y      d△y          

#-------------再计算下一个局部梯度^y相对于w的梯度
# d△y     dwx
#------ = ----- = x
#  dw      dw

#————————根据链式法则求解最终的梯度

#  dloss     dloss    ds
#  ------ = -------·-----  = 2·s·1 = -2
#   d△y      ds      d△y

#  dloss      dloss   d△y
#  ------ = -------·-------  = -2·x = -2
#   dw        d△y    dw

import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad=True)

#forward pass and compute the loss
y_hat = w * x
loss = ( y_hat - y )**2

print(loss)

#backward pass 
loss.backward()
print(w.grad)

### update weights
### next forward and backwars
