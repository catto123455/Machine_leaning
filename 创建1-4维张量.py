import torch

#创建一个1维数组
t1 = torch.tensor([1,2,3,4])

#创建一个2维数组
t2 = torch.tensor([[1,2,3],[4,5,6]])

#创建一个3维数组
t3 = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])

#创建一个4维数组
t4 = torch.tensor([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,0],[1,2]],[[3,4],[5,6]]]])

print(t1)
print(t2)
print(t3)
print(t4)
