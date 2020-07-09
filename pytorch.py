import torch

a = torch.tensor([1, 2, 3, 4, 5, 6])

print(a)
print(a.type)
print(a.dtype)
print(a.view(-1, 2))
print(a.type)
print(a.dtype)

x = torch.tensor(5., requires_grad=True)

y = x**2

y.backward()
print(x.grad)

u = torch.tensor(1., requires_grad=True)
v = torch.tensor(2., requires_grad=True)

f = u*v + u**2

f.backward()

print("derivative w.r.t. :u:", u.grad)
print("derivative w.r.t. :v:", v.grad)