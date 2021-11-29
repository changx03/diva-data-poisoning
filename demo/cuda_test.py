import torch

print('Is CUDA available?', torch.cuda.is_available())

device = torch.device('cuda')
print('Device:', device)
tensor1 = torch.randn(5, 5).to(device)
tensor2 = torch.randn(5, 5).to(device)
tensor3 = torch.matmul(tensor1, tensor2)
print('Pass test?', tensor3.is_cuda)
