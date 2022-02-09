import torch

PATH = '/data5/sukmin/kaggle/best.pt'
model = torch.load(PATH)

print(PATH)