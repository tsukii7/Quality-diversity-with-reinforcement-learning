# show_pkl.py

import pickle
import torch
path = 'target_net.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = torch.load(f)

print(data)
print(len(data))

