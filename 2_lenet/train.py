from collections.abc import Iterable, Iterator
import torch.utils
import torch.utils.data
import torchvision.transforms as tf
import torchvision
import torch 

# 图片预处理
transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 50000 张训练图片
# 下载过download = False就可以
train_set = torchvision.datasets.CIFAR10('/home/xiongkouqin/projects/data', train=True, download=False, transform=transform)

import os
num_workers = os.cpu_count()
batch_size = 32


# batch_size: how many we got on each iteration
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# same way to construct test data set
test_set = torchvision.datasets.CIFAR10('/home/xiongkouqin/projects/data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=len(test_set),
    num_workers=num_workers
)

# 里面最后的label都用数字来表示了，我们写一个每个类表示什么
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# import sys
# print(sys.path)
# ['/home/xiongkouqin/projects/image_classify/2_1_lenet',...

from model import LeNet

net = LeNet()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

print(isinstance(test_loader, Iterable))
print(isinstance(test_loader, Iterator))

test_images, test_labels = next(iter(test_loader))

num_epochs = 40
for epoch in range(num_epochs):
    training_loss = 0.0
    for step, data in enumerate(train_loader, start=1):
        inputs, labels = data
        optimizer.zero_grad() # clear the gradients
        outputs = net(inputs) # calculate the ouput 
        loss = loss_function(outputs, labels) # calculate the loss 
        loss.backward() # get gradients 
        optimizer.step() # back propagation 
        training_loss += loss.item()
        
        if step % 500 == 0:
            with torch.no_grad():
                outputs = net(test_images)
                # print(outputs.shape) # torch.Size([10000, 10])
                predict_y = torch.max(outputs, dim=1)[1] # max返回值[0] values, [1] indices 
                accuracy = torch.eq(predict_y, test_labels).sum().item() / test_labels.size(0) # size(0)获取的是第一个纬度的size
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % (epoch + 1, step, training_loss / 500, accuracy))   
                training_loss = 0.0  
                
                
path_to_save_model = '/home/xiongkouqin/projects/image_classify/2_lenet/net.pth'  
torch.save(net.state_dict(), path_to_save_model)