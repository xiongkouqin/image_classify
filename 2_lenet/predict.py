from torchvision import transforms
import torch 

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

from model import LeNet

# construct model and load trained parameters
net = LeNet()
net.load_state_dict(torch.load('/home/xiongkouqin/projects/image_classify/2_lenet/net.pth'))

from PIL import Image

# load the image and convert it to how we take a training image
im = Image.open('/home/xiongkouqin/projects/image_classify/2_lenet/cat.jpg')
im = transform(im)
im = torch.unsqueeze(im, dim=0)

with torch.no_grad():
    outputs = net(im)
    predict_y = torch.max(outputs, dim=1)[1]
    print(predict_y)
    print(classes[predict_y]) # 预测错了


