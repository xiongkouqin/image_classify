import torch 
import torchvision.transforms as tf 
from PIL import Image
import os
import json 
from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transform = tf.Compose([
        tf.Resize((224, 224)),
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # load image
    image_path = '/home/xiongkouqin/projects/image_classify/3_alexnet/tulip.jpg'
    assert os.path.exists(image_path), f'image file {image_path} does not exist!'
    im = Image.open(image_path)
    im = data_transform(im)
    im = torch.unsqueeze(im, dim=0)
    
    # load index to class name mapping 
    json_path = '/home/xiongkouqin/projects/image_classify/4_vgg/class_indices.json'
    assert os.path.exists(json_path), f'json file {json_path} does not exist!'
    with open(json_path, 'r') as f:
        index_to_class = json.load(f) # a dict 
    # use index_to_class is allowed 
        
    # load the model
    model_name = 'vgg16'
    net = vgg(model_name='vgg16', num_classes=5, init_weights=False)
    net_path = f'/home/xiongkouqin/projects/image_classify/4_vgg/{model_name}_net.pth'
    assert os.path.exists(net_path), f'net path {net_path} does not exist!'
    net.load_state_dict(torch.load(net_path))
    
    net.eval()
    with torch.no_grad():
        output = net(im)
        predict_y = torch.max(output, dim=1)[1] # index 
        predict_prob = torch.softmax(output,dim=1)[0][predict_y]
    
    print(f"predict result {index_to_class[str(predict_y.item())]}, with probability {predict_prob.item()*100:.2f}%")

if __name__ == '__main__':
    main()