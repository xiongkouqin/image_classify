import model 
import model1
import torch 

net1 = model.resnet34()

net2 = model1.resnet34()

model_weight_path = '/home/xiongkouqin/projects/image_classify/6_resnet/resnet34-pretrain.pth'

mk1, uk1 = net1.load_state_dict(torch.load(model_weight_path), strict=False)
print(f"Missing Keys for My Model: \n{mk1}")
print(f"Unexpected Keys for My Model: \n{uk1}")

mk2, uk2 = net2.load_state_dict(torch.load(model_weight_path), strict=False)
print(f"Missing Keys for My Model: \n{mk2}")
print(f"Unexpected Keys for My Model: \n{uk2}")