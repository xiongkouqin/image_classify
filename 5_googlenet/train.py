import torch 
import torch.utils
import torch.utils.data
import torchvision.transforms as tf
import os
import torchvision.datasets as datasets
import json
from model import GoogLeNet
import torch.nn as nn
from tqdm import tqdm


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"training on device {device}")
    
    data_transform = {
        "train": tf.Compose([
            tf.RandomResizedCrop(224), # can be int, it will be used for both height and width  
            tf.RandomHorizontalFlip(),
            tf.ToTensor(),
            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": tf.Compose([
            tf.Resize((224, 224)), # has to be tuple, otherwise it will keep the ratio 
            tf.ToTensor(),
            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    project_root = '/home/xiongkouqin/projects/image_classify/5_googlenet'
    data_root = '/home/xiongkouqin/projects/data'
    image_path = os.path.join(data_root, 'flower_data')
    assert os.path.exists(image_path), f'{image_path} does not exist!'
    
    train_dataset = datasets.ImageFolder(os.path.join(image_path, 'train'), transform=data_transform["train"])
    train_num = len(train_dataset)
    
    cls_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
    json_str = json.dumps(cls_dict, indent=4)
    with open(os.path.join(project_root, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)
        
    batch_size = 32
    num_workers = min(os.cpu_count(), batch_size)
    print(f'using {num_workers} workers to load data')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    
    val_dataset = datasets.ImageFolder(os.path.join(image_path, 'val'), transform=data_transform['val'])
    val_num = len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    
    print(f'using {train_num} images for training, {val_num} images for validation')
    
    # 暂时跳过一下 imshow 
    
    net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
    net.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer =  torch.optim.Adam(net.parameters(), lr=0.001)
    
    num_epochs = 10
    save_path = os.path.join(project_root, 'googlenet.pth')
    best_acc = 0.0
    train_steps = len(train_loader) # for each epoch, how mnay batches are there
    
    for epoch in range(1, num_epochs+1):
        # train
        net.train() # make the model in training mode, have effects on for example, Dropout, BatchNorm
        running_loss = 0.0
        train_bar = tqdm(train_loader,ncols=120)
        for step, data in enumerate(train_bar):
            images, labels = data 
            optimizer.zero_grad()
            logits, aux_logits1, aux_logits2 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + 0.3 * loss1 + 0.3 * loss2
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            train_bar.desc = f'train epoch {epoch}/{num_epochs} loss:{loss:.3f}'
            
        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, ncols=120)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum()
                val_bar.desc = f'val epoch {epoch}/{num_epochs} correct_num:{acc}'
        
        val_accuracy = acc / val_num
        print(f'epoch [{epoch}] train_loss: {running_loss/train_steps:.3f}, val_accuracy: {val_accuracy:.3f}')
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
    
    print('Finish Training.')

if __name__ == '__main__':
    main()