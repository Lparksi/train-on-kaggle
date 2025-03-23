import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, datasets
from tqdm import tqdm
import json
from torchvision.models import resnext101_32x8d
from config import Config

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def get_transform(phase):
    """根据阶段生成数据变换"""
    cfg = Config.data_transform[phase]
    transforms_list = []
    
    if phase == "train":
        if "RandomResizedCrop" in cfg:
            transforms_list.append(transforms.RandomResizedCrop(cfg["RandomResizedCrop"]))
        if "RandomHorizontalFlip" in cfg and cfg["RandomHorizontalFlip"]:
            transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomRotation(15))  # 添加随机旋转增强
    else:  # val
        if "Resize" in cfg:
            transforms_list.append(transforms.Resize(cfg["Resize"]))
        if "CenterCrop" in cfg:
            transforms_list.append(transforms.CenterCrop(cfg["CenterCrop"]))
    
    transforms_list.append(transforms.Grayscale(num_output_channels=3))  # 转换为灰度并复制为三通道
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=cfg["Normalize"]["mean"], 
                                              std=cfg["Normalize"]["std"]))
    
    return transforms.Compose(transforms_list)

def train(rank, world_size):
    """训练函数"""
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"Using {world_size} GPUs for training.")

    # 数据加载
    train_dataset = datasets.ImageFolder(
        root=os.path.join(Config.dataset["root"], Config.dataset["train_dir"]),
        transform=get_transform("train")
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    nw = min([os.cpu_count(), Config.batch_size if Config.batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        sampler=train_sampler,
        num_workers=nw
    )

    if rank == 0:
        validate_dataset = datasets.ImageFolder(
            root=os.path.join(Config.dataset["root"], Config.dataset["val_dir"]),
            transform=get_transform("val")
        )
        validate_loader = torch.utils.data.DataLoader(
            validate_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=nw
        )
        print(f"Number of classes: {len(train_dataset.class_to_idx)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(validate_dataset)}")

        cla_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json.dumps(cla_dict, indent=4))

    # 模型加载
    model = resnext101_32x8d(pretrained=True)  # 使用预训练模型并自动下载
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, Config.dataset["num_classes"])  # 调整分类层
    
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scaler = torch.amp.GradScaler('cuda')

    # 检查点加载
    start_epoch = 0
    best_acc = 0.0
    if Config.resume:
        checkpoint = torch.load(Config.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        if rank == 0:
            print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.3f}")

    # 训练循环
    for epoch in range(start_epoch, Config.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, disable=rank != 0)
        
        for data in train_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{Config.epochs}] loss:{loss:.3f}"

        dist.barrier()

        if rank == 0:
            model.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = model(val_images)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels).sum().item()
                    val_bar.desc = f"valid epoch[{epoch+1}/{Config.epochs}]"

            val_accurate = acc / len(validate_dataset)
            train_loss = running_loss / len(train_loader)
            print(f'[epoch {epoch+1}] train_loss: {train_loss:.3f} val_accuracy: {val_accurate:.3f}')

            if val_accurate > best_acc:
                best_acc = val_accurate
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(checkpoint, Config.save_path)
                print(f"New best accuracy: {best_acc:.3f}, model saved.")

    if rank == 0:
        print('Finished Training')
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 1)

if __name__ == '__main__':
    main()
