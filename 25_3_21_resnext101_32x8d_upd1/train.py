import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from config import Config
from torchvision.models import resnext101_32x8d

# 自定义二值化变换类
class Binarize(object):
    """将灰度图像二值化为黑白图像"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, img):
        img = transforms.functional.to_grayscale(img, num_output_channels=1)
        img = torch.where(img > self.threshold, torch.ones_like(img), torch.zeros_like(img))
        return img

# 数据变换函数
def get_transform(phase):
    """根据训练或验证阶段生成数据变换"""
    cfg = Config.data_transform[phase]
    transforms_list = []
    
    if phase == "train":
        if "RandomResizedCrop" in cfg:
            transforms_list.append(transforms.RandomResizedCrop(cfg["RandomResizedCrop"]))
        if "RandomHorizontalFlip" in cfg and cfg["RandomHorizontalFlip"]:
            transforms_list.append(transforms.RandomHorizontalFlip())
    else:  # val
        if "Resize" in cfg:
            transforms_list.append(transforms.Resize(cfg["Resize"]))
        if "CenterCrop" in cfg:
            transforms_list.append(transforms.CenterCrop(cfg["CenterCrop"]))
    
    transforms_list.append(Binarize(threshold=cfg["BinarizeThreshold"]))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=cfg["Normalize"]["mean"], 
                                               std=cfg["Normalize"]["std"]))
    
    return transforms.Compose(transforms_list)

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 训练函数
def train(rank, world_size, resume_path=Config.resume_path, max_epochs=Config.max_epochs, 
          patience=Config.patience, min_delta=Config.min_delta):
    # 初始化分布式训练
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 数据加载
    train_dataset = datasets.ImageFolder(
        root=Config.dataset["train_root"],
        transform=get_transform("train")
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        sampler=train_sampler,
        num_workers=min(os.cpu_count() // world_size, 4),  # 每个进程分配合理的工作线程
        pin_memory=True  # 加速数据传输到 GPU
    )

    # 仅在 rank 0 加载验证集（减少冗余）
    if rank == 0:
        val_dataset = datasets.ImageFolder(
            root=Config.dataset["val_root"],
            transform=get_transform("val")
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=min(os.cpu_count(), 4),
            pin_memory=True
        )
        print(f"Number of classes: {len(train_dataset.classes)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    else:
        val_loader = None

    # 模型加载
    model = resnext101_32x8d(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, Config.dataset["num_classes"])
    model = model.to(device)
    model = DDP(model, device_ids=[rank])  # 封装为 DDP 模型

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # 继续训练：加载检查点
    start_epoch = 0
    best_acc = 0.0
    if resume_path and os.path.exists(resume_path) and rank == 0:
        checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])  # DDP 使用 .module
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.3f}")

    # 广播检查点参数到所有进程
    dist.barrier()
    if resume_path and os.path.exists(resume_path):
        broadcast_dict = {
            'start_epoch': start_epoch,
            'best_acc': best_acc
        }
        for key, value in broadcast_dict.items():
            tensor = torch.tensor(value, dtype=torch.float32 if key == 'best_acc' else torch.int64).to(device)
            dist.broadcast(tensor, src=0)
            broadcast_dict[key] = tensor.item()
        start_epoch = int(broadcast_dict['start_epoch'])
        best_acc = broadcast_dict['best_acc']

    # 早停参数
    patience_counter = 0
    best_model_path = Config.save_path

    # 训练循环
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # 确保每个 epoch 打乱数据
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, disable=rank != 0)  # 仅 rank 0 显示进度
        
        for data in train_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{max_epochs}] loss:{loss:.3f}"

        # 同步所有进程的损失
        loss_tensor = torch.tensor(running_loss).to(device)
        dist.reduce(loss_tensor, dst=0)
        if rank == 0:
            avg_train_loss = loss_tensor.item() / (world_size * len(train_loader))
        else:
            avg_train_loss = 0.0

        # 验证（仅 rank 0 执行）
        val_accuracy = 0.0
        if rank == 0:
            model.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    outputs = model(val_images)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels).sum().item()
                    val_bar.desc = f"valid epoch[{epoch+1}/{max_epochs}]"
            val_accuracy = acc / len(val_dataset)

        # 广播验证准确率
        val_acc_tensor = torch.tensor(val_accuracy).to(device)
        dist.broadcast(val_acc_tensor, src=0)
        val_accuracy = val_acc_tensor.item()

        # 仅 rank 0 打印和保存
        if rank == 0:
            print(f"[epoch {epoch+1}] train_loss: {avg_train_loss:.3f} val_accuracy: {val_accuracy:.3f}")
            if val_accuracy > best_acc + min_delta:
                best_acc = val_accuracy
                patience_counter = 0
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),  # DDP 使用 .module
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(checkpoint, best_model_path)
                print(f"New best accuracy: {best_acc:.3f}, model saved to {best_model_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # 广播耐心计数器
        patience_tensor = torch.tensor(patience_counter).to(device)
        dist.broadcast(patience_tensor, src=0)
        patience_counter = int(patience_tensor.item())

        # 早停检查
        if patience_counter >= patience:
            if rank == 0:
                print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        dist.barrier()  # 同步所有进程

    if rank == 0:
        print("Finished Training")
    cleanup()

# 主函数
def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(rank=0, world_size=1)

if __name__ == "__main__":
    # 示例环境变量设置
    # os.environ['RESUME_PATH'] = '/kaggle/input/checkpoint/best_model.pth'  # 若继续训练

    main()
