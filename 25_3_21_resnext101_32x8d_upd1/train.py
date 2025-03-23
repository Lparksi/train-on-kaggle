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
import matplotlib.pyplot as plt
import numpy as np

# ... (Binarize类和get_transform函数保持不变) ...

# 用于存储训练数据的类
class TrainingStats:
    def __init__(self):
        self.train_losses = []
        self.val_accuracies = []
        self.best_acc = 0.0

    def update(self, train_loss, val_accuracy):
        self.train_losses.append(train_loss)
        self.val_accuracies.append(val_accuracy)
        self.best_acc = max(self.best_acc, val_accuracy)

    def plot_and_save(self, save_dir="training_plots"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制训练损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'train_loss.png'))
        plt.close()

        # 绘制验证准确率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.plot([self.best_acc] * len(self.val_accuracies), 
                label='Best Accuracy', linestyle='--')
        plt.title('Validation Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'val_accuracy.png'))
        plt.close()

# 修改训练函数
def train(rank, world_size, resume_path=Config.resume_path, max_epochs=Config.max_epochs, 
          patience=Config.patience, min_delta=Config.min_delta):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    stats = TrainingStats() if rank == 0 else None
    
    if rank == 0:
        print(f"使用 {world_size} 个GPU进行训练")

    # 数据加载部分保持不变
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
        num_workers=min(os.cpu_count() // world_size, 4),
        pin_memory=True
    )

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
        print(f"类别数: {len(train_dataset.classes)}")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(val_dataset)}")

    # 使用不带预训练权重的模型
    model = resnext101_32x8d(weights=None)  # 不加载预训练权重
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, Config.dataset["num_classes"])
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # 检查点加载逻辑保持不变
    start_epoch = 0
    best_acc = 0.0
    if resume_path and os.path.exists(resume_path) and rank == 0:
        checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        if rank == 0:
            stats.best_acc = best_acc
        print(f"从epoch {start_epoch}继续训练，最佳准确率: {best_acc:.3f}")

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

    patience_counter = 0
    best_model_path = Config.save_path

    # 训练循环
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, disable=rank != 0)
        
        for data in train_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"训练 epoch[{epoch+1}/{max_epochs}] 损失:{loss:.3f}"

        loss_tensor = torch.tensor(running_loss).to(device)
        dist.reduce(loss_tensor, dst=0)
        if rank == 0:
            avg_train_loss = loss_tensor.item() / (world_size * len(train_loader))
        else:
            avg_train_loss = 0.0

        # 验证
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
                    val_bar.desc = f"验证 epoch[{epoch+1}/{max_epochs}]"
            val_accuracy = acc / len(val_dataset)
            # 更新统计数据
            stats.update(avg_train_loss, val_accuracy)

        val_acc_tensor = torch.tensor(val_accuracy).to(device)
        dist.broadcast(val_acc_tensor, src=0)
        val_accuracy = val_acc_tensor.item()

        if rank == 0:
            print(f"[epoch {epoch+1}] 训练损失: {avg_train_loss:.3f} 验证准确率: {val_accuracy:.3f}")
            if val_accuracy > best_acc + min_delta:
                best_acc = val_accuracy
                patience_counter = 0
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(checkpoint, best_model_path)
                print(f"新的最佳准确率: {best_acc:.3f}, 模型已保存至 {best_model_path}")
            else:
                patience_counter += 1
                print(f"无改进。耐心计数: {patience_counter}/{patience}")

        patience_tensor = torch.tensor(patience_counter).to(device)
        dist.broadcast(patience_tensor, src=0)
        patience_counter = int(patience_tensor.item())

        if patience_counter >= patience:
            if rank == 0:
                print(f"在 {epoch+1} 个epoch后触发早停")
            break

        dist.barrier()

    if rank == 0:
        print("训练完成")
        stats.plot_and_save()  # 保存训练过程中的图像
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(rank=0, world_size=1)

if __name__ == "__main__":
    main()
