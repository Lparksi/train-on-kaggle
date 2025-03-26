import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, datasets
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from config import Config

# 用于记录训练过程中的参数（扩展）
history = {
    'train_loss': [],
    'train_accuracy': [],  # 新增：训练准确率
    'val_loss': [],        # 新增：验证损失
    'val_accuracy': [],
    'learning_rate': []    # 新增：学习率
}

def setup(rank, world_size):
    """初始化分布式环境"""
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')  # 默认使用 127.0.0.1
    master_port = os.getenv('MASTER_PORT', '29500')      # 默认端口
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except RuntimeError as e:
        print(f"分布式初始化失败: {e}")
        raise

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_transform(phase):
    """根据配置生成数据变换"""
    cfg = Config.data_transform[phase]
    transforms_list = []
    
    if "RandomResizedCrop" in cfg:
        transforms_list.append(transforms.RandomResizedCrop(cfg["RandomResizedCrop"]))
    if "RandomHorizontalFlip" in cfg and cfg["RandomHorizontalFlip"]:
        transforms_list.append(transforms.RandomHorizontalFlip())
    if "Resize" in cfg:
        transforms_list.append(transforms.Resize(cfg["Resize"]))
    if "CenterCrop" in cfg:
        transforms_list.append(transforms.CenterCrop(cfg["CenterCrop"]))
    
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=cfg["Normalize"]["mean"], 
                                              std=cfg["Normalize"]["std"]))
    
    return transforms.Compose(transforms_list)

def plot_metrics(history, save_dir='./plots'):
    """绘制训练过程中的参数变化并保存（扩展）"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    
    # 绘制训练和验证准确率
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()
    
    # 绘制学习率
    plt.figure(figsize=(10, 6))
    plt.plot(history['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
    plt.close()

class ResNeXtWithFPN(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXtWithFPN, self).__init__()
        
        # 加载预训练的 ResNeXt101
        base_model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        
        # 定义特征提取层
        self.feature_extractor = create_feature_extractor(
            base_model,
            return_nodes={
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4': 'layer4',
            }
        )
        
        # 定义 FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],  # ResNeXt 各层的输出通道数
            out_channels=256  # FPN 统一输出通道数
        )
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 融合多层特征的全连接层
        self.fc = nn.Linear(256 * 4, num_classes)  # 融合 4 层特征，每层 256 通道
        
    def forward(self, x):
        # 提取多层特征
        features = self.feature_extractor(x)
        
        # 将特征送入 FPN
        fpn_features = self.fpn(features)
        
        # 融合所有层的特征
        pooled_features = []
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            pooled = self.pool(fpn_features[layer_name])
            pooled = pooled.view(pooled.size(0), -1)
            pooled_features.append(pooled)
        
        # 拼接所有特征
        combined_features = torch.cat(pooled_features, dim=1)  # [batch_size, 256*4]
        
        # 分类
        x = self.fc(combined_features)
        return x

def train(rank, world_size):
    """训练函数"""
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"使用 {world_size} 个GPU进行训练。")

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
        print(f"类别数量: {len(train_dataset.class_to_idx)}")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(validate_dataset)}")

        cla_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json.dumps(cla_dict, indent=4))

    # 模型加载
    model = ResNeXtWithFPN(num_classes=Config.dataset["num_classes"])
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scaler = torch.amp.GradScaler('cuda')

    # 检查点加载
    start_epoch = 0
    best_acc = 0.0
    if Config.resume and os.path.exists(Config.resume):
        try:
            checkpoint = torch.load(Config.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            if rank == 0:
                print(f"从第 {start_epoch} 个epoch继续训练，最佳准确率: {best_acc:.3f}")
        except Exception as e:
            if rank == 0:
                print(f"加载检查点失败: {e}，从头开始训练")
    elif Config.resume and rank == 0:
        print(f"检查点文件 {Config.resume} 不存在，从头开始训练")

    # 训练开始计时
    start_time = time.time()
    max_duration = 5 * 3600  # 5小时转换为秒

    # 训练循环
    try:
        for epoch in range(start_epoch, Config.epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            running_loss = 0.0
            train_correct = 0  # 新增：记录训练正确预测数
            train_total = 0    # 新增：记录训练样本总数
            
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
                # 计算训练准确率
                predict_y = torch.max(outputs, dim=1)[1]
                train_correct += torch.eq(predict_y, labels).sum().item()
                train_total += labels.size(0)
                
                train_bar.desc = f"训练 epoch[{epoch+1}/{Config.epochs}] 损失:{loss:.3f}"

            dist.barrier()

            if rank == 0:
                model.eval()
                val_loss = 0.0     # 新增：验证损失
                val_correct = 0    # 新增：验证正确预测数
                val_total = 0      # 新增：验证样本总数
                
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        with torch.amp.autocast('cuda'):
                            outputs = model(val_images)
                            loss = loss_function(outputs, val_labels)
                        val_loss += loss.item()
                        predict_y = torch.max(outputs, dim=1)[1]
                        val_correct += torch.eq(predict_y, val_labels).sum().item()
                        val_total += val_labels.size(0)
                        val_bar.desc = f"验证 epoch[{epoch+1}/{Config.epochs}]"

                # 计算指标
                train_loss = running_loss / len(train_loader)
                train_accuracy = train_correct / train_total
                val_loss = val_loss / len(validate_loader)
                val_accuracy = val_correct / val_total
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'[epoch {epoch+1}] 训练损失: {train_loss:.3f} 训练准确率: {train_accuracy:.3f}')
                print(f'验证损失: {val_loss:.3f} 验证准确率: {val_accuracy:.3f} 学习率: {current_lr:.6f}')

                # 记录训练过程中的参数
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                history['learning_rate'].append(current_lr)

                # 保存最佳模型到 Config.save_path
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'epoch': epoch,
                        'best_acc': best_acc
                    }
                    torch.save(checkpoint, Config.save_path)
                    print(f"新的最佳准确率: {best_acc:.3f}，模型已保存至 {Config.save_path}")

                # 保存当前epoch的模型到 latest.pth
                latest_checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc
                }
                torch.save(latest_checkpoint, 'latest.pth')
                print(f"当前模型已保存至 latest.pth")

                # 绘制图表
                plot_metrics(history)

                # 检查训练时间
                elapsed_time = time.time() - start_time
                if elapsed_time > max_duration:
                    print(f"训练时间超过5小时（已用时 {elapsed_time/3600:.2f} 小时），退出训练。")
                    print(f"当前最佳准确率: {best_acc:.3f}，已保存至 {Config.save_path}")
                    print(f"最后一个epoch模型已保存至 latest.pth")
                    plot_metrics(history)
                    return

        if rank == 0:
            print('训练完成')
            print(f"最后一个epoch模型已保存至 latest.pth")
            plot_metrics(history)
    finally:
        cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        try:
            mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
        except Exception as e:
            print(f"分布式训练失败: {e}")
            cleanup()
    else:
        train(0, 1)

if __name__ == '__main__':
    main()
