import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnext101_32x8d
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config

# 自定义二值化变换类
class Binarize(object):
    """将灰度图像二值化为黑白图像"""
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, img):
        img = transforms.functional.to_grayscale(img, num_output_channels=1)  # 转换为灰度图像
        img = torch.where(img > self.threshold, torch.ones_like(img), torch.zeros_like(img))  # 二值化
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
    
    transforms_list.append(Binarize(threshold=cfg["BinarizeThreshold"]))  # 灰度 + 二值化
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=cfg["Normalize"]["mean"], 
                                               std=cfg["Normalize"]["std"]))
    
    return transforms.Compose(transforms_list)

# 训练函数
def train(resume_path=Config.resume_path, max_epochs=Config.max_epochs, 
          patience=Config.patience, min_delta=Config.min_delta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    train_dataset = datasets.ImageFolder(
        root=Config.dataset["train_root"],
        transform=get_transform("train")
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=min(os.cpu_count(), 4)  # 限制工作线程数，避免 Kaggle 超载
    )

    val_dataset = datasets.ImageFolder(
        root=Config.dataset["val_root"],
        transform=get_transform("val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count(), 4)
    )

    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 模型加载
    model = resnext101_32x8d(pretrained=False)  # 不使用预训练权重
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道为 1
    model.fc = nn.Linear(model.fc.in_features, Config.dataset["num_classes"])  # 调整分类层
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # 继续训练：加载检查点
    start_epoch = 0
    best_acc = 0.0
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.3f}")

    # 早停参数
    patience_counter = 0
    best_model_path = Config.save_path

    # 训练循环
    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
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

        # 验证
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
        train_loss = running_loss / len(train_loader)
        print(f"[epoch {epoch+1}] train_loss: {train_loss:.3f} val_accuracy: {val_accuracy:.3f}")

        # 保存最佳模型并检查早停
        if val_accuracy > best_acc + min_delta:
            best_acc = val_accuracy
            patience_counter = 0  # 重置耐心计数器
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }
            torch.save(checkpoint, best_model_path)
            print(f"New best accuracy: {best_acc:.3f}, model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # 早停检查
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("Finished Training")

if __name__ == "__main__":
    # 示例环境变量设置（可选，在 Kaggle Notebook 中运行时可手动设置）
    os.environ['TRAIN_DATA_ROOT'] = '/kaggle/input/train'
    os.environ['VAL_DATA_ROOT'] = '/kaggle/input/val'
    os.environ['BATCH_SIZE'] = '32'
    os.environ['LR'] = '0.001'
    os.environ['SAVE_PATH'] = '/kaggle/working/best_model.pth'
    # os.environ['RESUME_PATH'] = '/kaggle/input/checkpoint/best_model.pth'  # 若继续训练

    # 运行训练
    train()
