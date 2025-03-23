import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnext101_32x8d
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    transforms_list = []
    if phase == "train":
        transforms_list.append(transforms.RandomResizedCrop(224))  # 示例大小
        transforms_list.append(transforms.RandomHorizontalFlip())
    else:  # val
        transforms_list.append(transforms.Resize(256))  # 示例大小
        transforms_list.append(transforms.CenterCrop(224))
    
    transforms_list.append(Binarize(threshold=0.5))  # 灰度 + 二值化
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # 单通道归一化
    
    return transforms.Compose(transforms_list)

# 训练函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    train_dataset = datasets.ImageFolder(
        root="path/to/train",  # 替换为实际训练数据路径
        transform=get_transform("train")
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # 示例批量大小
        shuffle=True,
        num_workers=4  # 示例工作线程数
    )

    val_dataset = datasets.ImageFolder(
        root="path/to/val",  # 替换为实际验证数据路径
        transform=get_transform("val")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # 模型加载
    model = resnext101_32x8d(pretrained=False)  # 不使用预训练权重
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道为 1
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # 调整分类层
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 示例学习率

    # 训练循环
    best_acc = 0.0
    for epoch in range(10):  # 示例epoch数
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
            train_bar.desc = f"train epoch[{epoch+1}/10] loss:{loss:.3f}"

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
                val_bar.desc = f"valid epoch[{epoch+1}/10]"

        val_accuracy = acc / len(val_dataset)
        print(f"[epoch {epoch+1}] train_loss: {running_loss/len(train_loader):.3f} val_accuracy: {val_accuracy:.3f}")

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best accuracy: {best_acc:.3f}, model saved.")

    print("Finished Training")

if __name__ == "__main__":
    train()
