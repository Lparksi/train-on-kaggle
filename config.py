# config.py
class Config:
    # 数据路径
    data_root = "/kaggle/input/hust-obc-train-val-8-2"
    pretrained_path = "./resnext101_32x8d-pre.pth"
    save_path = "./resnext101_32x8d_1781.pth"
    resume = None  # 检查点路径，若继续训练则指定，例如 "./resnext101_32x8d_1781.pth"

    # 训练参数
    epochs = 10
    batch_size = 16
    lr = 0.0001
    num_classes = 1781

    # 数据预处理
    data_transform = {
        "train": {
            "RandomResizedCrop": 224,
            "RandomHorizontalFlip": True,
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "val": {
            "Resize": 256,
            "CenterCrop": 224,
            "Normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
