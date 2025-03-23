import os

class Config:
    # 数据集配置
    dataset = {
        "root": os.getenv("DATA_ROOT", "/kaggle/input/hust-obc-train-val-8-2"),
        "train_dir": os.getenv("TRAIN_DIR", "train"),
        "val_dir": os.getenv("VAL_DIR", "val"),
        "num_classes": int(os.getenv("NUM_CLASSES", 1781))
    }

    # 模型路径配置
    pretrained_path = os.getenv("PRETRAINED_PATH", "./resnext101_32x8d-pre.pth")
    save_path = os.getenv("SAVE_PATH", "./resnext101_32x8d_1781.pth")
    resume = os.getenv("RESUME", None)

    # 训练参数
    epochs = int(os.getenv("EPOCHS", 10))
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    lr = float(os.getenv("LR", 0.0001))

    # 数据预处理
    data_transform = {
        "train": {
            "RandomResizedCrop": int(os.getenv("TRAIN_RESIZED_CROP", 224)),
            "RandomHorizontalFlip": os.getenv("TRAIN_HFLIP", "True").lower() == "true",
            "Normalize": {
                "mean": [float(x) for x in os.getenv("TRAIN_NORM_MEAN", "0.485,0.456,0.406").split(",")],
                "std": [float(x) for x in os.getenv("TRAIN_NORM_STD", "0.229,0.224,0.225").split(",")]
            }
        },
        "val": {
            "Resize": int(os.getenv("VAL_RESIZE", 256)),
            "CenterCrop": int(os.getenv("VAL_CROP", 224)),
            "Normalize": {
                "mean": [float(x) for x in os.getenv("VAL_NORM_MEAN", "0.485,0.456,0.406").split(",")],
                "std": [float(x) for x in os.getenv("VAL_NORM_STD", "0.229,0.224,0.225").split(",")]
            }
        }
    }
