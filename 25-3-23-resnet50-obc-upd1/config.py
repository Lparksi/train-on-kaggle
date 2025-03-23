import os

class Config:
    # 数据集配置
    dataset = {
        "root": os.getenv("DATA_ROOT", "/kaggle/input/hust-obc-train-val-8-2"),
        "train_dir": os.getenv("TRAIN_DIR", "train"),
        "val_dir": os.getenv("VAL_DIR", "val"),
        "num_classes": int(os.getenv("NUM_CLASSES", 1781))
    }

    # 模型保存路径和恢复训练选项
    save_path = os.getenv("SAVE_PATH", "./resnet50_1781.pth")
    resume = os.getenv("RESUME", None)

    # 训练参数
    epochs = int(os.getenv("EPOCHS", 10))
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    lr = float(os.getenv("LR", 0.0001))

    # 早停参数
    early_stopping_patience = int(os.getenv("EARLY_STOPPING_PATIENCE", 10))  # 默认耐心值为 10

    # 数据预处理（灰度图，分辨率 40-80）
    data_transform = {
        "train": {
            "Grayscale": 1,
            "RandomResizedCrop": int(os.getenv("TRAIN_RESIZED_CROP", 64)),
            "RandomHorizontalFlip": os.getenv("TRAIN_HFLIP", "True").lower() == "true",
            "RandomRotation": int(os.getenv("TRAIN_ROTATION", 15)),
            "ColorJitter": {
                "brightness": float(os.getenv("TRAIN_BRIGHTNESS", 0.2)),
                "contrast": float(os.getenv("TRAIN_CONTRAST", 0.2)),
                "saturation": float(os.getenv("TRAIN_SATURATION", 0.2))
            },
            "RandomErasing": {
                "p": float(os.getenv("TRAIN_ERASING_P", 0.5)),
                "scale": (float(os.getenv("TRAIN_ERASING_SCALE_MIN", 0.02)), 
                          float(os.getenv("TRAIN_ERASING_SCALE_MAX", 0.33)))
            },
            "Normalize": {
                "mean": [float(os.getenv("TRAIN_NORM_MEAN", "0.5"))],
                "std": [float(os.getenv("TRAIN_NORM_STD", "0.5"))]
            }
        },
        "val": {
            "Grayscale": 1,
            "Resize": int(os.getenv("VAL_RESIZE", 80)),
            "CenterCrop": int(os.getenv("VAL_CROP", 64)),
            "Normalize": {
                "mean": [float(os.getenv("VAL_NORM_MEAN", "0.5"))],
                "std": [float(os.getenv("VAL_NORM_STD", "0.5"))]
            }
        }
    }
