import os

class Config:
    # 数据集配置保持不变
    dataset = {
        "root": os.getenv("DATA_ROOT", "/kaggle/input/hust-obc-train-val-8-2"),
        "train_dir": os.getenv("TRAIN_DIR", "train"),
        "val_dir": os.getenv("VAL_DIR", "val"),
        "num_classes": int(os.getenv("NUM_CLASSES", 1781))
    }

    # 修改模型路径为 resnet50
    pretrained_path = os.getenv("PRETRAINED_PATH", "./resnet50-pre.pth")
    save_path = os.getenv("SAVE_PATH", "./resnet50_1781.pth")
    resume = os.getenv("RESUME", None)

    # 训练参数保持不变
    epochs = int(os.getenv("EPOCHS", 10))
    batch_size = int(os.getenv("BATCH_SIZE", 16))
    lr = float(os.getenv("LR", 0.0001))

    # 修改数据预处理为灰度图，分辨率 40-80
    data_transform = {
        "train": {
            "Grayscale": 1,  # 转换为单通道灰度图
            "RandomResizedCrop": int(os.getenv("TRAIN_RESIZED_CROP", 64)),  # 随机裁剪到 64x64
            "RandomHorizontalFlip": os.getenv("TRAIN_HFLIP", "True").lower() == "true",  # 默认开启水平翻转
            "RandomRotation": int(os.getenv("TRAIN_ROTATION", 15)),  # 新增：随机旋转角度，默认 ±15 度
            "ColorJitter": {  # 新增：颜色抖动参数
                "brightness": float(os.getenv("TRAIN_BRIGHTNESS", 0.2)),
                "contrast": float(os.getenv("TRAIN_CONTRAST", 0.2)),
                "saturation": float(os.getenv("TRAIN_SATURATION", 0.2))
            },
            "RandomErasing": {  # 新增：随机擦除参数
                "p": float(os.getenv("TRAIN_ERASING_P", 0.5)),
                "scale": (float(os.getenv("TRAIN_ERASING_SCALE_MIN", 0.02)), 
                          float(os.getenv("TRAIN_ERASING_SCALE_MAX", 0.33)))
            },
            "Normalize": {
                "mean": [float(os.getenv("TRAIN_NORM_MEAN", "0.5"))],  # 灰度图单通道均值
                "std": [float(os.getenv("TRAIN_NORM_STD", "0.5"))]     # 灰度图单通道标准差
            }
        },
        "val": {
            "Grayscale": 1,  # 转换为单通道灰度图
            "Resize": int(os.getenv("VAL_RESIZE", 80)),    # 缩放到 80
            "CenterCrop": int(os.getenv("VAL_CROP", 64)),  # 中心裁剪到 64
            "Normalize": {
                "mean": [float(os.getenv("VAL_NORM_MEAN", "0.5"))],
                "std": [float(os.getenv("VAL_NORM_STD", "0.5"))]
            }
        }
    }
