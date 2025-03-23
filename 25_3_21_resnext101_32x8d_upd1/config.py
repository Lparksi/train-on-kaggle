import os

class Config:
    # 数据集配置
    dataset = {
        "train_root": os.environ.get("TRAIN_DATA_ROOT", "/kaggle/input/train"),  # 训练数据路径
        "val_root": os.environ.get("VAL_DATA_ROOT", "/kaggle/input/val"),       # 验证数据路径
        "num_classes": int(os.environ.get("NUM_CLASSES", 1781))                 # 分类数量
    }

    # 模型路径配置
    save_path = os.environ.get("SAVE_PATH", "./best_model.pth")  # 模型保存路径
    resume_path = os.environ.get("RESUME_PATH", None)            # 检查点路径，用于继续训练

    # 训练参数
    batch_size = int(os.environ.get("BATCH_SIZE", 32))           # 批次大小
    lr = float(os.environ.get("LR", 0.001))                      # 学习率
    max_epochs = int(os.environ.get("MAX_EPOCHS", 60))           # 最大 epoch 数

    # 早停参数
    patience = int(os.environ.get("PATIENCE", 10))               # 耐心值
    min_delta = float(os.environ.get("MIN_DELTA", 0.001))        # 最小改进阈值

    # 数据预处理
    data_transform = {
        "train": {
            "RandomResizedCrop": int(os.environ.get("TRAIN_RESIZED_CROP", 224)),
            "RandomHorizontalFlip": os.environ.get("TRAIN_HFLIP", "True").lower() == "true",
            "BinarizeThreshold": float(os.environ.get("BINARIZE_THRESHOLD", 0.5)),
            "Normalize": {
                "mean": [float(x) for x in os.environ.get("TRAIN_NORM_MEAN", "0.5").split(",")],
                "std": [float(x) for x in os.environ.get("TRAIN_NORM_STD", "0.5").split(",")]
            }
        },
        "val": {
            "Resize": int(os.environ.get("VAL_RESIZE", 256)),
            "CenterCrop": int(os.environ.get("VAL_CROP", 224)),
            "BinarizeThreshold": float(os.environ.get("BINARIZE_THRESHOLD", 0.5)),
            "Normalize": {
                "mean": [float(x) for x in os.environ.get("VAL_NORM_MEAN", "0.5").split(",")],
                "std": [float(x) for x in os.environ.get("VAL_NORM_STD", "0.5").split(",")]
            }
        }
    }
