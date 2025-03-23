from config import Config

def train():
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
        num_workers=4
    )
    # ... 其他代码保持不变

    # 调用训练
    train(resume_path=Config.resume_path, max_epochs=Config.max_epochs, 
          patience=Config.patience, min_delta=Config.min_delta)

if __name__ == "__main__":
    train()
