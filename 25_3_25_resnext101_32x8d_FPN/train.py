def train(rank, world_size):
    """训练函数"""
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        print(f"使用 {world_size} 个GPU进行训练。")

    # 数据加载部分保持不变
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
                train_bar.desc = f"训练 epoch[{epoch+1}/{Config.epochs}] 损失:{loss:.3f}"

            dist.barrier()

            if rank == 0:
                model.eval()
                acc = 0.0
                with torch.no_grad():
                    val_bar = tqdm(validate_loader, file=sys.stdout)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        val_images, val_labels = val_images.to(device), val_labels.to(device)
                        with torch.amp.autocast('cuda'):
                            outputs = model(val_images)
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels).sum().item()
                        val_bar.desc = f"验证 epoch[{epoch+1}/{Config.epochs}]"

                val_accurate = acc / len(validate_dataset)
                train_loss = running_loss / len(train_loader)
                print(f'[epoch {epoch+1}] 训练损失: {train_loss:.3f} 验证准确率: {val_accurate:.3f}')

                # 记录训练过程中的参数
                history['train_loss'].append(train_loss)
                history['val_accuracy'].append(val_accurate)

                # 保存最佳模型到 Config.save_path
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'epoch': epoch,
                        'best_acc': best_acc
                    }
                    torch.save(checkpoint, Config.save_path)
                    print(f"新的最佳准确率: {best_acc:.3f}，模型已保存至 {Config.save_path}")

                # 新增：保存当前epoch的模型到 latest.pth
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
                    plot_metrics(history)  # 保存最终图表
                    return  # 退出训练

        if rank == 0:
            print('训练完成')
            print(f"最后一个epoch模型已保存至 latest.pth")
            plot_metrics(history)
    finally:
        cleanup()
