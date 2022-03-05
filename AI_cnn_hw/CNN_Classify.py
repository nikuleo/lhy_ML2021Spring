import time
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transform
from torchvision.datasets import DatasetFolder
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt

config = {
    'epoch': 30,
    'batch_size': 128,
    'device': "cuda",
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.0001,
        'weight_decay': 1e-5,
    },
    'save_path': 'models/model.pth'
}


# 数据增强
def loader_data(path, mode):
    data_tfm = {
        'train': transform.Compose([
            transform.CenterCrop(224),
            transform.RandomHorizontalFlip(0.5),
            transform.ColorJitter(brightness=0.5),
            transform.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transform.ToTensor()
        ]),
        'test': transform.Compose([
            transform.CenterCrop(224),
            transform.ToTensor()
        ])
    }

    def imgs_tfm(path):
        x = Image.open(path)
        x = x.resize((256, 256))
        return x

    data = DatasetFolder(path, loader=imgs_tfm, extensions="jpg", transform=data_tfm['test'])
    if mode == 'train':
        argu_data = DatasetFolder(path, loader=imgs_tfm, extensions='jpg', transform=data_tfm['train'])
        data = ConcatDataset([data, argu_data])
    return data


def train(model, train_loader, test_loader, device):
    n_epochs = config['epoch']
    epoch = 0
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    tr_loss_record = []
    tr_acc_record = []
    val_loss_record = []
    val_acc_record = []

    while epoch < n_epochs:
        epoch_start_time = time.time()
        model.train()
        train_loss = []
        train_acc = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)

        print(
            f"[ 训练 |epoch： {epoch + 1:03d} / {n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, 耗时： {time.time() - epoch_start_time:2.2f}s")
        val_loss, val_acc = dev(model, test_loader, device)
        print(
            f"[ 验证 |epoch： {epoch + 1:03d} / {n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}, 耗时： {time.time() - epoch_start_time:2.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])

        val_loss_record.append(val_loss)
        val_acc_record.append(val_acc)
        tr_loss_record.append(train_loss)
        tr_acc_record.append(train_acc)
        epoch += 1
    return tr_loss_record, tr_acc_record, val_loss_record, val_acc_record


def dev(model, valid_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    dev_loss = []
    dev_acc = []

    for imgs, labels in valid_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(imgs)

        # loss = model.cal_loss(logits, labels)
        loss = criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        dev_loss.append(loss.item())
        dev_acc.append(acc.item())
    dev_loss = sum(dev_loss) / len(dev_loss)
    dev_acc = sum(dev_acc) / len(dev_acc)
    return dev_loss, dev_acc

# 测试入口。传入模型，测试集数据，和训练设备（cuda）
def test(model, test_loader, device):
    model.eval()
    preds = []

    for imgs, labels in test_loader:
        with torch.no_grad():
            logits = model(imgs.to(device))
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return preds

def drawAcc(tr_acc, val_acc):
    x = np.arange(len(tr_acc))
    plt.plot(x, tr_acc, color="blue", label="Train")
    plt.plot(x, val_acc, color="red", label="Test")
    plt.legend(loc="upper right")
    plt.show()


def drawLoss(tr_loss, val_loss):
    x = np.arange(len(tr_loss))
    plt.plot(x, tr_loss, color="blue", label="Train")
    plt.plot(x, val_loss, color="red", label="Test")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    train_path = "../data/tumor/Training"
    test_path = "../data/tumor/Testing"

    device = config['device']
    train_set = loader_data(train_path, 'train')
    test_set = loader_data(test_path, 'test')

    train_loader = DataLoader(train_set, config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, config['batch_size'], shuffle=False, num_workers=0)

    model = torchvision.models.resnet34(pretrained=True).to(device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4).to(device)
    model.device = device
    for param in model.parameters():
        param.requires_grad = True

    tr_loss_record, tr_acc_record, val_loss_record, val_acc_record = train(model, train_loader, test_loader, device)

    drawAcc(tr_acc_record, val_acc_record)
    drawLoss(tr_loss_record, val_loss_record)

    # 使用该模型测试，请在前面读入测试数据
    # 模型保存在models目录下
    # preds = test(model, 测试数据, device)
