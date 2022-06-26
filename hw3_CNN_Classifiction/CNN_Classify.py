import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset, DataLoader
from torchvision.datasets import DatasetFolder
from torchsummary import summary
import csv
import matplotlib.pyplot as plt
import time

# 参数设置
config = {
    'epoch': 11,
    'batch_size': 128,
    'device': "cuda",
    'do_semi': True,
    'threshord': 0.9,
    'best_acc': 0.75,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.0001,
        'weight_decay': 1e-5,
    },
    'save_path': 'models/model.ckpt',
    'log_path': 'models/logs',
    'checkpoint_path': 'models/ckpt.pth',
    'remuse': False,
}


def load_data(path, mode):
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

    # data = DatasetFolder(path, loader=lambda x: Image.open(x), extensions="jpg", transform=data_tfm['test'])
    data = DatasetFolder(path, loader=imgs_tfm, extensions="jpg", transform=data_tfm['test'])

    if mode == 'train':
        # argu_data = DatasetFolder(path, loader=lambda x: Image.open(x), extensions="jpg", transform=data_tfm['train'])
        argu_data = DatasetFolder(path, loader=imgs_tfm, extensions="jpg", transform=data_tfm['train'])
        data = ConcatDataset([data, argu_data])

    return data


class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx][0], self.y[idx]


def get_pseudo_labels(dataset, model, threshold=config['threshord']):
    unlabeled_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    device = config['device']
    model.eval()
    softmax = nn.Softmax(dim=-1)

    idx = []
    labels = []

    for i, batch in enumerate(unlabeled_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)

        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * config['batch_size'] + j)
                labels.append(int(torch.argmax(x)))

        model.train()
        print("\nNew data: {:5d}\n".format(len(idx)))
        dataset = PseudoDataset(Subset(dataset, idx), labels)
        return dataset

    return dataset


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        self.cnn_layers = nn.Sequential(

            nn.Conv2d(3, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 11)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


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


def train(model, train_loader, valid_loader, unlabeled_set, train_set, device):
    n_epochs = config['epoch']
    start_epoch = -1
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    tr_loss_record = []
    val_loss_record = []
    tr_acc_record = []
    val_acc_record = []

    # 恢复训练
    if config['remuse']:
        print("恢复训练")
        checkpoit = torch.load(config['checkpoint_path'])
        start_epoch = checkpoit['epoch']
        model.load_state_dict(checkpoit['model'])
        optimizer.load_state_dict(checkpoit['optimizer'])
        best_acc = checkpoit['best_acc']
        tr_loss_record = checkpoit['tr_loss_record']
        tr_acc_record = checkpoit['tr_acc_record']
        val_loss_record = checkpoit['val_loss_record']
        val_acc_record = checkpoit['val_acc_record']

    for epoch in range(start_epoch + 1, n_epochs):

        epoch_start_time = time.time()
        # 半监督
        if config['do_semi'] and best_acc > config['best_acc']:
            pseudo_set = get_pseudo_labels(unlabeled_set, model, config['threshord'])
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,
                                      drop_last=True)
        model.train()
        train_loss = []
        train_acc = []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            # loss = model.cal_loss(logits, labels)
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

        val_loss, val_acc = dev(model, valid_loader, device)
        print(
            f"[ 验证 |epoch： {epoch + 1:03d} / {n_epochs:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}, 耗时： {time.time() - epoch_start_time:2.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])

        tr_loss_record.append(train_loss)
        tr_acc_record.append(train_acc)
        val_loss_record.append(val_loss)
        val_acc_record.append(val_acc)

        if epoch % 5 == 0:
            checkpoit = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'tr_loss_record': tr_loss_record,
                'tr_acc_record': tr_acc_record,
                'val_loss_record': val_loss_record,
                'val_acc_record': val_acc_record
            }
            torch.save(checkpoit, config['checkpoint_path'])

    return tr_loss_record, tr_acc_record, val_loss_record, val_acc_record


def test(model, test_loader, device):
    model.eval()
    preds = []

    for imgs, labels in test_loader:
        with torch.no_grad():
            logits = model(imgs.to(device))
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return preds


def save_preds(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Category'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


def drawAcc(tr_acc, val_acc):
    x = np.arange(len(tr_acc))
    plt.plot(x, tr_acc, color="blue", label="Train")
    plt.plot(x, val_acc, color="red", label="Valid")
    plt.legend(loc="upper right")
    plt.show()


def drawLoss(tr_loss, val_loss):
    x = np.arange(len(tr_loss))
    plt.plot(x, tr_loss, color="blue", label="Train")
    plt.plot(x, val_loss, color="red", label="Valid")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    train_path = "../data/hw3/training/labeled"
    val_path = "../data/hw3/validation"
    unlabeled_path = "../data/hw3/training/unlabeled"
    test_path = "../data/hw3/testing"

    device = config['device']

    train_set = load_data(train_path, 'train')
    valid_set = load_data(val_path, 'valid')
    unlabeled_set = load_data(val_path, 'unlabeled')
    test_set = load_data(test_path, 'test')

    train_loader = DataLoader(train_set, config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, config['batch_size'], shuffle=False, num_workers=0)

    # model = CNNNet().to(device)
    model = torchvision.models.resnet34(pretrained=True).to(device)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 11)
    model.device = device
    for param in model.parameters():
        param.requires_grad = True

    summary(model, input_size=(3, 128, 128), device=device)

    tr_loss_record, tr_acc_record, val_loss_record, val_acc_record = train(model, train_loader, valid_loader,
                                                                           unlabeled_set, train_set, device)
    drawAcc(tr_acc_record, val_acc_record)
    drawLoss(tr_loss_record, val_loss_record)

    preds = test(model, test_loader, device)
    save_preds(preds, "pred.csv")
