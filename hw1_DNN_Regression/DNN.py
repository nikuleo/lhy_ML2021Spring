import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt


# 保证模型可复现性
def init_seed(seed=0):
    torch.backends.cudnn.deterministic = True  # 保证训练参数计算一致
    torch.backends.cudnn.benchmark = False  # 训练加速
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_learning_curve(loss_record, title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]  # 设置x_2为range train的范围 步长为 train/dev
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to('cuda'), y.to('cuda')
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


# 将训练集拆成训练集和验证集，默认值是0.25，可以调
def _train_dev_split(X, Y, dev_ratio=0.08):
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]  # 省略对列的操作，仅对行进行分割列不变


# 数据预处理 & 筛选特征, 这里为了先预处理全部数据保证测试集和验证集的归一化一致。之后再分train和dev
def data_prep(path, mode='train'):
    with open(path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])[:, 1:].astype(float)

    feats = list(range(40)) + sorted([75, 57, 42, 60, 78, 43, 61, 79, 40, 58, 76, 41, 59, 77])
    if mode == 'test':
        data = data[:, feats]
    else:
        target = data[:, -1]
        data = data[:, feats]

    data[:, 40:] = (data[:, 40:] - np.mean(data[:, 40:], axis=0)) / np.std(data[:, 40:], axis=0)
    if mode == 'test':
        return data
    else:
        return data, target


class COVID19Dataset(Dataset):
    def __init__(self, x, y=None, mode='train'):
        self.x = torch.FloatTensor(x)
        self.mode = mode
        if y is not None:
            self.y = torch.FloatTensor(y)
        self.dim = self.x.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.mode in ['train', 'dev']:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(p=0.5),  # 传入概率为0.5
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.Dropout(p=0.5),  # 传入概率为0.5
            nn.SiLU(),
            nn.Linear(128, 1)
        )

        self.criterion = nn.MSELoss(reduction='mean')  # reduction置为mean，该批次的总loss 除以mean

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def dev(dv_set, model):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to('cuda'), y.to('cuda')
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)  # 因为pytorch的loss计算默认取一批数据的平均值所以要乘回来
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss


def train(tr_set, dv_set, model, cfg):
    n_epochs = cfg['n_epochs']
    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to('cuda'), y.to('cuda')
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())  # .detach()将tensor从计算图拷贝出来，.item() 取tensor中的数值

        dev_mse = dev(dv_set, model)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, train loss = {:.4f}, dev loss = {:4f})'.format(epoch + 1, mse_loss,
                                                                                               min_mse))
            torch.save(model.state_dict(), 'models/model' + '.pth')
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > cfg['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return mse_loss, loss_record


def test(tt_set, model):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to('cuda')
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    return preds


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == '__main__':
    # 准备数据，设置参数

    init_seed(2333)

    data_path = '../data/hw1/covid.train.csv'
    test_path = '../data/hw1/covid.test.csv'

    all_x, all_y = data_prep(data_path, 'train')
    train_x, train_y, dev_x, dev_y = _train_dev_split(all_x, all_y)
    test_x = data_prep(test_path, 'test')

    train_set = COVID19Dataset(train_x, train_y, 'train')
    dev_set = COVID19Dataset(dev_x, dev_y, 'dev')
    test_set = COVID19Dataset(test_x, None, 'test')

    os.makedirs('models', exist_ok=True)
    config = {
        'fold_num': 1,
        'n_epochs': 8000,
        'batch_size': 64,
        'optimizer': 'SGD',
        'optim_hparas': {
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        },
        'early_stop': 500,
        'save_path': 'models/model.pth'
    }

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    # 训练

    model = NeuralNet(train_loader.dataset.dim).to('cuda')
    model_loss, model_loss_record = train(train_loader, dev_loader, model, config)

    plot_learning_curve(model_loss_record, title='DNN')

    del model
    model = NeuralNet(train_loader.dataset.dim).to('cuda')
    ckpt = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(ckpt)
    plot_pred(dev_loader, model)

    # 测试
    preds = test(test_loader, model)
    save_pred(preds, 'pred.csv')
