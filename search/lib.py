import torch
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn.functional as F
import operator
import dataloader
from time import time
import pickle as pkl


def build_loader_cifar(train_batch_size, val_batch_size, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root='./CIFAR10_Dataset', train=True,
                                 download=True, transform=train_transform)
    train_loader = data.DataLoader(train_set, batch_size=train_batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)

    val_set = datasets.CIFAR10(root='./CIFAR10_Dataset', train=False,
                               download=True, transform=val_transform)
    val_loader = data.DataLoader(val_set, batch_size=val_batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def build_loader_cifar_alt(train_batch_size, val_batch_size, num_workers=0):
    train_loader_normal, train_loader_switch, val_loader = dataloader.get_loader(train_batch_size, val_batch_size,
                                                                                 num_workers)

    return train_loader_normal, train_loader_switch, val_loader


def train(epoch, train_loader, percent_used, algorithm, optimizer):
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(algorithm.device), y.to(algorithm.device)
        pred = algorithm.model(x)
        loss = F.nll_loss(pred, y)
        loss.backward()
        optimizer.step()
        if batch % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(x), len(train_loader.dataset),
                       100. * batch / len(train_loader), loss.item()))

        if batch / len(train_loader) > percent_used:
            break

    return y


def train_switch(train_loader, algorithm, optimizer, optimizer_no_weight_decay, folder):
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        optimizer_no_weight_decay.zero_grad()
        x, y = x.to(algorithm.device), y.to(algorithm.device)
        pred = algorithm.model(x)
        loss, loss_prediction = algorithm.loss_cal(pred, y)
        loss.backward()
        optimizer.step()
        optimizer_no_weight_decay.step()
        if batch % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Pred Loss: {:.6f}'.format(
                '---', batch * len(x), len(train_loader.dataset),
                       100. * batch / len(train_loader), loss.item(), loss_prediction.item()))
    # algorithm update
    start = time()
    algorithm.update(y)
    end = time()
    print('Algorithm update elapse: {0}s'.format(end - start))

    # save algorithm information
    switch_norm, gamma_norm, omega_norm, hessian_norm, cov_norm, \
    switch_reduct, gamma_reduct, omega_reduct, hessian_reduct, cov_reduct = algorithm.extract_info()
    with open(folder + 'switch_norm.pkl', 'wb') as f:
        pkl.dump(switch_norm, f)
    with open(folder + 'gamma_norm.pkl', 'wb') as f:
        pkl.dump(gamma_norm, f)
    with open(folder + 'omega_norm.pkl', 'wb') as f:
        pkl.dump(omega_norm, f)
    with open(folder + 'hessian_norm.pkl', 'wb') as f:
        pkl.dump(hessian_norm, f)
    with open(folder + 'cov_norm.pkl', 'wb') as f:
        pkl.dump(cov_norm, f)
    with open(folder + 'switch_reduct.pkl', 'wb') as f:
        pkl.dump(switch_reduct, f)
    with open(folder + 'gamma_reduct.pkl', 'wb') as f:
        pkl.dump(gamma_reduct, f)
    with open(folder + 'omega_reduct.pkl', 'wb') as f:
        pkl.dump(omega_reduct, f)
    with open(folder + 'hessian_reduct.pkl', 'wb') as f:
        pkl.dump(hessian_reduct, f)
    with open(folder + 'cov_reduct.pkl', 'wb') as f:
        pkl.dump(cov_reduct, f)


def validate(val_loader, algorithm):
    with torch.no_grad():
        val_loss = 0
        correct = 0
        for _, (x, y) in enumerate(val_loader):
            x, y = x.to(algorithm.device), y.to(algorithm.device)
            val_pred = algorithm.model(x)
            val_loss = F.nll_loss(val_pred, y).item()
            sparse_pred = val_pred.max(1, keepdim=True)[1]
            correct += sparse_pred.eq(y.view_as(sparse_pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

        return 100. * correct / len(val_loader.dataset), val_loss
