import torch
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import operator
import dataloader


class AvgrageMeter:
    def __init__(self):
        self.avg = None
        self.sum = None
        self.cnt = None
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def build_loader_imagenet(train_DataPath, val_DataPath, train_batch_size, val_batch_size, num_workers=0):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(root=train_DataPath, transform=train_transform)
    train_loader = data.DataLoader(train_set, batch_size=train_batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)

    val_set = datasets.ImageFolder(root=val_DataPath, transform=val_transform)
    val_loader = data.DataLoader(val_set, batch_size=val_batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


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
    train_loader, val_loader = dataloader.get_loader(train_batch_size, val_batch_size, num_workers)

    return train_loader, val_loader


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_imagenet(train_loader, criterion, model, device, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        prec1, prec5 = accuracy(logits, y, topk=(1, 5))
        n = x.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        if batch % 100 == 0:
            print("Train: %03d ... Loss: %e ... Top1 Acc: %f ... Top5 Acc: %f" % (batch, objs.avg, top1.avg, top5.avg))

    return top1.avg, objs.avg


def validate_imagenet(val_loader, criterion, model, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    with torch.no_grad():
        for batch, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            prec1, prec5 = accuracy(logits, y, topk=(1, 5))

            n = x.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if batch % 100 == 0:
                print("Valid: %03d ... Loss: %e ... Top1 Acc: %f ... Top5 Acc: %f" % (
                batch, objs.avg, top1.avg, top5.avg))

        return top1.avg, top5.avg, objs.avg


def train_cifar(epoch, train_loader, percent_used, model, device, optimizer):
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x)
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


def validate_cifar(val_loader, model, device):
    with torch.no_grad():
        val_loss = 0
        correct = 0
        for _, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            val_pred = model(x)
            val_loss += F.nll_loss(val_pred, y).item()
            sparse_pred = val_pred.max(1, keepdim=True)[1]
            correct += sparse_pred.eq(y.view_as(sparse_pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

        return 100. * correct / len(val_loader.dataset), val_loss
