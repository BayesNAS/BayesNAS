import torch
from time import time
import OneShot
import Algorithm
import lib
import os
import pickle as pkl
import argparse

parser = argparse.ArgumentParser("search")
parser.add_argument("--lambda_child", type=float, default=None, help="desired lambda value for the child switches")
parser.add_argument("--lambda_origin", type=float, default=None, help="desired lambda value for the origin switch")
parser.add_argument("--train_batch_size", type=int, default=18, help="training batch size")
parser.add_argument("--val_batch_size", type=int, default=18, help="validation batch size")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight_decay")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
args = parser.parse_args()

if __name__ == '__main__':  # multi-processing protection
    folder_head = './'
    folder = folder_head + 'lambda_' + str(args.lambda_origin) + '_' + str(args.lambda_child) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    epochs = 1
    lr = args.lr
    lambda_child = args.lambda_child
    lambda_origin = args.lambda_origin
    percent_used = 1

    train_loader_normal, train_loader_switch, val_loader = lib.build_loader_cifar_alt(train_batch_size,
                                                                                      val_batch_size, num_workers=4)
    model = OneShot.NetWork(num_classes=10, save_device=OneShot.DEVICE)
    algorithm = Algorithm.Algorithm(model=model, device=OneShot.DEVICE, lambda_child=lambda_child,
                                    lambda_origin=lambda_origin)

    normal_params, switch_params = algorithm.model.return_params()
    optimizer = torch.optim.SGD(params=normal_params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0
    acc_history = []
    loss_history = []

    # normal training&evaluation
    for epoch in range(epochs):
        # training
        start = time()
        y = lib.train(epoch=epoch, train_loader=train_loader_normal, percent_used=percent_used, algorithm=algorithm,
                      optimizer=optimizer)
        end = time()
        print('Training elapse: {0}s'.format(end - start))
        # validation
        start = time()
        acc, val_loss = lib.validate(val_loader=val_loader, algorithm=algorithm)
        acc_history.append(acc)
        loss_history.append(val_loss)
        end = time()
        print('Validation elapse: {0}s'.format(end - start))
        # save the best model
        if acc > best_acc:
            print('Best model saved.')
            best_acc = acc
            state = {'acc': acc, 'state_dict': algorithm.model.state_dict()}
            torch.save(state, folder + 'best_saved_model.pt')

        with open(folder + 'acc_history.pt', 'wb') as f:
            pkl.dump(acc_history, f)
        with open(folder + 'loss_history.pt', 'wb') as f:
            pkl.dump(loss_history, f)
    # switch training, algorithm update and evaluation
    optimizer_no_weight_decay = torch.optim.SGD(params=switch_params, lr=lr, momentum=args.momentum, weight_decay=0)
    lib.train_switch(train_loader_switch, algorithm, optimizer, optimizer_no_weight_decay, folder)
    # validation
    start = time()
    acc, val_loss = lib.validate(val_loader=val_loader,
                                 algorithm=algorithm)  # we change here to use validation to calculate hessian
    acc_history.append(acc)
    loss_history.append(val_loss)
    end = time()
    print('Validation elapse: {0}s'.format(end - start))

    # save the best model
    if acc > best_acc:
        print('Best model saved.')
        best_acc = acc
        state = {'acc': acc, 'state_dict': algorithm.model.state_dict()}
        torch.save(state, folder + 'best_saved_model.pt')

    with open(folder + 'acc_history.pt', 'wb') as f:
        pkl.dump(acc_history, f)
    with open(folder + 'loss_history.pt', 'wb') as f:
        pkl.dump(loss_history, f)
