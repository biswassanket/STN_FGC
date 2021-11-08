#!/usr/bin/env python
# coding: utf-8

# import argparse 
# import pytorch_lightning as pl
# import torch
# import torchvision
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from six.moves import urllib
# from torchvision import datasets, transforms

# from STN import SimpleSTN, CoordConvSTN
# from spinal import SpinalNet, SpinalNetCoordConv
# from vit import ViT
# from utils import plot_metric, convert_image_np, compare_stns, plot_wrong_preds
# pl.utilities.seed.seed_everything(1) 

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
import argparse
from torch.utils.tensorboard import SummaryWriter
import time 
import os 
from torch.optim import lr_scheduler
import random

from utils import plot_metric, convert_image_np, compare_stns, plot_wrong_preds
from STN import SimpleSTN, CoordSTN
from vit_pytorch import VisionTransformer


random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

batch_size=64
num_workers=2
# Training dataset

trainset = torchvision.datasets.MNIST(root='.',
                                      train=True,
                                      download=True,
                                      #transform=tr.ToTensor()
                                      )

testset = torchvision.datasets.MNIST(root='.',
                                     train=False,
                                     download=True,
                                     #transform=tr.ToTensor()
                                    )                                      
train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(root='.', train=True, download=True,
                transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])), batch_size=64, shuffle=True, num_workers=2)

# Validation dataset
test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=False, num_workers=2)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"





def train(model, criterion, optimizer, schedler, epochs, train_log='train_log', test_log='test_log', saved_model='model'):
    best_acc = 0.0
    begin = time.time()
    for epoch in range(epochs):
        logs = open(train_log, 'a')
        model.train()
        running_corrects = 0
        running_loss = 0.0
        schedler.step()
        for i, (images, labels) in enumerate(train_loader):
            start = time.time()
            images = images.cuda()
            labels = labels.cuda()
            # import pdb; pdb.set_trace()
            # outputs, hidden = model(images)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if i % 10 == 0:
                print('Epoch: {}/{}, Iter: {}/{:.0f}, Loss: {:.4f}, Time: {:.4f}s/batch'
                     .format(epoch, epochs, i, trainset.__len__()/batch_size+1, loss.item(), time.time()-start))
        epoch_loss = running_loss / trainset.__len__()
        epoch_acc = running_corrects.double() / trainset.__len__()

        log = 'Epoch: {}/{}, Loss: {:.4f} Acc: {}/{}, {:.4f}, Time: {:.0f}s'.format(epoch, 
                                                              epochs,
                                                              epoch_loss, 
                                                              running_corrects, trainset.__len__(), epoch_acc, 
                                                              time.time()-begin)
        print(log)
        logs.write(log+'\n')
        test_acc = validate(model, test_log=test_log)
        torch.save(model.state_dict(), '{}_latest.pkl'.format(saved_model))
        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '{}_best.pkl'.format(saved_model))


def validate(model, crop=0, test_log=''):
    begin = time.time()
    if test_log != '':
        logs = open(test_log, 'a')
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()

            # outputs, hidden = model(images)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / testset.__len__()
        epoch_acc = running_corrects.double() / testset.__len__()
        log = 'Test Loss: {:.4f} Acc: {}/{}, {:.4f}, Time: {:.0f}s'.format(epoch_loss, 
                                                       running_corrects, testset.__len__(), epoch_acc, 
                                                       time.time()-begin)
        print(log)
        if test_log != '':
            logs.write(log+'\n')
        return epoch_acc

def main(args):
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # # Training dataset
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root='.', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])), batch_size=args.bs, shuffle=True, num_workers=args.workers)

    # # Validation dataset
    # validation_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(root='.', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])), batch_size=args.bs, shuffle=False, num_workers=args.workers)
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Set the model type
    if args.model == "stn":
        model = SimpleSTN()
    elif args.model == "stncoordconv":
        model = CoordSTN(coordconv_localization=args.localization, with_r=args.rchannel)
    elif args.model == "vit":
        model = VisionTransformer(embed_dim=64,
        hidden_dim=128,
        num_channels=1,
        num_heads=8,
        num_layers=6,
        num_classes=64,
        patch_size=7,
        num_patches=64,
        dropout=0.2)
    elif args.model == "spinal":
        model = SpinalNet(first_HL=8, lr=args.lr)
    elif args.model == "spinalstn":
        model = SpinalNetCoordConv(first_HL=8, lr=args.lr)

    gpus = 1 if torch.cuda.is_available() and args.device == 'gpu' else 0

    

    # logger = TensorBoardLogger("logs", name=args.model)
    # early_stop_callback = EarlyStopping(monitor="validation_loss", 
    #                                     min_delta=args.mindelta,
    #                                     patience=args.patience, 
    #                                     verbose=True, 
    #                                     mode="min")
    
    # trainer = pl.Trainer(logger=logger,
    #                      callbacks=[early_stop_callback],
    #                      gpus=gpus,
    #                      min_epochs=1,
    #                      max_epochs=args.maxepochs,
    #                      deterministic=True)
    # trainer.fit(model,
    #             train_dataloaders=train_loader, 
    #             val_dataloaders=validation_loader)

    # print("Validation accuracy = %.04f and loss = %.04f at epoch %d" %
    #   (trainer.logged_metrics['validation_acc'], 
    #    trainer.logged_metrics['validation_loss'], 
    #    trainer.logged_metrics['epoch']))

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    if not os.path.exists('logs'):
        os.mkdir('logs')
    # train_log = 'logs/train_resnet_multi975_stn101'
    train_log = 'logs/train_mnist_resnet_stn50'
    # test_log = 'logs/test_resnet_multi975_stn101'
    test_log = 'logs/test_mnist_resnet_stn50'
    # saved_model = 'resnet_multi975_stn101'
    saved_model = 'resnet_mnist_stn50'
    train(model, criterion, optimizer, exp_lr_scheduler, epochs=50, train_log=train_log, test_log=test_log, saved_model=saved_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'MNIST-benchmarks-STN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu', 
                        help='Device on which to run the experiments.')
    parser.add_argument('--workers', type=int, default=2, 
                        help='Number of workers for dataloaders.')
    parser.add_argument('--bs', type=int, default=64, 
                        help='Batch size.')
    parser.add_argument('--maxepochs', type=int, metavar='MAX_EPOCHS', default=20, 
                        help='Maximum number of epochs to run the experiment for.')

    parser.add_argument('--patience', type=int, metavar='PATIENCE', default=5, 
                        help='Number of epochs with no improvement before triggering early stopping.')
    parser.add_argument('--mindelta', type=float, metavar='MIN_DELTA', default=0.005, 
                        help='Required improvement in the validation loss for early stopping.')
    parser.add_argument('--model', type=str, choices=['stn', 'stncoordconv', 'vit', 'spinal', 'spinalstn'], default='stn', help='Type of model to train.')
    parser.add_argument('--localization', default=False, action='store_true', 
                        help='Whether to use CoordConv in the localization network.')
    parser.add_argument('--rchannel', default=False, action='store_true',
                        help='Whether to use r-th channel in the network')
    parser.add_argument('--lr', type=float, metavar='LR', default=0.01, 
                        help='Learning rate for SGD.')
    parser.add_argument('--logs', type=str, metavar='LOGPATH', default='logs/', 
                        help='Directory to store tensorboard logs.')

    args = parser.parse_args()
    
    main(args)



