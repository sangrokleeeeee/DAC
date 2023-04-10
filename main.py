import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision.transforms as T

from others import swin, swin_sc, swin_p, swin_k
from mobilenet import mobilenet_sc, mobilenet, mobilenet_p, mobilenet_n
from model import Classifier, DAC50, DAC50_CC3, DAC50_PC3, DAC50_CC2, DAC50_PC2, DAC50_CC1, DAC50_PC1
from data import ForeverDataIterator, get_dataset
from utils import accuracy, AverageMeter, CompleteLogger, ProgressMeter, ResizeImage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    random.seed(0)
    torch.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = True

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        T.RandomGrayscale(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = T.Compose([
        ResizeImage(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_dataset, num_classes = get_dataset(dataset_name=args.data, root=args.root, task_list=args.sources,
                                                   split='train', download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, drop_last=True)
    val_dataset, _ = get_dataset(dataset_name=args.data, root=args.root, task_list=args.sources, split='val',
                                       download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataset, _ = get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets, split='test',
                                        download=True, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print("train_dataset_size: ", len(train_dataset))
    print('val_dataset_size: ', len(val_dataset))
    print("test_dataset_size: ", len(test_dataset))
    train_iter = ForeverDataIterator(train_loader)

    if args.arch == 'dac':
        backbone = DAC50()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'dac_cc1':
        backbone = DAC50_CC1()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'dac_pc1':
        backbone = DAC50_PC1()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'dac_cc2':
        backbone = DAC50_CC2()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'dac_pc2':
        backbone = DAC50_PC2()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'dac_cc3':
        backbone = DAC50_CC3()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'dac_pc3':
        backbone = DAC50_PC3()
        classifier = Classifier(backbone, num_classes).to(device)
    elif args.arch == 'swin':
        classifier = swin(num_classes).to(device)
    elif args.arch == 'swin_sc':
        classifier = swin_sc(num_classes).to(device)
    elif args.arch == 'swin_p':
        classifier = swin_p(num_classes).to(device)
    elif args.arch == 'swin_k':
        classifier = swin_k(num_classes).to(device)
    elif args.arch == 'mobilenet':
        classifier = mobilenet(num_classes).to(device)
    elif args.arch == 'mobilenet_sc':
        classifier = mobilenet_sc(num_classes).to(device)
    elif args.arch == 'mobilenet_p':
        classifier = mobilenet_p(num_classes).to(device)
    elif args.arch == 'mobilenet_n':
        classifier = mobilenet_n(num_classes).to(device)

    optimizer = SGD(classifier.parameters(), args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint, strict=False)

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_val_acc1 = 0.
    best_test_acc1 = 0.
    not_improved = 0
    prev_acc1 = 0.
    for epoch in range(args.epochs):
        best_val_acc1 = train(train_iter, val_loader, classifier, optimizer, lr_scheduler, epoch, args, logger, best_val_acc1)

        print("Evaluate on validation set...")
        acc1 = validate(val_loader, classifier, args, device)

        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 >= best_val_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        if prev_acc1 >= acc1:
            not_improved += 1
        else:
            not_improved = 0
        prev_acc1 = acc1
        
        best_val_acc1 = max(acc1, best_val_acc1)

        print("Evaluate on test set...")
        best_test_acc1 = max(best_test_acc1, validate(test_loader, classifier, args, device))
        # if not_improved >= 5:
        #     print("Early stopping...")
        #     break
    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args, device)
    print("test acc on test set = {}".format(acc1))
    print("oracle acc on test set = {}".format(best_test_acc1))
    logger.close()


def train(train_iter: ForeverDataIterator, val_loader, model: Classifier, optimizer,
          lr_scheduler: CosineAnnealingLR, epoch: int, args: argparse.Namespace, logger, best_val_acc1):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x, labels = next(train_iter)
        x = x.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y = model(x)

        loss = F.cross_entropy(y, labels)

        cls_acc = accuracy(y, labels)[0]
        losses.update(loss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 and i != 0:
            progress.display(i)

    return best_val_acc1


def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = F.cross_entropy(output, target)

            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS')
    parser.add_argument('-s', '--sources', nargs='+', default=None,
                        help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='target domain(s)')
    parser.add_argument('-a', '--arch', default='dac', type=str, help='model architecture')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=36, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
