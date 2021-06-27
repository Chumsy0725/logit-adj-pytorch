import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from model import resnet32
from dataset import CIFAR100LTNPZDataset
from utils import AverageMeter, save_checkpoint, accuracy, class_accuracy

parser = argparse.ArgumentParser(
    description='PyTorch implementation of the paper: Long-tail Learning via Logit Adjustment'
)

parser.add_argument('--epochs', default=1241, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number ')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--logit_adj_post', help='adjust logits post hoc', type=bool, default=False)
parser.add_argument('--logit_adj_train', help='adjust logits post hoc', type=bool, default=True)
parser.add_argument('--tro', help='adjust logits post hoc', type=float, default=1.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_acc = 0
args = parser.parse_args()


def main():
    global args, best_acc
    # cant do both at same time
    assert (not (args.logit_adj_post and args.logit_adj_train))
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet32())
    model = model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # CIFAR-10 dataset
    train_dataset = CIFAR100LTNPZDataset(root='data',
                                        train=True,
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])]))

    test_dataset = CIFAR100LTNPZDataset(root='data',
                                       train=False,
                                       transform=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)

    args.logit_adjustments = compute_adjustment(train_loader)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[604, 926, 1128])

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        # print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        if epoch % args.save_every == 0:
            acc = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        # is_best = acc > best_acc
        # best_acc = max(acc, best_acc)

        # if epoch > 0 and epoch % args.save_every == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_acc': best_acc,
        #     }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        # save_checkpoint({
        #     'state_dict': model.state_dict(),
        #     'best_acc': best_acc,
        # }, is_best, filename=os.path.join(args.save_dir, 'model_{}.th'))

    save_checkpoint(model.state_dict(), True, filename=os.path.join(args.save_dir,
                                                                    'model_acc:{}_adjlogit:{}_tro:{}.th'.format(acc,
                                                                                                                args.logit_adj_post,
                                                                                                                args.tro)))

    class_accuracy(val_loader, model)


def train(train_loader, model, criterion, optimizer, epoch):
    """ Run one train epoch """

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        input_var = inputs.to(device)
        target_var = target

        # compute output
        output = model(input_var)
        if args.logit_adj_train:
            output = output + args.logit_adjustments
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Training Loss | {loss.avg:.4f} \t'
        #           'Training Accuracy | {accuracies.avg:.3f} '.format(epoch, i, len(train_loader),
        #                                                              batch_time=batch_time, loss=losses,
        #                                                              accuracies=accuracies))

    print('Epoch: [{0}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Training Loss | {loss.avg:.4f} \t'
          'Training Accuracy | {accuracies.avg:.3f} '.format(epoch,
                                                             batch_time=batch_time, loss=losses,
                                                             accuracies=accuracies))


def validate(val_loader, model, criterion):
    """ Run evaluation """

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = inputs.to(device)
            target_var = target.to(device)

            # compute output
            output = model(input_var)
            if args.logit_adj_post:
                output = output - args.logit_adjustments

            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Time | {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Validation Loss | {loss.avg:.4f}\t'
              'Validation Accuracy |  {accuracies.avg:.3f}'.format(batch_time=batch_time, loss=losses,
                                                                   accuracies=accuracies))
    return accuracies.avg


def compute_adjustment(train_loader):
    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        for j in target:
            key = str(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    adjustments = args.tro * np.log(label_freq_array)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    return adjustments


if __name__ == '__main__':
    main()
