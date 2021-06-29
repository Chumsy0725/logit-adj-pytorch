import argparse


def get_arguments():
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
    parser.add_argument('--resume', default=False, type=bool, help='load from save weight')
    parser.add_argument('--evaluate', default=False, type=bool, help='evaluate model')
    parser.add_argument('--log_val', help='compute val acc', type=int, default=10)

    parser.add_argument('--logit_adj_post', help='adjust logits post hoc',
                        type=bool, default=False)
    parser.add_argument('--logit_adj_train', help='adjust logits post hoc',
                        type=bool, default=False)
    parser.add_argument('--tro', help='adjust logits post hoc',
                        type=float, default=1.0)

    return parser
