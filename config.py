import argparse


def get_arguments():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of the paper: Long-tail Learning via Logit Adjustment'
    )

    parser.add_argument('--dataset', default="cifar10-lt", type=str, help='Dataset to use.')
    parser.add_argument('--data_home', default="data", type=str,
                        help='Directory where data files are stored.')
    parser.add_argument('--num_workers', default=2, type=int, metavar='N',
                        help='number of workers at dataloader')

    parser.add_argument('--epochs', default=1241, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--evaluate', default=0, type=int, help='evaluate model')
    parser.add_argument('--log_val', help='compute val acc', type=int, default=10)

    parser.add_argument('--logit_adj_post', help='adjust logits post hoc', type=int, default=0, choices=[0, 1])
    parser.add_argument('--logit_adj_train', help='adjust logits in trainingc', type=int, default=1, choices=[0, 1])
    parser.add_argument('--tro', help='tro', type=float, default=1.0)

    return parser
