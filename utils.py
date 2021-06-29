import torch
import os
import numpy as np

classes_10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """ Save the training model """
    torch.save(state, filename)


def accuracy(outputs, labels):
    """Computes accuracy for given outputs and ground truths"""

    _, predicted = torch.max(outputs, 1)
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    return acc


def class_accuracy(test_loader, model, args, classes=classes_10):
    """ Computes the accuracy for each class"""

    with torch.no_grad():
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            output = model(images)

            if args.logit_adj_post:
                output = output - args.logit_adjustments

            _, predicted = torch.max(output, 1)

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        results = {}
        avg_acc = 0
        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            avg_acc += acc
            print(f'Accuracy of {classes[i]}: {acc} %')
            results["class/" + classes[i]] = acc
        results["AA"] = avg_acc / 10
        print("Average accuracy:{}".format(avg_acc / 10))
        return results


def make_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass


def log_hyperparameter(args):
    whole_dict = vars(args)
    hyperparam = {}
    keys = ['logit_adj_post', 'logit_adj_train', 'tro']
    for key in keys:
        hyperparam[key] = whole_dict[key]
    return hyperparam


def log_folders(args):
    log_dir = 'logs'
    exp_dir = 'adjpost:{}_adjpost:{}_tro:{}'.format(
        args.logit_adj_post,
        args.logit_adj_train,
        args.tro)
    exp_loc = os.path.join(log_dir, exp_dir)
    model_loc = os.path.join(exp_loc, "model_weights")
    make_dir(log_dir)
    make_dir(exp_loc)
    make_dir(model_loc)
    return exp_loc, model_loc


def compute_adjustment(train_loader, args):
    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(args.device)
        for j in target:
            key = str(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    adjustments = args.tro * np.log(label_freq_array)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(args.device)
    return adjustments
