# logit-adj-pytorch
## PyTorch implementation of the paper: Long-tail Learning via Logit Adjustment
This code implements the paper:
[Long-tail Learning via Logit Adjustment](https://arxiv.org/abs/2007.07314) : Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar. ICLR 2021.

### Running the code
```python
# To produce baseline (ERM) results:
python main.py --dataset cifar10-lt

# To produce posthoc logit-adjustment results:
python main.py --dataset cifar10-lt  --logit_adj_post 1

# To produce logit-adjustment loss results:
python main.py --dataset cifar10-lt  --logit_adj_train 0

# To monitor the training progress using Tensorboard:
tensorboard --logdir logs


```

Replace **cifar10-lt** above with **cifar100-lt** to obtain results for the CIFAR-100 long-tail dataset.

### Results

|   | Baseline | Post-hoc logit adjustment | Logit-adjusted loss|
| ------------- | ------------- | ------- | -------      |
| CIFAR10LT  | 0.7127  |   0.7816 | 0.7857 |
| CIFAR100LT | 0.3985 | 0.4404 | 0.4402 |
