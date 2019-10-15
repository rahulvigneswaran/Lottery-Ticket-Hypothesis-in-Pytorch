## Lottery ticket hypothesis
This repository contains an implementation of Lottery Ticket Hypothesis that can be easily adapted to any model/dataset.
		
### Requirements
```
pip install -r requirements.txt
```
### Instructions
```
python3 main.py --prune_type=lt --arch_type=fc1 --dataset=mnist --prune_percent=10 --prune_iterations=35
```
- `--prune_type` : Type of pruning  
	- Options : `lt` - Lottery Ticket Hypothesis, `reinit` - Random reinitialization
	- Default : `lt`
- `--arch_type`	 : Type of architecture
	- Options : `fc1` - Simple fully connected network, `lenet5` - LeNet5, `AlexNet` - AlexNet, `resnet18` - Resnet18, `vgg16` - VGG16 
	- Default : `fc1`
- `--dataset`	: Choice of dataset 
	- Options : `mnist`, `fashionmnist`, `cifar10`, `cifar100` 
	- Default : `mnist`
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
	- Default : `10`
- `--prune_iterations`	: Number of cycle of pruning that should be done. 
	- Default : `35`
- `--lr`	: Learning rate 
	- Default : `1.2e-3`
- `--batch_size`	: Batch size 
	- Default : `60`
- `--end_iter`	: Number of Epochs 
	- Default : `100`
- `--print_freq`	: Frequency for printing accuracy and loss 
	- Default : `1`
- `--valid_freq`	: Frequency for Validation 
	- Default : `1`
- `--gpu`	: Decide Which GPU the program should use 
	- Default : `0`

### Datasets and Models tested

|              | fc1                | LeNet5                | AlexNet                | VGG16                | Resnet18                 |
|--------------|:------------------:|:---------------------:|:----------------------:|:--------------------:|:------------------------:|
| MNIST        | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |                     !|  	:heavy_check_mark:	   |
| CIFAR10      | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark: !|	:heavy_check_mark:	   |
| FashionMNIST | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |                     !|	:heavy_check_mark:	   |
| CIFAR100     | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark: !|	:heavy_check_mark:     |


### Directory Tree
```
subspace-analysis-master
├── archs
│   ├── cifar10
│   │   ├── AlexNet.py
│   │   ├── densenet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   ├── cifar100
│   │   ├── AlexNet.py
│   │   ├── fc1.py
│   │   ├── LeNet5.py
│   │   ├── resnet.py
│   │   └── vgg.py
│   └── mnist
│       ├── AlexNet.py
│       ├── fc1.py
│       ├── LeNet5.py
│       ├── resnet.py
│       └── vgg.py
├── combine_plots.py
├── dumps
├── main.py
├── plots
├── README.md
├── requirements.txt
├── saves
└── utils.py

```

### Interesting Reads
-
- 
-

