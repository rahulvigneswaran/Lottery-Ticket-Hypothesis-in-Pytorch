# Lottery Ticket Hypothesis in Pytorch [![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]() 

This repository contains a **Pytorch** implementation of the paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) by [Jonathan Frankle](https://github.com/jfrankle) and [Michael Carbin](https://people.csail.mit.edu/mcarbin/) that can be **easily adapted to any model/dataset**.
		
## Requirements
```
pip3 install -r requirements.txt
```
## How to run the code ? 
### Using datasets/architectures included with this repository
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
### Using datasets/architectures that are not included with this repository
- Adding a new architecture :
	- For example, if you want to add an architecture named `new_model` with `mnist` dataset compatibility. 
		- Go to `/archs/mnist/` directory and create a file `new_model.py`.
		- Now paste your **Pytorch compatible** model inside `new_model.py`.
		- **IMPORTANT** : Make sure the *input size*, *number of classes*, *number of channels*, *batch size* in your `new_model.py` matches with the corresponding dataset that you are adding (in this case, it is `mnist`).
		- Now open `main.py` and go to `line 36` and look for the comment `# Data Loader`. Now find your corresponding dataset (in this case, `mnist`) and add `new_model` at the end of the line `from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet`.
		- Now go to `line 82` and add the following to it :
			```
			elif args.arch_type == "new_model":
        		model = new_model.new_model_name().to(device)
			``` 
			Here, `new_model_name()` is the name of the model that you have given inside `new_model.py`.
- Adding a new dataset :
	- For example, if you want to add a dataset named `new_dataset` with `fc1` architecture compatibility.
		- Go to `/archs` and create a directory named `new_dataset`.
		- Now go to /archs/new_dataset/` and add a file named `fc1.py` or copy paste it from existing dataset folder.
		- **IMPORTANT** : Make sure the *input size*, *number of classes*, *number of channels*, *batch size* in your `new_model.py` matches with the corresponding dataset that you are adding (in this case, it is `new_dataset`).
		- Now open `main.py` and goto `line 58` and add the following to it :
			```
			elif args.dataset == "cifar100":
        		traindataset = datasets.new_dataset('../data', train=True, download=True, transform=transform)
        		testdataset = datasets.new_dataset('../data', train=False, transform=transform)from archs.new_dataset import fc1
			``` 
			**Note** that as of now, you can add dataset that are [natively available in Pytorch](https://pytorch.org/docs/stable/torchvision/datasets.html). 

Kindly [raise an issue](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/issues) if you have any problem with the instructions. 


## Datasets and Architectures that were already tested

|              | fc1                | LeNet5                | AlexNet                | VGG16                | Resnet18                 |
|--------------|:------------------:|:---------------------:|:----------------------:|:--------------------:|:------------------------:|
| MNIST        | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |  	:heavy_check_mark:	   |
| CIFAR10      | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |	:heavy_check_mark:	   |
| FashionMNIST | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |	:heavy_check_mark:	   |
| CIFAR100     | :heavy_check_mark: |  :heavy_check_mark:   |   :heavy_check_mark:   |  :heavy_check_mark:  |	:heavy_check_mark:     |


## Repository Structure
```
Lottery-Ticket-Hypothesis-in-Pytorch
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

## Interesting papers that are related to Lottery Ticket Hypothesis which I enjoyed 
- [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)


## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase you are facing any difficulty with the code base or if you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch/issues)

<a href="https://www.buymeacoffee.com/MfgDK7aSC" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

