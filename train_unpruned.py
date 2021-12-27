import argparse
import numpy as np
from tqdm import tqdm
import torch
import os
import glob
from itertools import combinations
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from main import train, test, weight_init
from compare_rep import measure_change, read_dataset, extract_feat, get_concept
import utils


def train_unpruned(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader
    transform = None
    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        traindataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        testdataset = datasets.MNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        traindataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR10('../data', train=False, transform=transform)
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet

    elif args.dataset == "fashionmnist":
        traindataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
        testdataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar100":
        traindataset = datasets.CIFAR100('../data', train=True, download=True, transform=transform)
        testdataset = datasets.CIFAR100('../data', train=False, transform=transform)
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet

        # If you want to add extra datasets paste here

    else:
        print("\nWrong Dataset choice \n")
        exit()

    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               drop_last=False)
    # train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                              drop_last=True)

    for _ite in range(args.n_solution):
        # Importing Network Architecture
        if args.arch_type == "fc1":
            model = fc1.fc1().to(device)
        elif args.arch_type == "lenet5":
            model = LeNet5.LeNet5().to(device)
        elif args.arch_type == "alexnet":
            model = AlexNet.AlexNet().to(device)
        elif args.arch_type == "vgg16":
            model = vgg.vgg16().to(device)
        elif args.arch_type == "resnet18":
            model = resnet.resnet18().to(device)
        elif args.arch_type == "densenet121":
            model = densenet.densenet121().to(device)
            # If you want to add extra model paste here
        else:
            print("\nWrong Model choice\n")
            exit()

        # Weight Initialization
        model.apply(weight_init)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()  # Default was F.nll_loss

        # Pruning
        # NOTE First Pruning Iteration is of No Compression
        # bestacc = 0.0
        best_accuracy = 0
        # comp = np.zeros(ITERATION, float)
        # bestacc = np.zeros(ITERATION, float)
        # step = 0
        all_loss = np.zeros(args.end_iter, float)
        all_accuracy = np.zeros(args.end_iter, float)

        print(f"\n--- model {_ite} ---")
        pbar = tqdm(range(args.end_iter))

        for iter_ in pbar:
            # Frequency for Testing
            if iter_ % args.valid_freq == 0:
                accuracy = test(model, test_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/saves_unpruned/{args.arch_type}/{args.dataset}/")
                    torch.save(model,
                               f"{os.getcwd()}/saves_unpruned/{args.arch_type}/{args.dataset}/{_ite}_model.pth.tar")

            # Training
            loss = train(model, train_loader, optimizer, criterion)
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy

            # Frequency for Printing Accuracy and Loss
            if iter_ % args.print_freq == 0:
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')

        # Plotting Loss (Training), Accuracy (Testing), Iteration Curve
        # NOTE Loss is computed for every iteration while Accuracy is computed only for every {args.valid_freq} iterations. Therefore Accuracy saved is constant during the uncomputed iterations.
        # NOTE Normalized the accuracy to [0,100] for ease of plotting.
        plt.plot(np.arange(1, (args.end_iter) + 1),
                 100 * (all_loss - np.min(all_loss)) / np.ptp(all_loss).astype(float), c="blue", label="Loss")
        plt.plot(np.arange(1, (args.end_iter) + 1), all_accuracy, c="red", label="Accuracy")
        plt.title(f"Loss Vs Accuracy Vs Iterations ({args.dataset},{args.arch_type})")
        plt.xlabel("Iterations")
        plt.ylabel("Loss and Accuracy")
        plt.legend()
        plt.grid(color="gray")
        utils.checkdir(f"{os.getcwd()}/plots/unpruned/{args.arch_type}/{args.dataset}/")
        plt.savefig(
            f"{os.getcwd()}/plots/unpruned/{args.arch_type}/{args.dataset}/LossVsAccuracy_{_ite}.png",
            dpi=1200)
        plt.close()

        # Dump Plot values
        utils.checkdir(f"{os.getcwd()}/dumps/unpruned/{args.arch_type}/{args.dataset}/")
        all_loss.dump(f"{os.getcwd()}/dumps/unpruned/{args.arch_type}/{args.dataset}/all_loss_{_ite}.dat")
        all_accuracy.dump(
            f"{os.getcwd()}/dumps/unpruned/{args.arch_type}/{args.dataset}/all_accuracy_{_ite}.dat")


def compare_unpruned(arch, dataset, method, prune_type):
    testloader, idx_label = read_dataset(dataset)
    steps = len(glob.glob(f'./saves_unpruned/{arch}/{dataset}/*'))
    layer = 'fc2'

    concept_mats = []
    for i in range(steps):
        net = torch.load(f'./saves_unpruned/{arch}/{dataset}/{i}_model.pth.tar')
        feat, labels = extract_feat(net=net, layer=layer, dataloader=testloader, device=device, prune_type=prune_type)
        concept_mat = get_concept(feature_mat=feat, labels=labels, idx_label=idx_label)
        concept_mats.append(concept_mat)

    scores = []
    for comb in combinations(range(steps), 2):
        score = measure_change(concept_mats[comb[0]], concept_mats[comb[1]], method)
        scores.append(score)
        print(comb, score)

    print('mean', np.mean(scores), 'std', np.std(scores))


if __name__ == "__main__":
    # from gooey import Gooey
    # @Gooey

    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_solution", default=15, type=int)
    parser.add_argument("--lr", default=1.2e-3, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=100, type=int)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--valid_freq", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist | cifar10 | fashionmnist | cifar100")
    parser.add_argument("--arch_type", default="fc1", type=str,
                        help="fc1 | lenet5 | alexnet | vgg16 | resnet18 | densenet121")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # FIXME resample
    resample = False

    # Looping Entire process
    # for i in range(0, 5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_unpruned(args)
    compare_unpruned(arch=args.arch_type, dataset=args.dataset, method='rsa', prune_type='weight')
