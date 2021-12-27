# ANCHOR Libraries
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.spatial.distance import cosine
from itertools import combinations

import torchvision
import torchvision.transforms as transforms


def read_dataset(dataset):
    if dataset == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        batch_size = 4
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        idx_label = {0: 'airplane',
                     8: 'ship',
                     1: 'car',
                     9: 'truck',
                     2: 'bird',
                     6: 'frog',
                     3: 'cat',
                     5: 'dog',
                     4: 'deer',
                     7: 'horse',
                     }
    elif dataset == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
        batch_size = 4
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        idx_label = {0: '0',
                     1: '1',
                     2: '2',
                     3: '3',
                     4: '4',
                     5: '5',
                     6: '6',
                     7: '7',
                     8: '8',
                     9: '9',
                     }
    return testloader, idx_label


def extract_feat(net, layer, dataloader, device, prune_type):
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    net._modules[layer].register_forward_hook(get_features('feat'))

    feats, labels = [], []
    # loop through batches
    with torch.no_grad():
        for images, label in dataloader:
            outputs = net(images.to(device))
            feats.append(features['feat'].detach().cpu().numpy())
            labels.append(label.numpy())

    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    if prune_type == 'node':
        weight = net._modules[layer].weight.detach().cpu().numpy()
        node_mask = np.sum(np.abs(weight), axis=1) != 0
        feats = feats[:, node_mask]

    return feats, labels


def get_concept(feature_mat, labels, idx_label):
    num_classes = len(idx_label)
    avg_features = np.zeros([num_classes, feature_mat.reshape(feature_mat.shape[0], -1).shape[1]])
    for i in idx_label.keys():
        mask = (labels == i) * 1
        filter_mat = np.take(feature_mat, np.nonzero(mask), axis=0)[0]
        filter_mat = filter_mat.reshape(filter_mat.shape[0], -1)
        avg_features[i] = np.mean(filter_mat, axis=0)
    return avg_features


def concept_similarity(concept_features):
    num_classes = concept_features.shape[0]

    # avg_features = avg_features / np.sqrt(np.sum(avg_features**2, axis=1))[:, None]
    sim_mat = np.ones([num_classes, num_classes])
    for comb in combinations(range(num_classes), 2):
        sim_mat[comb[0], comb[1]] = 1 - cosine(concept_features[comb[0]], concept_features[comb[1]])
        #         sim_mat[comb[0], comb[1]] = spearmanr(avg_features[comb[0]], avg_features[comb[1]])[0]
        sim_mat[comb[1], comb[0]] = sim_mat[comb[0], comb[1]]

    return sim_mat


# ANCHOR Print table of zeros and non-zeros count
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(
            f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(
        f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total / nonzero:10.2f}x  ({100 * (total - nonzero) / total:6.2f}% pruned)')
    return (round((nonzero / total) * 100, 5))


def original_initialization(mask_temp, initial_state_dict):
    global model

    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask_temp[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]
    step = 0


# ANCHOR Checks of the directory exist and if not, creates a new directory
def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# FIXME
def plot_train_test_stats(stats,
                          epoch_num,
                          key1='train',
                          key2='test',
                          key1_label=None,
                          key2_label=None,
                          xlabel=None,
                          ylabel=None,
                          title=None,
                          yscale=None,
                          ylim_bottom=None,
                          ylim_top=None,
                          savefig=None,
                          sns_style='darkgrid'
                          ):
    assert len(stats[key1]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key1, len(stats[key1]),
                                                                                         epoch_num)
    assert len(stats[key2]) == epoch_num, "len(stats['{}'])({}) != epoch_num({})".format(key2, len(stats[key2]),
                                                                                         epoch_num)

    plt.clf()
    sns.set_style(sns_style)
    x_ticks = np.arange(epoch_num)

    plt.plot(x_ticks, stats[key1], label=key1_label)
    plt.plot(x_ticks, stats[key2], label=key2_label)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    if yscale is not None:
        plt.yscale(yscale)

    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        plt.ylim(top=ylim_top)

    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fancybox=True)

    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()


def extract_best_acc(arch, dataset):
    import re
    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    acc_files = glob.glob(f'{os.getcwd()}/dumps/lt/{arch}/{dataset}/*accuracy*.dat')
    acc_files = natural_sort(acc_files)
    acc_files.reverse()

    best_acc = []
    for file in acc_files:
        acc = np.load(file, allow_pickle=True)
        best_acc.append(max(acc))
    np.save(f"{os.getcwd()}/dumps/lt/{arch}/{dataset}/lt_bestaccuracy.dat", np.array(best_acc))


def _test():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # batch_size = 4
    # trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=0)
    # testset = torchvision.datasets.CIFAR10(root='../data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=0)
    # idx_label = {0: 'airplane',
    #              8: 'ship',
    #              1: 'car',
    #              9: 'truck',
    #              2: 'bird',
    #              6: 'frog',
    #              3: 'cat',
    #              5: 'dog',
    #              4: 'deer',
    #              7: 'horse',
    #              }
    # for i in range(0, 28, 1):
    #     test_net = torch.load(f'./saves/lenet5/cifar10/{i}_model_lt.pth.tar')
    #     feats_current, labels = extract_feat(net=test_net, layer='fc2', dataloader=testloader)
    #     avg = np.mean(np.abs(feats_current), axis=0)
    #     # print(avg)
    #     print(i)
    #     print(np.sum(np.isclose(0, avg)) / len(avg))
    #     weights = test_net.fc2.weight.detach().cpu().numpy()
    #     print(0.9**i, np.count_nonzero(np.sum(weights, axis=1)) / weights.shape[0])
    #     print(np.sum(weights, axis=1).shape)
    global model
    model = torch.load(f'./saves/lenet5/mnist/0_model_lt.pth.tar', map_location=torch.device('cpu'))

    def _make_mask(model):
        global step
        global mask
        step = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                step = step + 1
        mask = [None] * step
        step = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                mask[step] = np.ones_like(tensor)
                step = step + 1
        step = 0

    _make_mask(model)
    for i in range(50):
        print('\n', 0.9**i)
        prune_node(percent=10, current_ite=i)


def prune_node(percent, current_ite, resample=False, reinit=False, **kwargs):
    global step
    global mask
    global model

    # Calculate percentile value
    step = 0
    for name, param in model.named_parameters():

        # We do not prune bias term
        if 'weight' in name:
            if 'fc2' in name:
                tensor = param.data.cpu().numpy()
                sum_abs_row = np.sum(np.abs(tensor), axis=1)
                alive = sum_abs_row[np.nonzero(sum_abs_row)]
                if (len(alive)/tensor.shape[0]) > (((100 - percent)/100)**current_ite):
                    percentile_value = np.percentile(alive, percent)
                else:
                    print('already')
                    break

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = (sum_abs_row > percentile_value).astype(np.float32)
                new_mask = np.tile(new_mask.reshape(-1, 1), mask[step].shape[1])
                print(new_mask)

                # Apply new weight and mask
                print('before', np.count_nonzero(np.sum(param.data.detach().cpu().numpy(), axis=1)),
                      np.count_nonzero(np.sum(param.data.detach().cpu().numpy(), axis=1)) / param.data.shape[0])
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                print('after', np.count_nonzero(np.sum(param.data.detach().cpu().numpy(), axis=1)),
                      np.count_nonzero(np.sum(param.data.detach().cpu().numpy(), axis=1)) / param.data.shape[0])
                mask[step] = new_mask
            step += 1
    step = 0

