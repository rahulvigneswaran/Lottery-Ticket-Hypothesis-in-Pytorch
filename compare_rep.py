import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from utils import checkdir, read_dataset, extract_feat, get_concept, concept_similarity
import cca_core


def compare_representation(arch, dataset, is_instance, prune_type, method):
    """
    Read the saved models, extract the feature matrices, then compare them.
    Args:
        arch: network architecture
        dataset: dataset name
        is_instance: True if applying comparison on instance level, False for concept level
        prune_type: either 'node' or 'weight'
        method: either 'procrustes', 'rsa' (2nd-order isomorphism), or 'cca' (canonical correlation analysis)
    Returns: None. Save scores array to file.

    """

    # read dataset
    testloader, idx_label = read_dataset(dataset)
    checkdir(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/")

    # read acc files to know how many steps
    acc = np.load(f'./dumps/lt/{prune_type}/{arch}/{dataset}/lt_bestaccuracy.dat', allow_pickle=True) / 100
    steps = len(acc)

    score_current_next, score_unpruned = [], []
    count_node = []
    layer = 'fc2'

    # init net
    print('init random')
    init_net = torch.load(f'./saves/{prune_type}/{arch}/{dataset}/initial_state_dict_lt.pth.tar')
    feats_current, labels = extract_feat(net=init_net, layer=layer,
                                         dataloader=testloader, device=device, prune_type=prune_type)
    if not is_instance:
        feats_current = get_concept(feature_mat=feats_current, labels=labels, idx_label=idx_label)
    # unpruned net
    unpruned_net = torch.load(f'./saves/{prune_type}/{arch}/{dataset}/0_model_lt.pth.tar')
    feats_unpruned, labels = extract_feat(net=unpruned_net, layer=layer,
                                          dataloader=testloader, device=device, prune_type=prune_type)
    if not is_instance:
        feats_unpruned = get_concept(feature_mat=feats_unpruned, labels=labels, idx_label=idx_label)

    print('prune and fine tune')
    for i in range(steps):
        net = torch.load(f'./saves/{prune_type}/{arch}/{dataset}/{i}_model_lt.pth.tar')
        alive = np.count_nonzero(np.sum(np.abs(net._modules[layer].weight.detach().cpu().numpy()), axis=1))
        if i != 0 and prune_type == 'node' and alive == count_node[-1][1]:
            continue
        count_node.append([i, alive])
        feats_next, labels = extract_feat(net=net, layer=layer,
                                          dataloader=testloader, device=device, prune_type=prune_type)
        if not is_instance:
            feats_next = get_concept(feature_mat=feats_next, labels=labels, idx_label=idx_label)
        score_c_n = measure_change(feats_current, feats_next, method)
        score_current_next.append(score_c_n)
        score_u_p = measure_change(feats_unpruned, feats_next, method)
        score_unpruned.append(score_u_p)
        feats_current = feats_next
        print('step', i, 'score', score_c_n, score_u_p)

    # save score array to file
    if is_instance:
        np.save(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_instance_current_next.npy", np.array(score_current_next))
        np.save(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_instance_unpruned.npy", np.array(score_unpruned))
    else:
        np.save(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_concept_current_next.npy", np.array(score_current_next))
        np.save(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_concept_unpruned.npy", np.array(score_unpruned))
    np.save(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/alive.npy", np.array(count_node))


def measure_change(original_mat, changed_mat, method):
    """
    Metric to measure the similarity between two feature matrices.
    Args:
        original_mat: feature matrix, shape num_sample x num_feature
        changed_mat: feature matrix, shape num_sample x num_feature
        method: either 'procrustes', 'rsa' (2nd-order isomorphism), or 'cca' (canonical correlation analysis)
    Returns: a scalar score of similarity.
    """

    if method == 'procrustes':
        try:
            mtx1, mtx2, disparity = procrustes(original_mat, changed_mat)  # square error
        except ValueError:  # in some case the feature matrix has all 1 elements  # TODO
            return None
        return np.sqrt(disparity / len(original_mat))  # root mean square error

    elif method == 'rsa':
        # compute similarity matrix
        original_sim_mat = concept_similarity(concept_features=original_mat)
        changed_sim_mat = concept_similarity(concept_features=changed_mat)

        # compute spearman score
        spearman_score = spearmanr(original_sim_mat, changed_sim_mat, axis=None)[0]
        return spearman_score

    elif method == 'cca':
        # CCA require transpose matrices: num_feature x num_sample
        original_mat, changed_mat = original_mat.T, changed_mat.T

        # it is recommended to do SVD as a preprocess step
        original_mat = original_mat - np.mean(original_mat, axis=1, keepdims=True)
        changed_mat = changed_mat - np.mean(changed_mat, axis=1, keepdims=True)
        U1, s1, V1 = np.linalg.svd(original_mat, full_matrices=False)
        U2, s2, V2 = np.linalg.svd(changed_mat, full_matrices=False)
        n_dim_original, n_dim_changed = int(1.0*original_mat.shape[0]), int(1.0*changed_mat.shape[0])
        if n_dim_original == 0:
            n_dim_original = original_mat.shape[0]
        if n_dim_changed == 0:
            n_dim_changed = changed_mat.shape[0]
        original_mat = np.dot(s1[:n_dim_original] * np.eye(n_dim_original), V1[:n_dim_original])
        changed_mat = np.dot(s2[:n_dim_changed] * np.eye(n_dim_changed), V2[:n_dim_changed])

        # compute CCA score
        svcca_results = cca_core.get_cca_similarity(original_mat, changed_mat, epsilon=1e-10, verbose=False)
        return np.mean(svcca_results["cca_coef1"])

    else:
        return 0


def plot(arch, dataset, is_instance, prune_type, method):
    """
    Plot the results of comparisons.
    Args:
        arch: network architecture
        dataset: dataset name
        is_instance: True if applying comparison on instance level, False for concept level
        prune_type: either 'node' or 'weight'
        method: either 'procrustes', 'rsa' (2nd-order isomorphism), or 'cca' (canonical correlation analysis)
    Returns: None. Save the plots to png files.
    """

    acc = np.load(f'./dumps/lt/{prune_type}/{arch}/{dataset}/lt_bestaccuracy.dat', allow_pickle=True) / 100
    alive = np.load(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/alive.npy", allow_pickle=True)
    steps = len(acc)  # 35
    if is_instance:
        score_current_next = np.load(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_instance_current_next.npy", allow_pickle=True)
        score_unpruned = np.load(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_instance_unpruned.npy", allow_pickle=True)
    else:
        score_current_next = np.load(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_concept_current_next.npy", allow_pickle=True)
        score_unpruned = np.load(f"{os.getcwd()}/compare_results/{prune_type}/{arch}/{dataset}/{method}_concept_unpruned.npy", allow_pickle=True)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if prune_type == 'weight':
        x_axis = [str(round(0.9**i * 100, 1)) for i in range(0, steps)]
    else:
        x_axis = [str(i) for i in alive[:, 1]]
        acc = acc[alive[:, 0]]
    plt_acc = ax1.plot(x_axis, acc, color='b', label="best accs")
    plt_c_n = ax2.plot(x_axis, np.append(score_current_next[1:], [None]), color='g', label="current_next")
    plt_u_p = ax2.plot(x_axis, score_unpruned, color='orange', label="unpruned_pruned")
    ax1.tick_params(axis='x', labelrotation=90)
    fig.subplots_adjust(bottom=0.15)
    plt.title(f"{arch}, {dataset}, {prune_type}, {method}")
    plt.xlabel("% keep")
    ax1.set_ylim(0.095, 1.05)
    ax1.set_ylabel('acc')
    if method == 'procrustes':
        ax2.set_ylabel('rmse')
        ax2.set_ylim(0, 0.3)
    elif method == 'rsa':
        ax2.set_ylabel('spearmanr')
        ax2.set_ylim(0.3, 1.05)
    else:
        ax2.set_ylabel('mean_cca_sim')
        ax2.set_ylim(-0.05, 1.05)
    # added these lines
    lns = plt_acc + plt_c_n + plt_u_p
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')
    plt.grid(color="gray")

    if is_instance:
        plt.savefig(f"{os.getcwd()}/plots/lt/{prune_type}/{arch}/{dataset}/{method}_instance.png", dpi=500)
    else:
        plt.savefig(f"{os.getcwd()}/plots/lt/{prune_type}/{arch}/{dataset}/{method}_concept.png", dpi=500)
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prune_type = 'node'
    method = 'cca'
    for d in ['mnist', 'cifar10']:
        compare_representation(arch='lenet5', dataset=d, is_instance=True, prune_type=prune_type, method=method)
        plot(arch='lenet5', dataset=d, is_instance=True, prune_type=prune_type, method=method)

