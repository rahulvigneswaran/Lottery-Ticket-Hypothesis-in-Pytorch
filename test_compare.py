import pickle

from compare_rep import measure_change


def test():
    # CCA
    print('\nCCA')
    with open('./result_compare/node/lenet5/cifar10/feat_mats_instance', 'rb') as f:
        feat_mats_instance = pickle.load(f)

    cca_u_p, cca_c_n = [], []
    for i in range(len(feat_mats_instance) - 1):
        score_u_p = measure_change(feat_mats_instance[0], feat_mats_instance[i], 'cca')
        cca_u_p.append(score_u_p)
        score_c_n = measure_change(feat_mats_instance[i], feat_mats_instance[i+1], 'cca')
        cca_c_n.append(score_c_n)
        print('num_neuron', feat_mats_instance[i].shape[1],
              ' unpruned vs. pruned', score_u_p, ' current vs. next', score_c_n)

    # Spearman
    print('\nSpearman')
    with open('./result_compare/node/lenet5/cifar10/feat_mats_concept', 'rb') as f:
        feat_mats_concept = pickle.load(f)

    spearman_u_p, spearman_c_n = [], []
    for i in range(len(feat_mats_concept) - 1):
        score_u_p = measure_change(feat_mats_concept[0], feat_mats_concept[i], 'rsa')
        spearman_u_p.append(score_u_p)
        score_c_n = measure_change(feat_mats_concept[i], feat_mats_concept[i+1], 'rsa')
        spearman_c_n.append(score_c_n)
        print('num_neuron', feat_mats_concept[i].shape[1],
              ' unpruned vs. pruned', score_u_p, ' current vs. next', score_c_n)


if __name__ == '__main__':
    test()
