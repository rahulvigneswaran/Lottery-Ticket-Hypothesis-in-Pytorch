import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm


DPI = 1200
prune_iterations = 35
arch_types = ["fc1", "lenet5", "resnet18"]
datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]


for arch_type in tqdm(arch_types):
    for dataset in tqdm(datasets):
        d = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_compression.dat", allow_pickle=True)
        b = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_bestaccuracy.dat", allow_pickle=True)
        c = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/reinit_bestaccuracy.dat", allow_pickle=True)

        #plt.clf()
        #sns.set_style('darkgrid')
        #plt.style.use('seaborn-darkgrid')
        a = np.arange(prune_iterations)
        plt.plot(a, b, c="blue", label="Winning tickets") 
        plt.plot(a, c, c="red", label="Random reinit") 
        plt.title(f"Test Accuracy vs Weights % ({arch_type} | {dataset})") 
        plt.xlabel("Weights %") 
        plt.ylabel("Test accuracy") 
        plt.xticks(a, d, rotation ="vertical") 
        plt.ylim(0,100)
        plt.legend() 
        plt.grid(color="gray") 

        plt.savefig(f"{os.getcwd()}/plots/lt/combined_plots/combined_{arch_type}_{dataset}.png", dpi=DPI, bbox_inches='tight') 
        plt.close()
        #print(f"\n combined_{arch_type}_{dataset} plotted!\n")
