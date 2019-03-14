import csv
import pdb
import numpy as np
results = {}

with open("hparam_tuning_results_hospital_100_reduced_hparam_space.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        optimizer, seed, learning_rate, weight_decay, prec, rec, rep_recall, f1, rep_f1 = line
        if (learning_rate, weight_decay) not in results:
            results[(learning_rate, weight_decay)] = []
        results[(learning_rate, weight_decay)].append(float(f1))

avg_results = {}
for learning_rate, weight_decay in results:
    vals = results[(learning_rate, weight_decay)]
    avg_results[(np.log10(float(learning_rate)), np.log10(float(weight_decay)))] = sum(vals) / len(vals)



import itertools
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = []
    y = []
    z = []

    for (learning_rate, weight_decay) in avg_results:
        x.append(learning_rate)
        y.append(weight_decay)
        z.append(avg_results[(learning_rate, weight_decay)])
        print((learning_rate, weight_decay, avg_results[(learning_rate, weight_decay)]))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)
    # import pdb; pdb.set_trace()
    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot
    plt.imshow(zz, extent=(x.min(), x.max(), y.max(), y.min()))
    plt.scatter(x, y, c=z)
    plt.colorbar()
    ax = plt.gca()
    ax.set_xlabel("Log Learning Rate")
    ax.set_ylabel("Log Weight Decay Rate")
    plt.show()

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

main()
