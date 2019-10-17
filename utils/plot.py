import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.autograd import Variable
import pandas as pd
import torch


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def episode_durations_uncer(ep):
    mean = ep.mean(axis=0)
    std = ep.var(axis=0)
    plt.plot(smooth(mean, 1))
    plt.fill_between(np.arange(ep.shape[1]), mean - std, mean + std, alpha=0.3)
    plt.title('Episode durations per episode')
    plt.show()


def episode_durations(ep, ep2=None):
    plt.plot(smooth(ep, 1))
    if ep2:
        plt.plot(smooth(ep2, 1))
    plt.title('Episode durations per episode')
    plt.show()


def visualize_policy(model):
    X = np.random.uniform(-1.2, 0.6, 10000)
    Y = np.random.uniform(-0.07, 0.07, 10000)
    Z = []

    for i in range(len(X)):
        _, temp = torch.max(model(Variable(torch.from_numpy(np.array([X[i], Y[i]]))).type(torch.FloatTensor)), dim=-1)
        z = temp.item()
        Z.append(z)

    Z = pd.Series(Z)
    colors = {0: 'blue', 1: 'lime', 2: 'red'}
    colors = Z.apply(lambda x: colors[x])
    labels = ['Left', 'Right', 'Nothing']
    fig = plt.figure(3, figsize=[7, 7])
    ax = fig.gca()
    plt.set_cmap('brg')
    surf = ax.scatter(X, Y, c=Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Policy')
    recs = []
    for i in range(0, 3):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=sorted(colors.unique())[i]))
    plt.legend(recs, labels, loc=4, ncol=3)
    # fig.savefig('Policy.png')
    plt.show()
