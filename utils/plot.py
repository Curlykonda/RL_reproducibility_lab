import numpy as np
import matplotlib.pyplot as plt


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def episode_durations(ep, ep2=None):
    plt.plot(smooth(ep, 10))
    if ep2:
        plt.plot(smooth(ep2, 10))
    plt.title('Episode durations per episode')
    plt.show()