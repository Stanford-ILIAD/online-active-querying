import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd

orange = '#E48B10'
blue = '#42838F'
gray = '#808080'

SMALL = 12
MEDSMALL = 16
MEDIUM = 20
BIGGER = 25

plt.rc('font', size=BIGGER)         # Default
plt.rc('axes', titlesize=MEDIUM)    # Axes titles
plt.rc('axes', labelsize=MEDIUM)    # x and y labels
plt.rc('xtick', labelsize=SMALL)    # x tick labels
plt.rc('ytick', labelsize=SMALL)    # y tick labels
plt.rc('legend', fontsize=MEDSMALL) # Legend labels
plt.rc('figure', titlesize=BIGGER)  # Figure title

plt.rc('font', family='serif')
plt.rc('font', serif='Palatino')
plt.rc('figure', dpi=100)
plt.rc('text', usetex=True)


def gen_plot(env, title, out):
    plt.figure()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    evoi = np.array(sorted(torch.load(f'results/gridworld-evoi-{env}'))).T
    uncertainty = np.array(sorted(torch.load(f'results/gridworld-uncertainty-{env}'))).T
    random = np.array(sorted(torch.load(f'results/gridworld-random-{env}'))).T

    p = plt.plot([], [], c=orange, label='ours')
    plt.plot(*evoi, c=p[0].get_color(), zorder=-1)
    plt.scatter(*evoi, c=p[0].get_color(), zorder=1)

    p = plt.plot([], [], label='uncertainty', c=blue)
    plt.plot(*uncertainty, c=p[0].get_color(), zorder=-1)
    plt.scatter(*uncertainty, c=p[0].get_color(), zorder=1)

    p = plt.plot([], [], label='random', c=gray)
    plt.plot(*random, c=p[0].get_color(), zorder=-1)
    plt.scatter(*random, c=p[0].get_color(), zorder=1)

    plt.xlabel('Queries')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()

    plt.savefig(f'results/{out}', dpi=300, bbox_inches='tight')

gen_plot('empty', 'Empty Grid Environment', 'empty.png')
gen_plot('maze', 'Maze Grid Environment', 'maze.png')
gen_plot('rooms', 'Rooms Grid Environment', 'rooms.png')
