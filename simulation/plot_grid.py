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


def sort_by(arr):
    return np.array(sorted(list(map(tuple, list(arr.T))))).T

def gen_plot(env, title, out, vecdiff):

    plt.figure()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    evoi = np.array((torch.load(f'results/gridworld-evoi-{env}'))).T
    uncertainty = np.array((torch.load(f'results/gridworld-uncertainty-{env}'))).T
    random = np.array((torch.load(f'results/gridworld-random-{env}'))).T

    p = plt.plot([], [], c=orange, label='ours')
    plt.plot(*sort_by(evoi), c=p[0].get_color(), zorder=-1)
    plt.scatter(*sort_by(evoi), c=p[0].get_color(), zorder=1)
    ox, oy = -5, -5
    locs = [0, 60, 70, 80, 90, 100, 120]
    print(evoi.T)
    for i, (x, y) in enumerate(reversed(evoi.T)):
        if i in locs:
            if x < ox + 0.5:
                continue
            p = evoi.shape[1] - i - 1
            plt.annotate(f'$c={1e-4 * 1.05 ** (p):.2g}$', (x, y), vecdiff(x, y),fontsize=9, arrowprops=dict(arrowstyle='->'))
            ox = x
            oy = y

    p = plt.plot([], [], label='uncertainty', c=blue)
    plt.plot(*sort_by(uncertainty), c=p[0].get_color(), zorder=-1)
    plt.scatter(*sort_by(uncertainty), c=p[0].get_color(), zorder=1)

    p = plt.plot([], [], label='random', c=gray)
    plt.plot(*sort_by(random), c=p[0].get_color(), zorder=-1)

    plt.scatter(*sort_by(random), c=p[0].get_color(), zorder=1)

    plt.xlabel('Queries')
    plt.ylabel('Score')
    plt.title(title)

    plt.savefig(f'results/{out}', dpi=300, bbox_inches='tight')

def vecdiff(x, y):
    dx = 1 - x
    dy = 0.9 - y
    norm = (np.abs(dx / 16) ** 2.2 + np.abs(dy * 1.) ** 2.2) ** (1/2.2) * 6
    return x + dx / norm - .8, y + dy / norm

gen_plot('empty', 'Empty Grid Environment', 'empty.png', vecdiff)

def vecdiff(x, y):
    dx = 1 - x
    dy = 0.7 - y
    norm = np.sqrt((dx / 28) ** 2 + (dy * 1.2) ** 2) * 8
    if x == 0:
        dy += 0.4
        dx -= 0.2
    return x + dx / norm - .9, y + dy / norm

gen_plot('maze', 'Maze Grid Environment', 'maze.png', vecdiff)

def vecdiff(x, y):
    dx = 1 - x
    dy = 0.7 - y
    norm = np.sqrt((dx / 48) ** 2 + (dy * 1) ** 2) * 8
    if y > .72:
        dy += 0.05
    if x == 0.:
        dx -= 2.5
    return x + dx / norm - .9, y + dy / norm

gen_plot('rooms', 'Rooms Grid Environment', 'rooms.png', vecdiff)
