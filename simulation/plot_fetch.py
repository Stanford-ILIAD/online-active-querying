import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd

orange = '#E48B10'
blue = '#42838F'
gray = '#808080'

SMALL = 8.5
MEDSMALL = 10
MEDIUM = 11
BIGGER = 14

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
#plt.rc('text', usetex=True)

plt.xlabel('Questions')
plt.ylabel('Score')
plt.title('Fetch Environment')

evoi = np.array(torch.load('results/evoi-fetch')).T[:140]
uncertainty = np.array(torch.load('results/uncertainty-fetch-alt')).T[:140]
random = np.array(torch.load('results/random-fetch')).T[:140]

data = [('ours', x[0], x[1]) for x in evoi] + [('uncertainty', x[0], x[1]) for x in uncertainty] + [('random', x[0], x[1]) for x in random]
df = pd.DataFrame(data, columns=['method', 'queries', 'score'])
df['score'] = df['score'] + 50

plt.figure(figsize=(8,2))
plt.tick_params(
    axis='both',
    which='both',
    left=False,
    )
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
sns.barplot(data=df, y='method', x='queries', ci=68, capsize=.3, palette=[orange, blue, gray], orient='h')
plt.ylabel(None)
plt.xlabel('queries')
plt.title('Fetch Block Pushing Number of Queries')
plt.savefig('results/fetch-queries.png', dpi=300)

plt.figure(figsize=(8,2))
plt.tick_params(
    axis='both',
    which='both',
    left=False,
    )
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
sns.barplot(data=df, y='method', x='score', ci=68, capsize=.3, palette=[orange, blue, gray], orient='h')
locs, labels = plt.xticks()
plt.xticks(locs, [x - 50 for x in locs])
plt.ylabel(None)
plt.xlabel('score')
plt.title('Fetch Block Pushing Score')
plt.savefig('results/fetch-scores.png', dpi=300)


