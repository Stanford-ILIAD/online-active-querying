import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd


plt.xlabel('Questions')
plt.ylabel('Score')
plt.title('Highway Environment')

evoi = np.array(torch.load('results/evoi-highway')).T
uncertainty = np.array(torch.load('results/uncertain-highway')).T
random = np.array(torch.load('results/random-highway')).T

data = [('evoi', x[0], x[1]) for x in evoi] + [('uncertainty', x[0], x[1]) for x in uncertainty] + [('random', x[0], x[1]) for x in random]
df = pd.DataFrame(data, columns=['method', 'queries', 'score'])
df['score'] = df['score'] - 20

plt.figure()
sns.barplot(data=df, x='method', y='queries', ci=68, capsize=.3)
plt.ylabel('queries')
plt.title('Driving Evironment Number of Queries')
plt.savefig('results/highway-queries.png', dpi=300)

plt.figure()
sns.barplot(data=df, x='method', y='score', ci=68, capsize=.3)
locs, labels = plt.yticks()
plt.yticks(locs, [x + 20 for x in locs])
plt.ylabel('score')
plt.title('Driving Environment Score')
plt.savefig('results/highway-scores.png', dpi=300)


