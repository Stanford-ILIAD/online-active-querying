import pandas as pd
from IPython import embed
import os
import matplotlib.pyplot as plt

df_door = []
df_peg = []

for x in os.listdir('logs'):
    if x.startswith('door'):
        try: df_door.append(pd.read_csv(f'logs/{x}/progress.csv')['eval/Returns Mean'].iloc[-1])
        except Exception as e: print(e)
    elif x.startswith('peg'):
        try: df_peg.append(pd.read_csv(f'logs/{x}/progress.csv')['eval/Returns Mean'].iloc[-1])
        except Exception as e: print(e)

plt.figure()
pd.Series(df_door).hist()
plt.savefig('door.png')

plt.figure()
pd.Series(df_peg).hist()
plt.savefig('peg.png')
