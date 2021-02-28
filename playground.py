import argparse
import os
import random
import sys
import time
import types


import kaggle_environments as kg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import torch

from agents.omdagents import EWAAgent
from agents.randomagent import RandomAgent

configuration = { 
      "actTimeout" : 1,
      "agentTimeout": 60,
      "runTimeout" : 1200 
   }
save_folder = Path('XP_'+str(time.time()))
if not save_folder.is_dir():
    save_folder.mkdir()

env = kg.make("rps", debug=True, configuration = configuration)

# Contestants ! 
# Making agents from classes
ewa_agent = EWAAgent(sample_mode=False)
ewa_agent_sample = EWAAgent(sample_mode=True)
random_agent = RandomAgent()

contestants = [
    # (EWAAgent(sample_mode=False), "EWA_nosample"),
    (EWAAgent(sample_mode=True), "EWA_sample"),
    (RandomAgent(), "Random"),
    ('rock', "default_rock"),
    ('reactionary', "default_reactionary"),
    ('statistical', "default_statistical")
]

# Init
num_contestants = len(contestants)
counter = [0 for c in contestants]
scores = [600 for c in contestants]
variances = [0 for c in contestants]
steps = 1000
num_episodes = 500
alpha = 0.1
history = pd.DataFrame(columns=["agent", "score"], index=range(num_episodes*2))

def is_playable(agent):
    return isinstance(agent, (str, types.FunctionType))

# Run the contest
start = time.time()
with (save_folder / "description.txt").open("w") as f:
    f.write(str(configuration)+"\n\n"+str([c[1] for c in contestants]))

for epi in range(0, num_episodes*2, 2):
    agentsId = np.random.choice(range(num_contestants), 2, replace=False)
    agents = [contestants[idx][0] if is_playable(contestants[idx][0]) else contestants[idx][0].play for idx in agentsId]
    results = kg.evaluate('rps', agents, configuration,  num_episodes=1)

    # Update history
    history.iloc[epi] = [contestants[agentsId[0]][1], results[0][0]]
    history.iloc[epi+1] = [contestants[agentsId[1]][1], results[0][1]]
    if i % 100 == 0:
        print(f"Round {i} completed")

# Final scores
stop = time.time()
print(f"Contest ended in {(stop -start)/60 :.2} minutes.")
sns.boxplot(x="agent", y="score", data=history, palette="Set2")
plt.savefig(save_folder / "boxplot.png")
print(f"Result graph saved at {save_folder / 'boxplot.png'}")
history.to_csv(save_folder / "scores.csv")
print(f"Result graph saved at {save_folder / 'boxplot.png'}")