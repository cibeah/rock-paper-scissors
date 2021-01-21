import sys
import os
import kaggle_environments as kg

# Authorized libraries : Python Standard Library, gym, numpy, scipy, pytorch (cpu only)

import numpy as np
import torch

from agents.omdagents import EWAAgent

# Symbols :
# 0 : Rock
# 1 : Paper
# 2: Scissors

# Make the environment 
# * episodeSteps 	Maximum number of steps in the episode.
# * agentTimeout 	Maximum runtime(seconds) to initialize an agent.
# * actTimeout 	Maximum runtime(seconds) to obtain an action from an agent.
# * runTimeout 	Maximum runtime(seconds) of an episode(not necessarily DONE).

# Agents from kaggle:  rock, paper, scissors, copy_opponent, reactionary, counter_reactionary, statistical

# Agent definition : function of two variabless :
# *  observation:  dict : 
# **      'remainingOverageTime': int
# **      'lastOpponentAction': 0
# **      'step': int
# * configuration : 
# **      'episodeSteps': int
# **      'agentTimeout': int
# **      'actTimeout': int
# **      'runTimeout': int
# **      'isProduction': boolean
# **      'signs': int 
# **      'tieRewardThreshold': int

# Making the environment
configuration = { 
      "actTimeout" : 1,
      "agentTimeout": 60,
      "runTimeout" : 1200 
   }

env = kg.make("rps", debug=True, configuration = configuration)

# Making agents from classes
ewa_agent = EWAAgent(sample_mode=False)
ewa_agent_sample = EWAAgent(sample_mode=True)

# Loading agent from file 
random_agent = "main.py"

# Evaluate 
agents = [ewa_agent_sample.play, ewa_agent.play]

steps = 1000
num_episodes = 10
results = kg.evaluate('rps', agents, configuration,  num_episodes= num_episodes)
agent_0_wins = [1 if rewards[0] > rewards[1] else 0 for rewards in results]
agent_0_draws= [1 if rewards[0] == rewards[1] else 0 for rewards in results]
print(f"""(Agent 0) Results. \n******Wins: {sum(agent_0_wins)/num_episodes * 100  :.2f}% of episodes
      Draws: {sum(agent_0_draws)/num_episodes * 100  :.2f}%""")
print(f"(Agent 0) Average rewards: {sum([rewards[0] for rewards in results])/num_episodes :.2f}.")
