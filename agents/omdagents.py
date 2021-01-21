import random
import numpy as np
from kaggle_environments.envs.rps.utils import get_score


class EWAAgent():
    """ A simple Online Mirror Descent Agent
    => Exponentiaed Weighted Algorithm !
    
    We predict a vector x of shape (3,)
    We return a sample of action / or the argmax ? to the environment

    We have the cost matrix:

    env\me  R  P  S
        R   0  -1 1
        P   1  0 -1 
        S   -1 1 0

    => 3 different cost functions.
    """
    def __init__(self, eta=1/3, sample_mode=False):
        self.theta = np.zeros(3)
        self.lastPrediction = None
        self.cost_matrix  = np.array([[0,-1,1], [1,0,-1], [-1,1,0]])
        self.eta = eta
        self.sample_mode = sample_mode

    def act(self, obs, reward, done, info):
        pass

    def play(self, obs, configuration):
        if obs.step == 0:
            self.lastPrediction = np.ones(3) / 3
            self.theta = np.zeros(3)
            return int(random.randint(0,2))
        # score = get_score(self.myLastAction, obs.lastOpponentAction)

        loss_vector = self.cost_matrix[obs.lastOpponentAction]
        # loss = loss_vector@self.lastPrediction
        self.theta -= self.eta * loss_vector
        self.lastPrediction = np.exp(self.theta) / np.sum(np.exp(self.theta))
        
        if self.sample_mode:
            return random.choices(range(3), self.lastPrediction, k=1)[0]
        else:
            return int(self.lastPrediction.argmax()) #other option: sample
