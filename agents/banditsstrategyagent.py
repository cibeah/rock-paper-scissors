from random import choice, choices
import numpy as np
from kaggle_environments.envs.rps.utils import get_score

CHOICES = [0, 1, 2]


def count(hist):
    hist = np.array(hist)
    freqs = [0, 0, 0]
    for i in range(1, 4):
        freqs[i - 1] = (hist == i).sum()
    return np.array(freqs)


## Strategies

## Strategies take player histories and return a choice.


def random_strategy(our_hist, their_hist):
    """
    Return random action.
    """
    if len(our_hist) == 0 or len(their_hist) == 0:
        return choice(CHOICES)
    return choice(CHOICES)


def proportional_strategy(our_hist, their_hist):
    """
    Return ideal response of their majority use,
    with the same frequency.
    """
    if len(our_hist) == 0 or len(their_hist) == 0:
        return choice(CHOICES)
    freqs = count(their_hist)
    prediction_for_them = choices(CHOICES, weights=freqs)[0]
    return CHOICES[(prediction_for_them + 1) % 3]


def greedy_proportional_strategy(our_hist, their_hist):
    """
    Like proportional_strategy, but argmax instead of sampling.
    """
    if len(our_hist) == 0 or len(their_hist) == 0:
        return choice(CHOICES)
    freqs = count(their_hist)
    prediction_for_them = np.argmax(freqs)
    return CHOICES[(prediction_for_them + 1) % 3]


def two_proportional_strategy(our_hist, their_hist):
    """
    Same as proportional_strategy but with sequences of length two.
    """
    if len(our_hist) == 0 or len(their_hist) == 0:
        return choice(CHOICES)
    last_action = their_hist[-1]
    two_seq = zip(their_hist[:-1], their_hist[1:])
    freqs = count(
        [next_action for action, next_action in two_seq if action == last_action]
    )
    prediction_for_them = choices(CHOICES, weights=freqs)[0]
    return CHOICES[(prediction_for_them + 1) % 3]


def two_greedy_proportional_strategy(our_hist, their_hist):
    """
    Same as proportional_strategy but with sequences of length two.
    """
    if len(our_hist) == 0 or len(their_hist) == 0:
        return choice(CHOICES)
    last_action = their_hist[-1]
    two_seq = zip(their_hist[:-1], their_hist[1:])
    freqs = count(
        [next_action for action, next_action in two_seq if action == last_action]
    )
    prediction_for_them = np.argmax(freqs)
    return CHOICES[(prediction_for_them + 1) % 3]


def counter_reactionary(our_hist, their_hist):
    """
    Same as gym's counter_reactionary.
    """
    if len(our_hist) == 0 or len(their_hist) == 0:
        return choice(CHOICES)
    elif our_hist[-1] == CHOICES[(their_hist[-1] + 1) % 3]:
        return CHOICES[(our_hist[-1] + 1) % 3]
    elif our_hist[-1] == CHOICES[(their_hist[-1] + 2) % 3]:
        return CHOICES[(our_hist[-1] + 2) % 3]


STRATEGIES = [
    random_strategy,
    proportional_strategy,
    greedy_proportional_strategy,
    two_proportional_strategy,
    two_greedy_proportional_strategy,
    counter_reactionary,
]


class BanditsOfStrategies:
    """
    Inspired from:
    https://rhettinger.github.io/rock_paper.html#our-approach

    Collect history of plays and learn bandit over strategies.

    env\me  | R  P  S
    --------|--------
         R  | 0  -1 1
         P  | 1  0 -1
         S  | -1 1  0
    """

    def __init__(self):
        self.strategies = STRATEGIES
        self.weights = np.ones(len(self.strategies))

        self.our_hist = []
        self.their_hist = []
        self.our_last_action = None
        self.predictions = np.zeros(len(self.strategies))

    def act(self, obs, reward, done, info):
        pass

    def play(self, obs, configuration):
        # def play(self, obs, configuration, step):
        if obs.step == 0:
            # if step == 0:
            self.our_last_action = choice(CHOICES)
            return self.our_last_action

        # update histories
        self.our_hist.append(self.our_last_action)
        their_last_action = obs.lastOpponentAction
        self.their_hist.append(their_last_action)

        # update weights for each arm
        for i, action in enumerate(self.predictions):
            if action == CHOICES[(their_last_action + 1) % 3]:
                self.weights[i] += 1

        # prediction from each arm
        self.predictions = [
            strategy(self.our_hist, self.their_hist) for strategy in self.strategies
        ]
        self.our_last_action = self.predictions[np.argmax(self.weights)]

        return self.our_last_action
