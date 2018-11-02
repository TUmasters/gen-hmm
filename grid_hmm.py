#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
import itertools


class GridHMM:
    def __init__(self, width, height, p):
        self.width = width
        self.height = height
        self.p = p
        self.states = list(itertools.product(range(width), range(height)))
        self.observations = self.states

    def _check_bound(self, x):
        return x[0] >= 0 and x[0] < self.width \
          and x[1] >= 0 and x[1] < self.height

    def _get_num_surround(self, x):
        return len(self._get_adj_states(x))

    def _get_adj_states(self, x):
        states = []
        for dx, dy in itertools.product(*[[-1, 0, 1]]*2):
            if dx == 0 and dy == 0:
                continue
            x1 = (x[0]+dx, x[1]+dy)
            if self._check_bound(x1):
                states += [x1]
        return states

    def _get_transition_states(self, x):
        adj_states = [(x[0]+dx,x[1]) for dx in [-1,1]
                          if self._check_bound((x[0]+dx,x[1]))]
        below_states = [(x[0]+dx, x[1]+1) for dx in [-1,0,1]
                            if self._check_bound((x[0]+dx,x[1]+1))]
        return adj_states, below_states

    def initial_p(self, x=None):
        """Returns the probability of entering state x initially. If x
        is not given, gives all nonzero initialization probabilities in
        the form (state, probability)"""
        if x:
            if x[1] > 0:
                return 0
            else:
                return 1.0/self.width
        else:
            return [((x,0),1.0/self.width) for x in range(self.width)]

    def transition_p(self, x1, x2=None):
        """Returns transition probability from state x1 to x2. If x2 is
        not given, then returns all nonzero transition probabilities in
        the form (state, probability)."""
        adj, below = self._get_transition_states(x1)
        probs = [0.05 for x3 in adj] + [0.3 for x3 in below]
        probs = [p / sum(probs) for p in probs]
        probs = zip(adj+below, probs)
        if x2:
            for x3, p in probs:
                if x2 == x3:
                    return p
            return 0
        else:
            return probs

    def observation_p(self, x, o=None):
        """Returns the probability of observing o in state x. If o is
        not given, then returns all nonzero observation probabilities in
        the form (observation, probability)."""
        if not self._check_bound(x):
            raise ValueError("Invalid state provided: {}".format(x))

        if o:
            if o == x:
                return self.p
            if abs(x[0] - o[0]) <= 1 and abs(x[1] - o[1]) <= 1:
                return (1-self.p) / self._gen_num_surround(x)
            else:
                return 0.
        else:
            adj = self._get_adj_states(x)
            probs = [(x, self.p)]
            probs += [(x1, (1-self.p) / len(adj)) for x1 in adj]
            return probs

    def sample(self, time):
        states, p = tuple(zip(*self.initial_p()))
        x = [npr.choice(list(states) + [()], p=list(p) + [0.0])]
        for _ in range(1, time):
            states, p = tuple(zip(*self.transition_p(x[-1])))
            x += [npr.choice(list(states) + [()], p=list(p) + [0.0])]
        o = []
        for x_i in x:
            states, p = tuple(zip(*self.observation_p(x_i)))
            o += [npr.choice(list(states) + [()], p=list(p) + [0.0])]
        sample = [{'t': t, 'x': s[0], 'o': s[1]} for t, s in enumerate(zip(x, o))]
        return sample
