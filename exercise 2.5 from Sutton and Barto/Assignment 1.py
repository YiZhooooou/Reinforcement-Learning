# Assignment 1 code for exercise 2.5 from Sutton and Barto
# Yi Zhou
# Sep. 23rd. 2021

import numpy as np 
import matplotlib.pyplot as plt

# Construct K-armed Bandit problem environment of Bandit
class Bandit:
    def __init__(self, k, e, m, a, bandit_type):
        # Number of arms
        self.k = k
        # Number of Bandit
        self.m = m
        # Epsilon
        self.e = e
        # Alpha
        self.a = a

        # the bendit type that is using 
        self.bandit_type = bandit_type

        # Math function computing real mean reward q*
        self.q_star = np.zeros((self.m, self.k))

        # Sample average method table
        self.N = np.zeros((self.m, self.k))

        # estimate reward mean based on the action (initialized to zeros)
        self.Q = np.zeros((self.m, self.k))
    
    # choose action for each bandit and return the action and its reward with sample average method
    def run(self):

        if self.bandit_type == 'sample average':
            # Pick one action
            action = [np.argmax(row) if np.random.binomial(1, self.e) == 0 else int(np.random.uniform(0,self.k)) for row in self.Q[:, ]]
            # Get the reward
            reward = [np.random.normal(self.q_star[i, value], 1) for i, value in enumerate(action)] 

            # Insert new action and reward to the table
            self.N = np.reshape([self.N[i, :] + Bandit.one_hot(value, self.k) for i, value in enumerate(action)], (self.m,self.k))
            self.Q = np.reshape([self.Q[i, :] + (1 / self.N[i, value]) * (reward[i] - self.Q[i, value]) * Bandit.one_hot(value, self.k) for i, value in enumerate(action)], (self.m, self.k))

    
        elif self.bandit_type == 'constant step size':
            # Pick one action
            action = [np.argmax(row) if np.random.binomial(1, self.e) == 0 else int(np.random.uniform(0, self.k)) for row in self.Q[:, ]]

            # Get rewards
            reward = [np.random.normal(self.q_star[i, value], 1) for i, value in enumerate(action)]

            # Insert estimate reward the table
            self.Q = np.reshape([self.Q[i, :] + self.a * (reward[i] - self.Q[i, value]) * Bandit.one_hot(value, self.k) for i, value in enumerate(action)], (self.m, self.k))

        return action, reward

    @staticmethod
    # Learn one_hot from matteocasolari's solution of this exercise
    def one_hot(pos, leng):
        """
        :param position: position of the '1' value
        :param length: length of the array
        :return: one-hot encoded vector, with all '0' except for one '1'
        """
        assert leng > pos, 'position {:d} greater than or equal to length {:d}'.format(pos, leng)
        oh = np.zeros(leng)
        oh[pos] = 1
        return oh

# A class to set the test number of Bendit
class SetNums:
    def __init__(self):
        # amount of bandit's arms
        self.k = 10
        # amount of bendit
        self.m = None
        # set epsilon for epsilon greed algortim
        self.e = 0.1
        # set mean and varience(sd)
        self.mu, self.sigma = 0, 0.01
        # set total steps for one circle
        self.steps = None
        # set alpha for constant step size
        self.a = 0.1

def do_exercise():
    print('Start computing')

    # Apply stats from exercise
    stats = SetNums()

    stats.m = 2000
    stats.steps = 10000
    stats.bandit_types = ['sample average', 'constant step size']
    

    # average reward collection
    ars = []
    # Percentage of Optimal Action collection
    poas = []

    for bandit_type in stats.bandit_types:
        # average reward 
        ar = []
        # Percentage of Optimal Action 
        poa = []

        # start to run begin with bandit
        bandit = Bandit(stats.k, stats.e, stats.m, stats.a, bandit_type)

        for j in range(stats.steps):
      
            # do sample average
            action, reward = bandit.run()

            # Store all the reands and compute percentage of optimal values
            ar.append(np.average(reward))
            poa.append(np.average([100 if value == np.argmax(bandit.q_star[i, :]) else 0 for i, value in enumerate(action)]))

            # make a change on the true value 
            bandit.q_star += np.random.normal(stats.mu, stats.sigma, (stats.m, stats.k))

        ars.append(ar)
        poas.append(poa)

    #draw the plot
    plot(stats.bandit_types, ars, poas)

def plot(bandit_types, ars, poas):

    for bandit_type, ar, poa in zip(bandit_types, ars, poas):
        plt.figure(0)
        plt.plot(ar, label = 'sample avergae')
        plt.legend(loc = 'best')
        plt.xlabel('step')
        plt.ylabel('Average reward')

        plt.figure(1)
        plt.plot(poa, label = 'sample avergae')
        plt.legend(loc = 'best')
        plt.xlabel('step')
        plt.ylabel('% Optimal action')

do_exercise()
plt.show()
print('plotting finished')