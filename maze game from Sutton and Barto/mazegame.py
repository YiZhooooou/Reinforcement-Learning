import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

# A class that load the csv file of a map to generate a good maze
class Maze:

    # Constructor
    def __init__(self):

        # A grid of map with height and width
        self.mazeMap = []
        self.map_height = 0
        self.map_width = 0

        # All possible actions
        self.action_U = 0
        self.action_D = 1
        self.action_L = 2
        self.action_R = 3
        self.actions = [self.action_U, self.action_D, self.action_L, self.action_R]

        # Get the two maze map files
        file_dir = os.path.dirname(__file__)
        self.first_maze_file = os.path.join(file_dir, 'map1.csv')
        self.second_maze_file = os.path.join(file_dir, 'map2.csv')

        # define map shape 
        self.start = 2
        self.goal = 3
        self.validPath = 1
        self.barrier = 0

        self.use_first_maze()

    def use_first_maze(self):
        # Start with the first maze
        with open(self.first_maze_file) as csv_file:
            self.mazeMap = [list(map(int, rec)) for rec in csv.reader(csv_file, delimiter=',')]
        # Get the height and width
        self.map_height = len(self.mazeMap)
        self.map_width = len(self.mazeMap[0])

    def use_second_maze(self):
        # change to the second maze
        with open(self.second_maze_file) as csv_file:
            self.mazeMap = [list(map(int, rec)) for rec in csv.reader(csv_file, delimiter=',')]
        # Get the height and width
        self.map_height = len(self.mazeMap)
        self.map_width = len(self.mazeMap[0])

    def take_action(self, state, action):
        # get the current postion 
        x, y = state

        # take whatever action is
        if action == self.action_U:
            x = max(x - 1, 0)
        elif action == self.action_D:
            x = min(x + 1, self.map_height - 1)
        elif action == self.action_L:
            y = max(y - 1, 0)
        elif action == self.action_R:
            y = min(y + 1, self.map_width - 1)

        # if reach barrier
        if self.mazeMap[x][y] == self.barrier:
            x, y = state

        # return the reward
        if self.mazeMap[x][y] == self.goal:
            reward = 1.0
        else:
            reward = 0.0
        
        return [x,y], reward

    def get_start_location(self):

        return [[x,y] for x in np.arange(self.map_height) for y in np.arange(self.map_width) if self.mazeMap[x][y] == 2]

    def get_goal_location(self):

        return [[x,y] for x in np.arange(self.map_height) for y in np.arange(self.map_width) if self.mazeMap[x][y] == 3]

# Class to do the dynamic q learning
# From the psydocode in Sutton page 135
class DynaQ:

    # Constructor
    def __init__(self, mazeMap, params):
        
        # get the maze map and paramters
        self.mazeMap = mazeMap
        self.params = params

        # initialize q value and the model
        self.q = np.zeros((self.mazeMap.map_height, self.mazeMap.map_width, len(self.mazeMap.actions)))
        self.model = np.empty((self.mazeMap.map_height, self.mazeMap.map_width, len(self.mazeMap.actions)), dtype = list)
        self.t = 0 # time counter to compute r + k sqrt(t)
        self.ts = np.zeros((self.mazeMap.map_height, self.mazeMap.map_width, len(self.mazeMap.actions)))

    # From the tabular dyna q doing step b
    def choose_action(self, state, method):    ######## may be deterministic 
        # based on the method using update action in different way
        # first is the basic DynaQ and DynaQ+, they are the same as the psydo code in textbook
        if method == 'D-Q' or method == 'D-Q+':
            # explore
            if np.random.binomial(1, self.params.epsilon) == 1:
                action = random.choice(self.mazeMap.actions)
                return action
            # exploit 
            else:
                values = self.q[state[0], state[1], :]
                action = random.choice([a for a, value in enumerate(values) if value == np.max(values)]) # not sure about declartion
                return action
        elif method == 'Alter-D-Q+':
            # explore
            if np.random.binomial(1, self.params.epsilon) == 1:
                action = random.choice(self.mazeMap.actions)
                return action
             # exploit with the extra reward
            else:
                values = self.q[state[0], state[1], :] + self.params.k * np.sqrt(self.t - self.ts[state[0], state[1], :])
                action = random.choice([a for a, value in enumerate(values) if value == np.max(values)]) # not sure about declartion
                return action
    
    # Do the DynaQ
    def do_DynaQ_Algorithm(self, method):
        # initialization: start from beginning with the first maze
        self.mazeMap.use_first_maze()
            
        # cumulative reward
        rewards = np.zeros(self.params.totalSteps)
        totalReward = 0
        # current step
        curStep = 0
        # which map is using 
        isSecondMap = False

        # Start doing Tabular Dyna Q
        while curStep < self.params.totalSteps:
            # step a
            # set current state to the beginning
            curState = random.choice(self.mazeMap.get_start_location())
            start_step = curStep
            
            while curState not in self.mazeMap.get_goal_location():
                curStep += 1

                # step b and c: choose and take an action
                action = self.choose_action(curState, method)
                nextState, reward = self.mazeMap.take_action(curState, action)
                totalReward += reward
                
                # step d: update q value
                self.q[curState[0], curState[1], action] += self.params.alpha * (reward + self.params.gamma * np.max(self.q[nextState[0], nextState[1], :]) - self.q[curState[0], curState[1], action])

                # step e: update model
                self.model[curState[0], curState[1], action] = [nextState, reward]

                # step f: planning
                self.t += 1
                self.ts[curState[0], curState[1], action] = self.t
                for s in range(self.params.n):
                    #random previously observed state
                    rState = random.choice([[x, y] for x in np.arange(self.mazeMap.map_height) for y in np.arange(self.mazeMap.map_width) if not all(v is None for v in self.model[x, y, :])])

                    #random action previously taken in S
                    #when we are running DynaQ and alterDynaQ+
                    if method == 'D-Q' or method == 'Alter-D-Q+':
                        rAction = random.choice([a for a in self.mazeMap.actions if self.model[rState[0], rState[1], a] is not None])
                        r_new_state, rReward = self.model[rState[0], rState[1], rAction]
                        self.q[rState[0], rState[1], rAction] += self.params.alpha * (rReward + self.params.gamma * np.max(self.q[r_new_state[0], r_new_state[1], :]) - self.q[rState[0], rState[1], rAction])

                    # DynaQ+
                    else:
                        # explore
                        rAction = random.choice(self.mazeMap.actions)

                        if self.model[rState[0], rState[1], rAction] is not None:
                            r_new_state, rReward = self.model[rState[0], rState[1], rAction]
                        else:
                            # this is when a state action pair has never happened before
                            r_new_state, rReward = rState, 0
                        
                        # update reward by adding the extra reward
                        rReward += self.params.k * np.sqrt(self.t - self.ts[rState[0], rState[1], rAction])

                        #update q table
                        self.q[rState[0], rState[1], rAction] += self.params.alpha * (rReward + self.params.gamma * np.max(self.q[r_new_state[0], r_new_state[1], :]) - self.q[rState[0], rState[1], rAction])

                curState = nextState
                
            rewards[start_step:curStep] = totalReward

            # check if we need to change the map
            if curStep > self.params.secondMapPoint and isSecondMap == False:
                self.mazeMap.use_second_maze()
                isSecondMap = True
        
        return rewards
    


# Class that you can change all paramaters

class Params:

    # Constructor
    def __init__(self):
        # Three possible actions
        self.methods = ['D-Q', 'D-Q+', 'Alter-D-Q+']

        # alpha for step size
        self.alpha = 0.5

        # epsilon for greedy
        self.epsilon = 0.1

        # gamma for discount
        self.gamma = 0.95

        # k for the extra reward
        self.k = 1e-4

        # n for n-step 
        self.n = 6

        # total steps
        self.totalSteps = 3000

        # time to second map
        self.secondMapPoint = 1000


def play_maze_game():
    print('start running')

    maze = Maze()
    print('maze size: height ', maze.map_height,' width ', maze.map_width)
    params = Params()

    outputs = []
    for method in params.methods:
        print('method: ', method)
        sumReward = np.zeros(params.totalSteps)
        for i in range(30):
            result = np.zeros(params.totalSteps)
            dynaQ = DynaQ(maze, params)
            result = dynaQ.do_DynaQ_Algorithm(method)
            sumReward += result
            
        sumReward /= 30
        outputs.append((sumReward, method))
        
    for output in outputs:
        reward, method = output
        plt.plot(reward, label = method)
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')

    plt.show()

play_maze_game()


