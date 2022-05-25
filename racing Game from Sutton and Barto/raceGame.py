import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# A class that load the csv file of a map to generate a good map
class raceMap:

    # Constructor
    def __init__(self):

        # A grid of map with height and width
        self.raceMap = []
        self.map_height = 0
        self.map_width = 0

        # Variables for velocity
        self.MAX_Vel = 5
        self.MIN_Vel = -5

        # Acceleration
        self.Max_acc = 1

        # probablity that no increment of velocity 
        self.NO_acc = 0.1

         # All possible actions
        self.actions = [[a_i, a_j] for a_j in range(-self.Max_acc, self.Max_acc + 1) for a_i in range(-self.Max_acc, self.Max_acc + 1)]

        # The reward or state of each position
        self.ON_state = 1
        self.OUT_state = 0
        self.START_state = 2
        self.FIN_state = 3

        # Draw the map with beautiful color
        self.Black = (0, 0, 0) # backgroud color
        self.Red = (1, 0, 0) # track color
        self.Yellow = (1, 1, 0) # start line color
        self.Green = (0, 128/255, 0) # finish line color
        self.White = (254/255, 254/255, 254/255) # car color

        # read the structure of map from a csv file
        file_dir = os.path.dirname(__file__)
        path = os.path.join(file_dir, 'map.csv')
        with open(path) as csv_file:
            self.raceMap = [list(map(int, rec)) for rec in csv.reader(csv_file, delimiter=',')]

        #Get the height and width
        self.map_height = len(self.raceMap)
        self.map_width = len(self.raceMap[0])

    def print_track(self, state = None):
        track_color = self.raceMap.copy()
        #color all the state
        track_color = [[self.Black if s == self.OUT_state
                                    else self.Red if s == self.ON_state
                                    else self.Yellow if s == self.START_state
                                    else self.Green for s in r] for r in track_color]
        # color the car
        if state is not None:
            i, j, v_i, v_j = state
            track_color[i][j] = self.White
        
        im = plt.imshow(track_color, origin='lower', interpolation='none', animated=True)
        plt.gca().invert_yaxis()

        return im
        
    # Take the action for next step
    # state has four variables: x and y cooridinates, x and y velocity from previous action
    # and decide the action to make the next move 
    # return the updated state, reward and check if a run is finished
    def take_action(self, state, action, win):
        # the position and velocity of current state
        i, j, v_i, v_j = state

        a_i, a_j = action

        # check if accelerate 
        if np.random.binomial(1, self.NO_acc) == 1 and not win:
            a_i = 0
            a_j = 0

        #record the previous state status
        pre_i = i
        pre_j = j 
        v_i += a_i
        v_j += a_j
        i -= v_i
        j += v_j
        # a flag to see if one episode is win already
        done = False   

        # Collect all the state that has been visted and check the state of car after one run to see if it hits the finished line
        visited = [[i_, j_] for i_ in range (min (i, pre_i), max(i, pre_i) + 1) for j_ in range (min (j, pre_j), max(j, pre_j) + 1)]
        check_visited = [self.raceMap[i_][j_] if 0 <= i_ < self.map_height and 0 <= j_ < self.map_width
                        else self.OUT_state for i_, j_ in visited]
        if self.FIN_state in check_visited:
            done = True
        elif self.OUT_state in check_visited:
            i, j, v_i, v_j = random.choice([[i, j, 0, 0] for i, j in self.get_locations(self.START_state)])
        
        return [i, j, v_i, v_j], -1, done
    
    def get_locations(self, state_type):

        return [[i, j] for i in np.arange(self.map_height) for j in np.arange(self.map_width) if self.raceMap[i][j] == state_type]

    # all actions for the policy 
    def allActions(self, state):
        actions = []
        allActions = self.actions.copy()
        _, _, v_i, v_j = state

        # only store the valid action with valid velocity
        for a in allActions:
            a_i, a_j = a
            if v_i + a_i < self.MIN_Vel or v_i + a_i > self.MAX_Vel:
                continue
            if v_j + a_j < self.MIN_Vel or v_j + a_j > self.MAX_Vel:
                continue
            if v_i + a_i == 0 and v_j + a_j == 0:
                continue
            actions.append(a)
        return actions

# the class of MC off policy algorithm
class MonteCarlo:
    def __init__(self, raceMap, params):

        self.raceMap = raceMap
        self.params = params

        # set up the date structure of the state-action(x, y coords, velocity in x and y, and its acceleration = 5)
        state_actions_s = (self.raceMap.map_height, self.raceMap.map_width, self.raceMap.MAX_Vel - self.raceMap.MIN_Vel + 1, self.raceMap.MAX_Vel - self.raceMap.MIN_Vel + 1, 2 * self.raceMap.Max_acc + 1, 2 * self.raceMap.Max_acc + 1) 

        # set up the date structure of state (x, y coords, velocity in x and y)
        states_s = (self.raceMap.map_height, self.raceMap.map_width, self.raceMap.MAX_Vel - self.raceMap.MIN_Vel + 1, self.raceMap.MAX_Vel - self.raceMap.MIN_Vel + 1)

        # Initialize, for all s ∈ S, a ∈ A(s)
        self.Q = np.full(state_actions_s, params.init_q)
        self.C = np.zeros(state_actions_s)
        self.policy = np.empty(states_s, dtype=object)
        # start with the random action
        for i in range(states_s[0]):
            for j in range(states_s[1]):
                for v_i in range(self.raceMap.MIN_Vel, self.raceMap.MAX_Vel + 1):
                    for v_j in range(self.raceMap.MIN_Vel, self.raceMap.MAX_Vel + 1):
                        self.policy[i, j, v_i, v_j] = random.choice(raceMap.allActions([i, j, v_i, v_j]))
        
    def do_MC_algoritm(self):

        i = 0
        num_epi = []
        total_steps = []
        print('policy chosed: ', self.params.b_policy)
        for i in range(self.params.episodes):
            b = self.policy

            # Generate an episode using soft policy b
            Ss, As, Rs = self.create_episode(b, self.params.b_policy)
            G = 0
            W = 1
            num_epi.append(i)
            total_steps.append(len(Ss))
            print('running episode:', i)
            for t in range(len(Ss) - 1, -1, -1):
                G = self.params.g * G + Rs[t]
                self.C[tuple(Ss[t] + As[t])] += W
                self.Q[tuple(Ss[t] + As[t])] += (W / self.C[tuple(Ss[t] + As[t])]) * (G - self.Q[tuple(Ss[t] + As[t])])
                self.policy[tuple(Ss[t])] = random.choice([a for a in self.raceMap.allActions(Ss[t]) if self.Q[tuple(Ss[t] + a)] == np.max([self.Q[tuple(Ss[t] + a)] for a in self.raceMap.allActions(Ss[t])])])
                if As[t] != self.policy[tuple(Ss[t])]:
                    break
                W *= 1 / (1 - self.params.e + self.params.e/len(self.raceMap.allActions(Ss[t])))
            if i > self.params.episodes:
                break
        return num_epi, total_steps

    def create_episode(self, policy, p_type, win = False, S_0 =None):

        assert S_0 is not None or not win

        # decreasing epislon
        if p_type == "decay" and self.params.e > self.params.e_MIN and not win:
            self.params.e *= self.params.e_decay 

        # Creat collections of state rewards and actions
        Ss = []
        As = []
        Rs = []

        # initialize the state with random variables
        if not win:
            Ss.append(random.choice([[i, j, 0, 0] for i, j in self.raceMap.get_locations(self.raceMap.START_state)]))
        else:
            Ss.append(S_0)

        # for different policy
        i = 0
        while True:
            if p_type == 'deterministic':
                As.append(policy[tuple(Ss[i])])
            elif np.random.binomial(1, self.params.e) == 1 or p_type == 'random':
                As.append(random.choice(self.raceMap.allActions(Ss[i])))
            else:
                As.append(policy[tuple(Ss[i])])
            state, reward, done = self.raceMap.take_action(Ss[i], As[i], win)

            Rs.append(reward)
            if done:
                break
            else:
                Ss.append(state)
            i += 1
        return Ss, As, Rs
    
    def create_exhibitions(self):

        starts = [[i, j, 0, 0] for i, j in self.raceMap.get_locations(self.raceMap.START_state)]
        random.shuffle(starts)
        for idx, S_0 in enumerate(starts[:self.params.exhibitions]):
            Ss, As, Rs = self.create_episode(self.policy, 'deterministic', win=True, S_0=S_0)

            # do the animation
            fig = plt.figure()
            ims = []
            for state in Ss:
                im = self.raceMap.print_track(state)
                ims.append([im])
            anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)
            plt.show()

class Params:
    def __init__(self):
        # amount of episode
        self.episodes = 5000

        # amount of exhibitions 
        self.exhibitions = 5

        # start q value
        self.init_q = -100000

        # set the discount value
        self.g = 1

        # choose the policy : decay, random, deterministic, e-greedy
        self.b_policy = 'e-greedy'
        
        # set up epsilon value for different b policy 
        if self.b_policy == 'decay' or self.b_policy == 'random':
            self.e = 1
        elif self.b_policy == 'deterministic':
            self.e = 0
        else:
            self.e = 0.05
        # if decay the ratio is needed
        self.e_decay = 0.999
        self.e_MIN = 0.1

def plot(num_epi, total_steps):

        plt.figure(0)
        plt.plot(num_epi, total_steps)
        plt.xlabel('# of episode')
        plt.ylabel('total_steps')

def do_exercise():
    print('start running, it might take a while')

    rmap = raceMap()
    params =  Params()

    OPMC = MonteCarlo(rmap, params)
    ne, ts = OPMC.do_MC_algoritm()
    plot(ne, ts)
    plt.show()
    OPMC.create_exhibitions()


do_exercise()


