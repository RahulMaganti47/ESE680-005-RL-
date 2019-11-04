from grid_world import *
import numpy as np
import scipy.signal
import gym
import pdb
import matplotlib.pyplot as plt
from mountain_car import *
#N_POS = 15
#N_VEL = 12

num_pos = 15
num_vel = 12
num_act = 3
pos_disc = np.linspace(-1.2, 0.6, num=num_pos+1)
vel_disc = np.linspace(-0.07, 0.07, num=num_vel+1)
act_disc = np.linspace(-1, 1, num=num_act+1)
act_means = (act_disc[1:] + act_disc[:-1]) / 2

#finds discrete index representing a certain continuous state
def find_discrete_state(state):
    discp = sum(state[0] > pos_disc[1:-1])
    discv = sum(state[1] > vel_disc[1:-1])
    return discp * num_vel + discv

def find_discrete_action(action):
    return sum(action > act_disc[1:-1])

class DiscreteSoftmaxPolicy(object):
    def __init__(self):
        self.weights = np.zeros([num_pos*num_vel, num_act])

    def softmax(self, x):    
        return np.exp(x) / np.sum(np.exp(x))

    def get_pi(self, st):
        logits = self.weights[st, : ]
        pi = self.softmax(logits)
        return pi

    # TODO: fill this function in    
    # it should take in an environment stat
    def act(self, state):
        st = find_discrete_state(state)
        pi = self.get_pi(st)
        ac = np.random.choice(num_act, 1, p=pi)[0]
        action = act_means[ac]
        return action, ac
    
    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, advantage):
        st = find_discrete_state(state)
        ac = find_discrete_action(action)

        pi = self.get_pi(st)

        #make onehot array representing current action
        act_onehot = np.zeros(pi.shape)
        act_onehot[ac, ...] = 1.0

        #calculate grad(logpi)
        dsoftmax = pi * (act_onehot - pi)
        dlog = dsoftmax/pi
        w_grad = np.zeros((num_vel*num_pos, num_act))
        w_grad[st, :] = dlog * advantage
        return w_grad

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += step_size * grad

class ValueEstimator(object):
    def __init__(self, num_states,num_actions, learning_rate_b):
        self.num_states = num_states
        self.num_actions = num_actions 
        self.learning_rate_b = learning_rate_b
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))
 
    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        st = find_discrete_state(state)
        return self.values[st]

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self,state,value_estimate,target):
        diff = target - value_estimate
        st = find_discrete_state(state)
        self.values[st] += self.learning_rate_b*diff

        
# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]

def get_discounted_returns(rewards, gamma, time_steps):
    moving_addition = 0
    discounted_returns = np.zeros(time_steps)
    for i in reversed(range(0, len(rewards))):
        moving_addition = moving_addition * gamma + rewards[i] 
        discounted_returns[i] = moving_addition
    return discounted_returns 

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, gamma, num_episodes, learning_rate_p, value_estimator):
    total = []
    total_per_reward = []
    for episode in range(num_episodes):
        rewards = []
        states = []
        actions = []
        ep_reward = 0
        episode_length = 0
        state = env.reset()
        while(True):
            action, act = policy.act(state)
            n_state, reward, done, _= env.step([action])
            states.append(state)
            rewards.append(reward)
            ep_reward += reward
            actions.append(action)
            episode_length += 1
            state = n_state

            if(done or episode_length >= 999):
                break
        total.append("Episode reward: {} ".format(ep_reward))
       
        print(ep_reward)
        if ((episode % 10) == 0):
            total_per_reward.append(ep_reward)
            #plt.plot(episode, ep_reward)

        discounted_returns = get_discounted_returns(rewards, gamma, episode_length)
        grad_v = np.zeros((policy.weights.shape))
        for step in range(episode_length):
            base_val = value_estimator.predict(states[step])
            adv = discounted_returns[step] - base_val
            grad_v += policy.compute_gradient(states[step], actions[step], adv)
            value_estimator.update(states[step],base_val,discounted_returns[step])

        policy.gradient_step(grad_v, learning_rate_p)
    return total


    

if __name__ == "__main__":
    gamma = .999
    env = Continuous_MountainCarEnv()
    num_of_episodes = 2000
    learning_rate_p = 1e-3 
    learning_rate_b = .01

    #num_actions = 3
    #num_states = N_POS * N_VEL #if you wish you can choose a different value.
    policy = DiscreteSoftmaxPolicy()
    num_states =  num_pos * num_vel
    value_estimator = ValueEstimator(num_states, num_act, learning_rate_b)

    for i in range(1):
        total_reward = reinforce(env, policy, gamma, num_of_episodes, learning_rate_p, value_estimator)
        print(total_reward)
    #Test time  
    state = env.reset()
    env.print_()
    done = False
    while not done:
        #input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print_()


