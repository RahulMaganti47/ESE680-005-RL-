from mountain_car import *
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal 
import torch.optim as optim

class CustomPolicy(nn.Module):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions

        #128 hidden layers and input features of position and velocity (num_states)
        self.w1 = nn.Linear(self.num_states, 128, bias=False)
        #from hidden layer to output (# of actions)
        self.w2 = nn.Linear(128, self.num_actions, bias=False)   

    # TODO: fill this function in
    # it should take in an environment state 
    # return the action that follows the policy's distribution
    def forward(self, x):    
        model = nn.Sequential(
            self.w1, 
            nn.Dropout(p=0.6), 
            nn.ReLU(), 
            self.w2,
            nn.Softmax(dim=-1)
        ) 
        return model(x) 

    def act(self, state):
        # choose action over distribution from [-1, 1]
        #convert to tensor
        state = torch.from_numpy(state).type(torch.FloatTensor)
        # variable # of inputs
        state = policy(Variable(state))  
    
        #sample from the Normal distribution
        m = dist.Normal()
        action = m.sample() 

        return action


    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy arra
    def compute_gradient(self, state, action, discounted_return):
        
        
        pass


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        pass

policy = CustomPolicy()
optimizer = optim.Adam(policy.parameters(), lr=.01)

# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewardsList, gamma):
    '''
    total_sum = 0
    discounted_returns =[]
    for i in reversed(range(0, len(rewards))): 
        total_sum = gamma*total_sum + rewards[i]
        discounted_returns[i] = total_sum
    
    return discounted_returns
    '''
    #Nikil's vectorized version
    rewards = np.array(rewardsList)
    rewards = rewards.reshape(1,rewards.size)
    to_the_nth = np.arange(rewards.size).reshape(rewards.shape)
    grid = (to_the_nth) + (-1*to_the_nth.transpose())
    gamma_grid = np.power(gamma,grid)
    gamma_grid[grid<0]= 0
    return np.sum(rewards*gamma_grid,axis=1)

# TODO: fill this function in  
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate):

    for i in range(num_episodes):
        rewards = [] 
        states = []
        actions = [] 

        t = 0
        score = 0
        state = env.reset()
        print("Episode # ", i)
        #step through env: run policy through sample trajectory
        while not done: 
            action = policy.act(state)
            next_state, reward, done, _ = env.step(action[0].data)
            rewards.append(reward)
            states.append(next_state)
            actions.append(action)

            score += reward 
            state = next_state

            #prevent over-long, never-ending trials
            if t==999:
                done = True
            
            t += 1
        num_time_steps = t
        print("score: ", score,"\n")
    
        #update network weights
        # take gradient
        # backpropagation steps
                    
    pass 



if __name__ == "__main__":
    gamma = 0.9
    num_episodes = 20000
    learning_rate = 1e-4
    env = Continuous_MountainCarEnv()

    policy = CustomPolicy(2, 1)
    reinforce(env, policy, gamma, num_episodes, learning_rate)

    # gives a sample of what the final policy looks like
    print("Rolling out final policy")   
    state = env.reset()
    env.print_()
    done = False 
    while not done:
        input("press enter to continue:")
        action = policy.act(state)
        state, reward, done, _ = env.step([action])
        env.print_()

