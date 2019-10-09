from mountain_car import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from decimal import Decimal 


class Policy(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Policy, self).__init__() 
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.layer1 = nn.Linear(self.num_states, 128, bias=False)
        self.action_mu = nn.Linear(128, self.num_actions, bias=False) 
             
        #maybe doesn't need to be in the policy
        # Episode policy and reward history 
        self.states_history = []
        self.rewards_this_episode = []
        self.policy_history = Variable(torch.Tensor()) 
        # Overall reward and loss history 
        self.reward_history = []
        self.loss_history = [] 

    def forward(self, x):
        model = torch.nn.Sequential( 
            self.layer1,
            nn.Dropout(p=0.6),
              nn.ReLU(),
            self.action_mu,
            nn.Softmax(dim=-1) 
        ) 
        return model(x) 

def act(state, policy, sigma):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.double)
    action_mean = policy(Variable(state))
    dist = Normal(action_mean, sigma)
    action = dist.sample()
    
    # TODO: Move this to its own function
    # Add log probability of our chosen action to our history    
    if policy.policy_history.size()[0] != 0:
            policy.policy_history = torch.cat([policy.policy_history, dist.log_prob(action)])
    else:
        policy.policy_history = (dist.log_prob(action))
    return action, dist.log_prob(action)
    
def get_discounted_returns(rewardsList, gamma):
    #Nikil's vectorized version
    rewards = np.array(rewardsList)
    rewards = rewards.reshape(1,rewards.size)
    to_the_nth = np.arange(rewards.size).reshape(rewards.shape)
    grid = (to_the_nth) + (-1*to_the_nth.transpose())
    gamma_grid = np.power(gamma,grid)
    gamma_grid[grid<0]= 0
    return np.sum(rewards*gamma_grid,axis=1).reshape(-1,1)
    
def update_policy(discounted_returns, gamma):
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(discounted_returns)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    return loss

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate, sigma):
    for episode in range(num_episodes): 
        done = False 
        state = env.reset()
        t = 0 
        score = 0
        print("Episode # ", episode)

        #generate trajectory for this episode
        while not done:
            a = act(state, policy, sigma)
            next_state, reward, done, _ = env.step(a)
            policy.rewards_this_episode.append(reward)
            policy.states_history.append(next_state)
            score += reward 
            state = next_state

            #prevent over-long, never-ending trials
            if t==999:
                done = True
            
            t += 1

        print("score: ", score,"\n")
        
        discounted_returns = get_discounted_returns(policy.rewards_this_episode, gamma) 
        # convert to tensor
        discounted_returns = torch.FloatTensor(discounted_returns)

        #learn from this episode
        loss = update_policy(discounted_returns, gamma)

        #Save and intialize episode history counters
        policy.loss_history.append(loss.data[0])
        policy.reward_history.append(policy.rewards_this_episode[-1])
        policy.policy_history = Variable(torch.Tensor())
        policy.rewards_this_episode= []

    #does this make sense?
    return policy


if __name__ == "__main__":
    #Hyperparameters
    gamma = .99
    learning_rate = 0.01 
    num_episodes = 10000
    sigma = .1

    policy = Policy(2, 1).double()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    env = Continuous_MountainCarEnv()
    reinforce(env, policy, gamma, num_episodes, learning_rate, sigma)

    # gives a sample of what the final policy looks like
    print("Rolling out final policy")
    state = env.reset()
    env.print_()
    done = False
    while not done:
        #input("press enter to continue:")
        action = act(state)
        state, reward, done, _ = env.step(action)
        env.print_()
