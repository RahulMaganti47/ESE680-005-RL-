import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import gym
from collections import deque
import numpy as np 
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch.autograd import Variable 


class Policy(nn.Module): 
    def __init__(self, s_size=6, h_size=50, a_size=3):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(s_size, h_size)
        self.l2 = nn.Linear(h_size, a_size)

        self.model = nn.Sequential(
            self.l1, 
            nn.ReLU(), 
            self.l2, 
            nn.Softmax(dim=1)
        )
         
    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item() - 1, m.log_prob(action)

def compute_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    moving_add = 0
    for i in reversed(range(0, len(rewards))):
        moving_add = moving_add*gamma + rewards[i]
        discounted_rewards[i] = moving_add

    return discounted_rewards

class Critic(nn.Module): 
    def __init__(self, state_dim=6, hidden_dim=20, output_dim=1, lambd=.9):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, output_dim, bias=False) 
        self.lambd = lambd 
 
    def forward(self, state): 
        state = torch.from_numpy(state).float().unsqueeze(0) 
        x = F.relu(self.l1(state)) 
        x = self.l2(x) 
        return F.softmax(x, dim=1)

def train_select_torque(n_episodes, policy, print_every=1): 
    total_rewards = [] 
    max_iters = 500 
    for ep in range(n_episodes): 
        _ = env.reset()
        score = 0 
        for _ in range(max_iters): 
            action = np.random.choice([-1, 1], 1, p=[.5, .5])[0]
            (_, reward, _, _) = env.step(action)
            score += reward 

        total_rewards.append(score) 
        print("Episode: {}, Score: {}".format(ep, score)) 
    
    return total_rewards

def train(n_episodes, policy, critic, gamma, print_every=4):
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    #optimizer_v = optim.Adam(critic.parameters(), lr=.05)
    scores_deque = deque()
    
    total_rewards = [] 
    for ep in range(n_episodes):  
        traj_log_probs = [] 
        rewards = []
        values = [] 
        state = env.reset()
        score = 0 
        done = False
        
        while not done: 
            action, log_prob = policy.act(state) 
            value = critic.forward(state) 
            values.append(value) 
            traj_log_probs.append(log_prob)
            next_state, reward, done, _ = env.step(action) 
            score += reward 
            rewards.append(reward) 

            state = next_state 

        scores_deque.append(score)  
        total_rewards.append(score)
            
        disc_rewards = compute_rewards(rewards, gamma)
        disc_rewards = torch.tensor([disc_rewards]).float()  
    
        values = torch.cat(values) 

        # advantage: discounted_rewards - function approximated values 
        advantage = disc_rewards - values
    
        policy_loss = []  
        for log_prob in traj_log_probs:
            policy_loss.append(-log_prob * advantage) 
         
        policy_loss = torch.cat(policy_loss).sum() 
    
        
        #value_loss = Variable(value_loss, requires_grad = True)

        # MSE loss 
        value_loss_1 = advantage.pow(2).mean()
        value_loss_1 = Variable(value_loss_1, requires_grad = True)
        #create one loss function for actor and critic
        total_loss = torch.stack([policy_loss, value_loss_1]).sum()
       
        # zero gradients from previous iteration 
        optimizer.zero_grad()
        total_loss.backward() 
        optimizer.step()

        if ep % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_deque)))        
    
    return scores_deque

    def load(self):
        with open('w.pkl', 'rb') as f:
            self.w = pickle.load(f) 
        
        with open('thetas.pkl', 'rb') as f:
            self.thetas = pickle.load(f)
            
if __name__ == "__main__":

    env = gym.make('Acrobot-v1')
    seed = 1
    env.seed(seed) 
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    num_episodes = 1000  
    gamma = .99 
    policy = Policy()
    critic = Critic() 
    total_rewards = train(num_episodes, policy, critic, gamma) 
    #total_rewards = train_select_torque(num_episodes, policy)

    # Plotting  
    plt.plot(range(len(total_rewards)), total_rewards, 'b-', label="training curve")
    plt.axhline(y=max(total_rewards), color='r', linestyle='-', label="maximum average reward") 
    plt.legend(loc="upper left")  
    plt.title('Episodes vs. Cumulative Rewards')
    plt.xlabel('Episode #')
    plt.ylabel('Cumulative Reward')
    plt.show()

