import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from pendulum_v2 import PendulumEnv
import time

class ActorCritic(object):

    def __init__(self, env, gamma=0.99, sigma=0.1, alpha_value=1e-2, alpha_policy=1e-2):
        # Upper and lower limits of the state 
        self.min_state = env.min_state
        self.max_state = env.max_state

        #RBF Function Approximator
        self.num_rbf_centers = 5j
        self.rbf_centers = np.mgrid[0:1:self.num_rbf_centers, 0:1.5:self.num_rbf_centers, 0:1:self.num_rbf_centers].reshape(3, -1).T
        self.w = np.zeros((self.rbf_centers.shape[0]))
        self.rbf_sigma = 0.1

        #Gaussian Policy
        self.thetas = np.zeros((self.rbf_centers.shape[0]))

        # Discount factor (don't tune)
        self.gamma = gamma

        # Standard deviation of the policy (need to tune)
        self.pi_sigma = sigma

        # Step sizes for the value function and policy
        self.alpha_value = alpha_value
        self.alpha_policy = alpha_policy
        # These need to be tuned separately

    def load(self):
        with open('w.pkl', 'rb') as f:
            self.w = pickle.load(f)
        
        with open('thetas.pkl', 'rb') as f:
            self.thetas = pickle.load(f)
    """
    Normalizes State to [0, 1] range for each feature
    """
    def normalize_state(self,  state):
        norm_state = ((state - self.min_state)/(self.max_state - self.min_state))
        return norm_state

    """
    args: state is Normalized [theta, vel]
    returns: real valued feature vector R^(num_rbf_centers**2)
    """
    def get_rbf_feature(self, state):
        out = np.exp(-np.square(np.linalg.norm(self.rbf_centers - state, axis=1))/(2.*self.rbf_sigma**2))
        return out
    
    """
    args: state is un-normalized [theta, val]
    returns: Value @ this state
    """
    def get_value(self, norm_state):
        #1) Normalize State & Get RBF Features x = (x_1(s), x_2(s)...)
        x = self.get_rbf_feature(norm_state)

        #2) Use RBF Features with Weights to get value
        value = np.squeeze(np.dot(x, self.w))
        return value

    def act(self, norm_state):
        #1) Normalize State & Get RBF Features
        x = self.get_rbf_feature(norm_state)

        #2) Use RBF Features with Theta to get policy mean
        mu = np.squeeze(np.dot(x, self.thetas))

        #3) Use mean and sigma to sample from gaussian
        action = np.random.normal(mu, self.pi_sigma)
        return action

    def update(self, norm_state, action, reward, norm_next_state, done, I):
        delta = reward + self.gamma * (1 - done) * self.get_value(norm_next_state) - self.get_value(norm_state)
        x = self.get_rbf_feature(norm_state)

        #Compute Gradients
        grad_v = x
        mu = np.squeeze(np.dot(x, self.thetas))
        grad_logpi = (1/self.pi_sigma**2) * (action - mu) * x

        #Update
        self.w = self.w + self.alpha_value * I * delta * grad_v
        self.thetas = self.thetas + self.alpha_policy * I * delta *  grad_logpi

def train(env, model):
    num_episodes = 10000
    MAX_EPISODE_LENGTH = 1000
    total_returns = []
    max_episode_returns = -1000000
    for i in range(num_episodes):
        done = False
        episode_length = 0
        episode_returns = 0
        sum_episode_returns = 0
        state = env.reset()
        norm_state = model.normalize_state(state)
        I = 1
        while not done:
            #1) Take an Action & Step
            action = model.act(norm_state)
            next_state, reward, done, _ = env.step([action])
            norm_next_state = model.normalize_state(next_state)

            #2) Update
            model.update(norm_state, action, reward, norm_next_state, done, I)
            I*= model.gamma
            norm_state = norm_next_state

            #3) Log & Render
            # env.render()

            episode_returns = episode_returns * 0.9 + reward * 0.1
            sum_episode_returns+= reward
            # if episode_length % 100 == 0:
            #     print(f"EPISODE:{i}, STEP:{episode_length}, RETURN:{episode_returns} CURR_ACTION:{action}")
            episode_length+=1
        total_returns.append(sum_episode_returns)

        if sum_episode_returns > max_episode_returns:
            with open('thetas.pkl', 'wb') as f:
                pickle.dump(model.thetas, f)

            with open('w.pkl', 'wb') as f:
                pickle.dump(model.w, f) 
            max_episode_returns = sum_episode_returns

        if i > 0 and i % 100 == 0:
            N = 100
            PAST100 = np.convolve(total_returns[0:i], np.ones(N)/N, mode='valid')
            plt.figure(2)
            plt.ion()
            plt.cla()
            plt.plot(range(PAST100.size), PAST100)
            plt.pause(0.1)
            plt.show()

    return total_returns


if __name__ == "__main__":
    env = PendulumEnv()
    policy = ActorCritic(env, sigma=0.5, alpha_value=5e-2, alpha_policy=1e-2)
    total_returns = train(env, policy)

    #Test Policy
    policy.load()
    state = env.reset()
    norm_state = policy.normalize_state(state)
    done = False
    while not done:
        act = policy.act(norm_state)
        next_state, _, done, _ = env.step([act])
        env.render()