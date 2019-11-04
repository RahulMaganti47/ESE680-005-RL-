from grid_world import *
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt



class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.zeros((self.num_states, self.num_actions))

    def r_softmax(self, x): 
        f = np.exp(x - np.max(x))
        return f / f.sum(axis=0)
    
    def get_logits(self, state):
        logits = self.r_softmax(self.weights[state, :])
        return logits
        
    def act(self, state):
        actions_a = self.get_logits(state)
        #where_nan = isnan(actions_a)
        actions = [0, 1, 2, 3]
        action = np.random.choice(actions, 1, p=actions_a)[0]
        return action
    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return,advantage):

        grad = np.zeros((self.num_states, self.num_actions))
        possible_actions = self.get_logits(state)
        one_hot = np.zeros(possible_actions.shape)
        one_hot[action] = 1.0
        d_sa = one_hot - possible_actions
        grad[state, :] = d_sa * advantage 
       
        return grad 

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += grad*step_size

class ValueEstimator(object):
    def __init__(self, num_states,num_actions):
        self.num_states = num_states
        self.num_actions = num_actions 
        
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))

    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        return self.values[state]

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self,state,value_estimate,target,value_step_size):
        difference = target - value_estimate
        #squared_difference = np.square(difference) 
        #N = squared_difference.size
        #loss = np.sum(squared_difference)  
        #mean_loss = (1/N) * loss
        self.values[state] += value_step_size * difference


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
def reinforce(env, policy,value_estimator, gamma, alpha_p, alpha_b, num_episodes):
    scores = [] 
    for i in range(num_episodes): 
        state = env.reset()
        states = []
        rewards = [] 
        actions = [] 
        time_steps = 0 
        done = False
        score = 0
        while not done:
            action = policy.act(state) 
            next_state, reward, done = env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            time_steps = time_steps + 1
            score += reward
            state = next_state

            #prevent infinite episodes 
            #if (time_steps==999): 
              #  done = True 
        
        print("Episode: {}, Score: {} ", i, score)
        scores.append(scores)
    
        discounted_returns = get_discounted_returns(rewards, gamma, time_steps)
        grad_p = np.zeros(policy.weights.shape)

        for t in range(time_steps): 
            # with baseline
            baseline_val = value_estimator.predict(states[t])
            advantage_func = discounted_returns[t] - baseline_val
            grad_p += policy.compute_gradient(states[t], actions[t], discounted_returns[t], advantage_func) 
            value_estimator.update(states[t], baseline_val, discounted_returns[t], alpha_b)

        policy.gradient_step(grad_p, alpha_p)

    #plt.plot(scores)

if __name__ == "__main__":
    env = GridWorld(MAP2)
    env.print_()
    gamma = .97
    num_episodes = 1000
    learning_rate_p = .01
    # learning rate for baselin e must be << learning rate for policy 
    learning_rate_b = .1
    #temperature must be tuned.
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions()) 
    value_estimator = ValueEstimator(env.get_num_states(), env.get_num_actions())
    reinforce(env, policy,value_estimator, gamma, learning_rate_p, learning_rate_b, num_episodes)

    #Test time
    state = env.reset()
    env.print_()
    done = False
    while not done:
        #input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print_()


