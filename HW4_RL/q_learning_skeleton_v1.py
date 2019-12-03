import numpy as np
from grid_world import *
import matplotlib.pyplot as plt
import random  
# TODO: Fill this function in
# Function that takes an a 2d numpy array Q (num_states by num_actions)
# an epsilon in the range [0, 1] and the state
# to output actions according to an Epsilon-Greedy policy
# (random actions are chosen with epsilon probability)

class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, eps, learning_rate=0.5, discount_factor=.999):
         # initialize Q values to something
        self.num_states = num_states
        self.num_actions = num_actions  
        
        #initilaize the q_table                                                                                                                                                         
        self.q_table = np.random.random_sample([self.num_states, self.num_actions])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps 
    
    def tabular_epsilon_greedy_policy(self, eps, state):

        actions = self.q_table[state]
        max_action = np.argmax(actions) 
        num_actions = len(actions)
        rand_action = np.random.choice(num_actions)
        temp_rand = random.uniform(0, 1)
        # choose random action with probability epsilon  
        if (eps > temp_rand):  
            return max_action 
        else:
             return rand_action

    def greedy_policy(self, state): 
        actions = self.q_table[state]
        max_action = np.argmax(actions)
        return max_action 

    # TODO: fill in this function
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        #get the q_values for the next state
        q_next = self.q_table[next_state]
        #set the q_next values to 0 if we terminate the episode 
        q_next = np.zeros([self.num_actions]) if done else q_next 
        
         #update equation: reward + gamma*max(q_next)
        q_target = reward + self.discount_factor*np.max(q_next)

        # perform the bootstrapped udpate
        self.q_table[state, action] += self.learning_rate*(q_target - self.q_table[state, action])

        # apply exponential decay to epsilon
        # (we get greedier as the policy goes on)
        if done:
            self.eps = self.eps * .99

                
        return self.q_table 

# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, eps, niter=1000): 
    
    num_times_reached = 0
    cum_rewards = [] 
    for ep in range(niter):
        state = env.reset()
        done = False
        score = 0 
        while not done: 
            action = qlearning.greedy_policy(state)
            (next_state, reward, done) = env.step(action) 
            state = next_state
            score += reward 

        cum_rewards.append(score)
        print("Episode: {}, Score: {}".format(ep, score))
        
        if (score > 0): num_times_reached +=1 
    

    print("numerator: {}".format(num_times_reached))
    frac = float(num_times_reached) / float(niter)
    return cum_rewards, frac  
 
def plotting_func(cum_rewards, title, x, y): 
    plt.plot(range(len(cum_rewards)), cum_rewards, "-b", label="training curve")
    plt.xlabel(str(x)) 
    plt.ylabel(str(y))
    plt.title(str(title)) 
    #plt.show()

def discounted_returns(returns, gamma):
    discounts = [gamma**i for i in range(len(rewards)+1)]
    R = sum([a*b for a,b in zip(discounts, rewards)])
    return R 

def print_mat(episode_num): 
    rows = env.n_rows
    cols = env.n_cols
    grid = np.zeros((rows, cols)) 
    n = env.get_num_states()  
  
    for m in range(n): 
        q_vals = q_table[m]  
        x = int(m/cols)  
        y = int(m%cols) 
        grid[x,y] = max(q_vals) 
  
    print("Grid containing vals: {}".format(grid))
    plt.title("Q-Values for MAP after {} episodes".format(episode_num)) 
    plt.imshow(grid)
    plt.show() 

if __name__ == "__main__":
    env = GridWorld(MAP2)
    eps = 1.0 
    seed = 3
    np.random.seed(seed)
    qlearning = QLearning(env.get_num_states(), env.get_num_actions(), eps)
 
    ## TODO: write training code here
    num_episodes = 1000 
    num_states = env.get_num_states()  
    total_reward = 0 
    cum_rewards = []    
    q_vals_start = [] 
    #q_max = np.zeros((num_states))
    
    num_times_reached = 0   
    q_table_start_action1 = []
    q_table_start_action2 = []
    q_table_start_action3 = []
    q_table_start_action4 = []
    for ep in range(num_episodes):    
        state = env.reset() 
        start_state = state
        score = 0 
        states = [] 
        rewards = [] 
        
        
        done = False   
        while not done:
            action = qlearning.tabular_epsilon_greedy_policy(eps, state)    
            (next_state, reward, done) = env.step(action)
            q_table = qlearning.update(state, action, reward, next_state, done)
            states.append(next_state) 
            rewards.append(reward) 
            state = next_state 
            score += reward
        
        if (score > 0): num_times_reached += 1 
        
        cum_rewards.append(score)
        total_reward += score 

        q_table_start_action1.append(q_table[0, 0])
        q_table_start_action2.append(q_table[0, 1]) 
        q_table_start_action3.append(q_table[0, 2]) 
        q_table_start_action4.append(q_table[0, 3]) 

    
        # 3) Matrix size of map containing the q values for each state 
        if (ep == 10):  
            print_mat(ep)  
        if (ep == 100): 
            print_mat(ep)  
        if (ep == 500): 
            print_mat(ep) 
        if (ep == 999): 
            print_mat(ep)

        print("Episode: {}, Score: {}".format(ep, score)) 
    
        # save q_table before convergence
        if ep is 10: 
            q_table_saved = q_table 
        
   
     
    ### ____ OUTPUTS ####

    
    ## plotting stuff    
    # 1) Q Learning on MAP3 
    max_rewards = np.argmax(cum_rewards)
    print(max_rewards) 
    plotting_func(cum_rewards, "episodes vs. cumulative rewards", "episodes" , "cumulative rewards") 
    plt.axhline(y=max(cum_rewards), color='r', linestyle='-', label="Max Cum. Reward")  
    plt.legend(loc="upper left") 
    plt.show() 
    plotting_func(q_vals_start, "episodes vs. q_vals", "episodes", "q val start") 
    plt.show() 

    
     # evaluate the greedy policy to see how well it performs
    print("Evaluate the epsilon-greedy policy only")
    total_rewards, frac = evaluate_greedy_policy(qlearning, env, eps)
    print("Finding goal: {} percent of the time".format(frac))
    plotting_func(total_rewards, "episodes vs. cumulative rewards (greedy policy)", "episodes", "greedy policy rewards")
    plt.show() 


    # 2 a) Plot of q-values for start state
    plt.plot(range(len(q_table_start_action1)), q_table_start_action1, "-b", label="action 1")
    plt.plot(range(len(q_table_start_action2)), q_table_start_action2, "-g", label="action 2")
    plt.plot(range(len(q_table_start_action3)), q_table_start_action3, "-y", label= "action 3")
    plt.plot(range(len(q_table_start_action4)), q_table_start_action4, "-r", label="action 4") 
    plt.title("Episode vs. Start state Q-Table values")
    plt.xlabel("Episode")
    plt.ylabel("Q-Table Values") 
    plt.legend(loc="upper left")  
    plt.show() 

    # 2b) Q Table before and after convergence s
    print("Q_table (before convergence): {}".format(q_table_saved))    
    print("Q_table (after convergence): {}".format(q_table))
    print(num_times_reached) 
    frac = float(num_times_reached) / float(num_episodes) 
    print("Frac: {}".format(frac))
    print("Total Reward: {}".format(total_reward))  
    
    
