"""
Snake Eater Q learning basic algorithm
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
import numpy as np
import random
import json
import time

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0, gamma=0.9, epsilon=0, epsilon_min=0.01, epsilon_decay=0.999):
        # Best values after hyperparameter tuning seem to be alpha = 0.2 and gamma = 0.9
        # With alpha and epsilon = 0 we can see how well training is being done
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.load_q_table()

    def choose_action(self, state, allowed_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(allowed_actions)  # Explore
            #print("Explore", action)
        else:
            #print("state action", state)
            action = np.argmax(self.q_table[state])  # Exploit
            print("Exploit", action)
            #print(self.q_table[state])
            
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        return action
    
    def save_hyperparams(self, episode_number, total_reward, filename = "hyperparams.txt"):
        """Stores hyperparameters after each run"""
        with open(filename, "a") as f:
            f.write(f"{episode_number}\t{self.alpha}\t{self.gamma}\t{self.epsilon}\t{total_reward}\n")


    def encode_state(self, state):
        """Encode state to obtain an integer"""
        
        simple_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        if isinstance(state, str):
            return simple_map[state]
        else:
            direction_y = {"UP": 0, "DOWN": 1}
            direction_x = {"LEFT": 0, "RIGHT": 1}
            # Unpack the tuple. (hor, ver) in our case.
            hor, ver = state
            # Offset combined states by 4
            return 4 + direction_y[ver] * 2 + direction_x[hor]


    def update_q_table(self, state, action, reward, next_state):
        # Your code here
        # Update the current Q-value using the Q-learning formula
        # if terminal_state:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        # else:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        enc_state = self.encode_state(state)
        enc_next_state = self.encode_state(next_state)

        # Our Q value
        current_q = self.q_table[enc_state][action]

        # Terminal state if  snake dies
        if  reward == -10:
            new_q = (1-self.alpha)*current_q + self.alpha*reward

        # Non-terminal state
        else:
            new_q = (1-self.alpha)*current_q + self.alpha*(reward+self.gamma*np.max(self.q_table[enc_next_state]))

        # Write back updated Q-value into the q_table
        self.q_table[enc_state][action] = new_q



    def save_q_table(self, filename="qtable.txt"):
        np.savetxt(filename, self.q_table)

    def load_q_table(self, filename="qtable.txt"):
        try:
            self.q_table = np.loadtxt(filename)
        except IOError:
            # If the file doesn't exist, initialize Q-table with zeros as per dimensions
            self.q_table = np.zeros((self.n_states, self.n_actions))
