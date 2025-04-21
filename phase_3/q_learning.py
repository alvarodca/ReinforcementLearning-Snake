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
    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.8, epsilon=0.05, epsilon_min=0, epsilon_decay=1):  # epsilon_min=0 for testing
        # Best values after hyperparameter tuning seem to be alpha = 0.1 and gamma = 0.9
        # To see if training is being done right, epsilon = 0
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
        else:
            action = np.argmax(self.q_table[state])  # Exploit
            
            
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        return action
    
    def save_hyperparams(self, episode_number, total_reward, filename = "hyperparams.txt"):
        """Stores hyperparameters after each run"""
        with open(filename, "a") as f:
            f.write(f"{episode_number}\t{self.alpha}\t{self.gamma}\t{self.epsilon}\t{total_reward}\n")

    def encode_state3(self, state):
        """
        Encodes the state tuple (food_state, danger) into an integer index,
        according to the following scheme:
        
        -- Food_state --
        There are 8 possibilities:
        "UP" -> 0, "DOWN" -> 1, "LEFT" -> 2, "RIGHT" -> 3 
        ("LEFT", "UP") -> 4, ("RIGHT", "UP") -> 5,
        ("LEFT", "DOWN") -> 6, ("RIGHT", "DOWN") -> 7.
        
        -- Danger --
        Danger is represented as a 3-tuple: (forced, add1, add2).
        - forced is the opposite of the current direction.
        - add1 and add2 are additional danger directions.
        
        We have three cases:
        1. Only one danger provided: (f, "none", "none"):
            4 possibilities.
        2. Two dangers provided: (f, a, "none") with a != f.
            For each forced (4 possibilities) there are 3 possible a values, totaling 12.
            We encode these with an offset of 4.
        3. Three dangers provided: (f, a, b) with f, a, b all distinct.
            For each forced there are 3×2 = 6 possibilities, totaling 24.
            We encode these with an offset of 16 (i.e. 4+12).
        
        The final danger code is in range 0–39. The overall state index is:
            state_index = food_index * 40 + danger_code,
        giving a total state space of 8 * 40 = 320.
        """
        food_state, danger = state

        # --- Encode food_state ---
        simple_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        if isinstance(food_state, str):
            food_index = simple_map[food_state]
        else:
            if food_state == ("LEFT", "UP"):
                food_index = 4
            elif food_state == ("RIGHT", "UP"):
                food_index = 5
            elif food_state == ("LEFT", "DOWN"):
                food_index = 6
            elif food_state == ("RIGHT", "DOWN"):
                food_index = 7

        # --- Process danger ---
        # Ensure danger is a 3-tuple.
        if len(danger) == 1:
            danger = (danger[0], "none", "none")
        elif len(danger) == 2:
            danger = (danger[0], danger[1], "none")
        # Now danger should be (forced, add1, add2).
        forced, add1, add2 = danger

        # Basic mapping for forced.
        forced_map = {"top": 0, "bottom": 1, "left": 2, "right": 3}
        f_idx = forced_map.get(forced, 0)

        # Determine which scenario we are in:
        # 1-danger case
        if add1 == "none":
            danger_code = f_idx  # values 0-3

        # 2-dangers case.
        elif add1 != "none" and add2 == "none":
            # Candidate list for add1: all directions except forced, in fixed order.
            candidates = [d for d in ["top", "bottom", "left", "right"] if d != forced]
            # There are 3 candidates. Find index of add1.
            try:
                pos = candidates.index(add1)
            except ValueError:
                pos = 0
            danger_code = 4 + (f_idx * 3) + pos  # offset 4, total range: 4 to 15

        # 3-dangers case: (f, a, b) with all distinct.
        else:
            candidates = [d for d in ["top", "bottom", "left", "right"] if d != forced]
            try:
                pos1 = candidates.index(add1)
            except ValueError:
                pos1 = 0
            # For the third danger, the candidates are those in "candidates" excluding a.
            candidates2 = [d for d in candidates if d != add1]
            try:
                pos2 = candidates2.index(add2)
            except ValueError:
                pos2 = 0
            danger_code = 16 + (f_idx * 6) + (pos1 * 2) + pos2  # offset = 16; range: 16 to 39

        state_index = food_index * 40 + danger_code
        return state_index

    def update_q_table(self, state, action, reward, next_state):
        # Your code here
        # Update the current Q-value using the Q-learning formula
        # if terminal_state:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        # else:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        enc_state = self.encode_state3(state)
        enc_next_state = self.encode_state3(next_state)

        # Our Q-Value
        current_q = self.q_table[enc_state][action]

        # Terminal state if  snake dies
        if  reward == -75:
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
