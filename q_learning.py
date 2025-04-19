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
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995):
        # Best values after hyperparameter tuning seem to be alpha = 0.2 and gamma = 0.9
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
        
        """simple_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        if isinstance(state, str):
            return simple_map[state]
        else:
            direction_y = {"UP": 0, "DOWN": 1}
            direction_x = {"LEFT": 0, "RIGHT": 1}
            # Unpack the tuple. (hor, ver) in our case.
            hor, ver = state
            # Offset combined states by 4
            return 4 + direction_y[ver] * 2 + direction_x[hor]
"""
    

        border, food_state = state

        # For border "none" use full mapping (8 outcomes):
        if border == "none":
            simple_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
            if isinstance(food_state, str):
                row = simple_map[food_state]
            else:
                direction_y = {"UP": 0, "DOWN": 1}
                direction_x = {"LEFT": 0, "RIGHT": 1}
                hor, ver = food_state
                row = 4 + (direction_y[ver] * 2 + direction_x[hor])

        elif border == "top":
            simple_map = {"DOWN": 0, "LEFT": 1, "RIGHT": 2}
            if isinstance(food_state, str):
                row = 8 + simple_map[food_state]
            else:
                direction_y = {"DOWN": 0}
                direction_x = {"LEFT": 0, "RIGHT": 1}
                hor, ver = food_state
                row = 11 + (direction_y[ver] + direction_x[hor])

        elif border == "bottom":
            simple_map = {"UP": 0, "LEFT": 1, "RIGHT": 2}
            if isinstance(food_state, str):
                row = 13 + simple_map[food_state]
            else:
                direction_y = {"UP": 0}
                direction_x = {"LEFT": 0, "RIGHT": 1}
                hor, ver = food_state
                row = 16 + (direction_y[ver] + direction_x[hor])
        
        elif border == "left":
            simple_map = {"UP": 0, "DOWN": 1, "RIGHT": 2}
            if isinstance(food_state, str):
                row = 18 + simple_map[food_state]
            else:
                direction_y = {"UP": 0, "DOWN": 1}
                direction_x = {"RIGHT": 0}
                hor, ver = food_state
                row = 21 + (direction_y[ver] + direction_x[hor])

        elif border == "right":
            simple_map = {"UP": 0, "DOWN": 1, "LEFT": 2}
            if isinstance(food_state, str):
                row = 23 + simple_map[food_state]
            else:
                direction_y = {"UP": 0, "DOWN": 1}
                direction_x = {"LEFT": 0}
                hor, ver = food_state
                row = 26 + (direction_y[ver] + direction_x[hor])

        return row
    


    def encode_state2(self, state):
        """
        Encodes the state tuple (food_state, danger) into an integer index.
        
        food_state is one of eight positions:
            For simple (aligned) positions:
                "UP"    -> 0, "DOWN"  -> 1, "LEFT"  -> 2, "RIGHT" -> 3
            And for diagonal (combined) positions:
                ("LEFT", "UP")    -> 4, ("RIGHT", "UP")    -> 5,
                ("LEFT", "DOWN")  -> 6, ("RIGHT", "DOWN")  -> 7

        danger is now a tuple of two strings:
            - The first element (forced) is always the blocked direction opposite to the snake's current movement.
            - The second element is either another immediate danger (if detected) or "none".
        We then map these danger combinations as follows:
        
            ("top", "none")   ->  0,    ("top", "bottom") ->  1,
            ("top", "left")   ->  2,    ("top", "right")  ->  3,
            
            ("bottom", "none")->  4,    ("bottom", "top") ->  5,
            ("bottom", "left")->  6,    ("bottom", "right")->  7,
            
            ("left", "none")  ->  8,    ("left", "top")   ->  9,
            ("left", "bottom")-> 10,    ("left", "right") -> 11,
            
            ("right", "none") -> 12,    ("right", "top")  -> 13,
            ("right", "bottom")-> 14,    ("right", "left") -> 15.
            
        Finally, the overall state index is computed as:
            state_index = food_state_index * 16 + danger_code
        Total state space: 8 * 16 = 128.
        """
        food_state, danger = state

        # Encode food_state
        simple_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        if isinstance(food_state, str):
            food_index = simple_map[food_state]
        else:
            # Diagonal positions mapping:
            if food_state == ("LEFT", "UP"):
                food_index = 4
            elif food_state == ("RIGHT", "UP"):
                food_index = 5
            elif food_state == ("LEFT", "DOWN"):
                food_index = 6
            elif food_state == ("RIGHT", "DOWN"):
                food_index = 7

        # Ensure danger is a 2-tuple.
        # If only one element is present, set second element as "none".
        if len(danger) == 1:
            danger = (danger[0], "none")
        
        # Define a manual mapping for the danger tuple:
        danger_map = {
            ("top", "none"): 0, ("top", "bottom"): 1, ("top", "left"): 2, ("top", "right"): 3,
            ("bottom", "none"): 4, ("bottom", "top"): 5, ("bottom", "left"): 6, ("bottom", "right"): 7,
            ("left", "none"): 8, ("left", "top"): 9, ("left", "bottom"): 10, ("left", "right"): 11,
            ("right", "none"): 12, ("right", "top"): 13, ("right", "bottom"): 14, ("right", "left"): 15,
        }
        danger_code = danger_map.get(danger, 0)
        
        state_index = food_index * 16 + danger_code
        return state_index
    

    def update_q_table(self, state, action, reward, next_state):
        # Your code here
        # Update the current Q-value using the Q-learning formula
        # if terminal_state:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        # else:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        enc_state = self.encode_state2(state)
        enc_next_state = self.encode_state2(next_state)

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
