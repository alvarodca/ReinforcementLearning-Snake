"""
Snake Eater Environment
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
import numpy as np
import random

class SnakeGameEnv:
    def __init__(self, frame_size_x=150, frame_size_y=150, growing_body=True):
        # Initializes the environment with default values
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.growing_body = growing_body
        self.reset()

    def reset(self):
        # Resets the environment with default values
        self.snake_pos = [50, 50]
        self.snake_body = [[50, 50], [60, 50], [70, 50]]
        
        # Generate food position with border appearance chance
        while True:
            if random.random() < 0.25:
                # 10% chance: choose a border randomly
                border = random.choice(["top", "bottom", "left", "right"])
                if border == "top":
                    x = random.randrange(0, self.frame_size_x, 10)
                    y = 0
                elif border == "bottom":
                    x = random.randrange(0, self.frame_size_x, 10)
                    y = self.frame_size_y - 10
                elif border == "left":
                    x = 0
                    y = random.randrange(0, self.frame_size_y, 10)
                elif border == "right":
                    x = self.frame_size_x - 10
                    y = random.randrange(0, self.frame_size_y, 10)
            else:
                # Otherwise, appear anywhere on the grid
                x = random.randrange(0, self.frame_size_x, 10)
                y = random.randrange(0, self.frame_size_y, 10)
                
            food_pos = [x, y]
            if food_pos not in self.snake_body:
                break

        self.food_pos = food_pos
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        self.reward = 0 # Initialize the starting reward
        return self.get_state()
    

    def step(self, action):
        # Implements the logic to change the snake's direction based on action
        # Update the snake's head position based on the direction
        # Check for collision with food, walls, or self
        # Update the score and reset food as necessary
        # Determine if the game is over

        previous_distance = abs(self.food_pos[0] - self.snake_body[0][0]) + \
                            abs(self.food_pos[1] - self.snake_body[0][1])
        
        self.update_snake_position(action)
        reward = self.calculate_reward(previous_distance)
        self.update_food_position()
        state = self.get_state()
        self.game_over = self.check_game_over()
        return state, reward, self.game_over

    def direction_to_food(self):
        """Obtains the direction from the head of the snake to the food"""

        # Calculating the  distance to the food
        distance_to_food_x = self.food_pos[0] - self.snake_body[0][0]
        distance_to_food_y = self.food_pos[1] - self.snake_body[0][1]

        # Calculating the direction with respect to the food
        if abs(distance_to_food_x) < abs(distance_to_food_y): # Vertical difference is higher
            if distance_to_food_y > 0:
                direction = "UP"
            else:
                direction = "DOWN"

        else:
            if distance_to_food_x > 0:
                direction = "RIGHT"
            else:
                direction = "LEFT"

        return direction
    
    def distance_to_food(self):
        """Obtains the relative distance from the head of the snake to the food, discretizing it in terms of closeness"""
        
        # Calculating the  distance to the food
        distance_to_food_x = self.food_pos[0] - self.snake_body[0][0]
        distance_to_food_y = self.food_pos[1] - self.snake_body[0][1]
        
        # Calculating the distance
        distance = abs(distance_to_food_x) + abs(distance_to_food_y)

        # Total size of the board
        board_size = self.frame_size_x + self.frame_size_y

        # Normalizing the distance
        normalized_distance = distance/board_size

        # Returning nominal values in terms of the distance
        if normalized_distance <= 0.33:
            rel_distance = "Close"
        elif normalized_distance >= 0.66:
            rel_distance = "Far"
        else:
            rel_distance = "Medium"

        return rel_distance
    

    def get_state(self):
        """Obtaining the current state of the game. Our snake currently has 28 different states. These are the combination
        of the direction to the food along with the closest direction"""
        
        head_x, head_y = self.snake_body[0]
        # Determine border state (cell size assumed as 10)
        if head_y == 0:
            border = "top"
        elif head_y >= self.frame_size_y - 10:
            border = "bottom"
        elif head_x == 0:
            border = "left"
        elif head_x >= self.frame_size_x - 10:
            border = "right"
        else:
            border = "none"

        food_x, food_y = self.food_pos

        # Determine food relation:
        if food_x == head_x:
            # aligned vertically → simple vertical state
            food_state = "UP" if food_y < head_y else "DOWN"
        elif food_y == head_y:
            # aligned horizontally → simple horizontal state
            food_state = "LEFT" if food_x < head_x else "RIGHT"
        else:
            # combined state
            hor = "LEFT" if food_x < head_x else "RIGHT"
            ver = "UP" if food_y < head_y else "DOWN"
            food_state = (hor, ver)

        
        return (border, food_state)



    def get_body(self):
    	return self.snake_body
    
    def get_food(self):
    	return self.food_pos
    
    def distance_to_border(self):
        """Calculates the distance from the snake's head to the closest border (in pixels)."""
        # Distance from left border
        d_left = self.snake_pos[0]
        # Distance from right border (considering a cell size of 10)
        d_right = self.frame_size_x - self.snake_pos[0] - 10
        # Distance from top border
        d_top = self.snake_pos[1]
        # Distance from bottom border
        d_bottom = self.frame_size_y - self.snake_pos[1] - 10
        
        return min(d_left, d_right, d_top, d_bottom)

    def calculate_reward(self, previous_distance):
        """Calculates the reward of the snake"""

        # Calculate the current Manhattan distance to food
        current_distance = abs(self.food_pos[0] - self.snake_body[0][0]) + \
                        abs(self.food_pos[1] - self.snake_body[0][1])
        
        # Reward for eating food or game over conditions
        if self.snake_pos == self.food_pos:
            return 100      
        elif self.check_game_over():
            return -75
        
        # Base reward based on whether the snake is moving closer or farther from the food.
        reward = 15 if (previous_distance - current_distance > 0) else -15
        
        # Check if the snake's head is at any border and add a penalty of -20
        head_x, head_y = self.snake_body[0]
        if head_x == 0 or head_x == self.frame_size_x - 10 or head_y == 0 or head_y == self.frame_size_y - 10:
            reward -= 20
        
        return reward
    

    def check_game_over(self):
        # Return True if the game is over, else False
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            return True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            return True
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return True
                
        return False

    def update_snake_position(self, action):
        # Updates the snake's position based on the action
        # Map action to direction
        change_to = ''
        direction = self.direction
        if action == 0:
            change_to = 'UP'
        elif action == 1:
            change_to = 'DOWN'
        elif action == 2:
            change_to = 'LEFT'
        elif action == 3:
            change_to = 'RIGHT'
    
        # Move the snake
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
    
        if direction == 'UP':
            self.snake_pos[1] -= 10
        elif direction == 'DOWN':
            self.snake_pos[1] += 10
        elif direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif direction == 'RIGHT':
            self.snake_pos[0] += 10
            
        self.direction = direction
        
        
        self.snake_body.insert(0, list(self.snake_pos))
        
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 10
            self.food_spawn = False
            # If the snake is not growing
            if not self.growing_body:
                self.snake_body.pop()
        else:
            self.snake_body.pop()
    
    def update_food_position(self):
        """Updates the food position"""
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_x//10)) * 10]
            while self.food_pos in self.snake_body: # Ensures that the food does not spawn inside the body
                self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_x//10)) * 10]
        self.food_spawn = True
        
        

