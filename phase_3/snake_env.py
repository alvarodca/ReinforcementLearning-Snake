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
            print("Food Position",food_pos)
            # Ensure the food is not inside the snake's body
            print("Snake_body", self.snake_body)
            if food_pos not in self.snake_body:
                break

        self.food_pos = food_pos
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        self.reward = 0 # Initialize the starting reward
        return self.get_state2()
    

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
        state = self.get_state3()
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
    

    def calculate_danger(self):
        """
        Calculates immediate danger around the snake's head.
        For each direction ("top", "bottom", "left", "right"), if the cell immediately 
        adjacent is off the board or part of the snake's body then that direction is dangerous.
        Importantly, we do not check the direction opposite to the snake's movement.
        
        Returns a dictionary with keys "top", "bottom", "left", "right" and values:
            1 if danger is present, 0 if not.
        """
        head_x, head_y = self.snake_body[0]
        step = 10  # grid cell size
        danger = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        
        # Get the opposite direction and skip checking it.
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        current_direction = self.direction
        opposite_direction = opposites.get(current_direction, None)
        
        # Check "top"
        if opposite_direction == "UP":
            danger["top"] = 0
        else:
            up_cell = [head_x, head_y - step]
            if head_y - step < 0 or up_cell in self.snake_body:
                danger["top"] = 1

        # Check "bottom"
        if opposite_direction == "DOWN":
            danger["bottom"] = 0
        else:
            down_cell = [head_x, head_y + step]
            if head_y + step >= self.frame_size_y or down_cell in self.snake_body:
                danger["bottom"] = 1

        # Check "left"
        if opposite_direction == "LEFT":
            danger["left"] = 0
        else:
            left_cell = [head_x - step, head_y]
            if head_x - step < 0 or left_cell in self.snake_body:
                danger["left"] = 1

        # Check "right"
        if opposite_direction == "RIGHT":
            danger["right"] = 0
        else:
            right_cell = [head_x + step, head_y]
            if head_x + step >= self.frame_size_x or right_cell in self.snake_body:
                danger["right"] = 1

        return danger
        


    def get_state2(self):
        """
        Returns the state as a tuple:
            (food_state, danger)
        
        food_state:
            One of eight values, depending on the relative position of the food:
            - "UP", "DOWN", "LEFT", "RIGHT" if snake and food are aligned,
            - or a tuple ("LEFT", "UP"), ("RIGHT", "UP"), ("LEFT", "DOWN"), ("RIGHT", "DOWN") if diagonal.
        
        danger:
            A tuple with one or two blocked directions.
            The first blocked direction is always the one opposite to the snake’s current movement.
            In addition, if another immediate danger (from calculate_danger) exists (besides the forced one),
            that second direction is included.
        """
        # Determine food_state
        head_x, head_y = self.snake_body[0]
        food_x, food_y = self.food_pos

        if food_x == head_x:
            food_state = "UP" if food_y < head_y else "DOWN"
        elif food_y == head_y:
            food_state = "LEFT" if food_x < head_x else "RIGHT"
        else:
            hor = "LEFT" if food_x < head_x else "RIGHT"
            ver = "UP" if food_y < head_y else "DOWN"
            food_state = (hor, ver)

        # Force the blocked (danger) direction: always include the opposite of the current movement.
        opposites = {"UP": "bottom", "DOWN": "top", "LEFT": "right", "RIGHT": "left"}
        forced_block = opposites.get(self.direction, "none")

        # Get current danger signals (calculate_danger does not consider the opposite)
        danger_dict = self.calculate_danger()

        # Check for any additional danger – excluding the forced_block,
        # so that if, for example, the snake has a danger on "top" plus the forced "RIGHT" (snake going left),
        # we include both.
        additional = "none"
        danger_order = ["top", "bottom", "left", "right"]
        for d in danger_order:
            if d != forced_block and danger_dict[d] == 1:
                additional = d
                break

        if additional:
            danger = (forced_block, additional)
        else:
            danger = (forced_block,)

        return (food_state, danger)
    

    def get_state3(self):
        """
        Returns the state as a tuple:
            (food_state, danger)

        food_state:
            One of eight values, depending on the relative position of the food:
            - "UP", "DOWN", "LEFT", "RIGHT" if aligned
            - or a tuple, e.g. ("LEFT", "UP"), ("RIGHT", "DOWN"), etc., if diagonal.

        danger:
            A tuple with three elements:
            - The first element is the forced blocked direction – always the opposite of the current movement.
            - The next two elements are additional immediate danger signals (if any) from calculate_danger,
                in a preset priority order. If fewer than two additional danger directions are detected,
                the remaining elements are set to "none".
        """
        # Determine food_state
        head_x, head_y = self.snake_body[0]
        food_x, food_y = self.food_pos

        if food_x == head_x:
            food_state = "UP" if food_y < head_y else "DOWN"
        elif food_y == head_y:
            food_state = "LEFT" if food_x < head_x else "RIGHT"
        else:
            hor = "LEFT" if food_x < head_x else "RIGHT"
            ver = "UP" if food_y < head_y else "DOWN"
            food_state = (hor, ver)

        # Force the blocked (danger) direction: always include opposite of the current movement.
        # (Note: Here we use a mapping that yields lowercase danger names.)
        opposites = {"UP": "bottom", "DOWN": "top", "LEFT": "right", "RIGHT": "left"}
        forced_block = opposites.get(self.direction, "none")

        # Get current danger signals.
        danger_dict = self.calculate_danger()

        # Look for up to two additional danger directions (excluding the forced_block).
        additional = []
        danger_order = ["top", "bottom", "left", "right"]
        for d in danger_order:
            if d != forced_block and danger_dict[d] == 1:
                additional.append(d)
            if len(additional) == 2:
                break

        # Ensure the danger tuple has three elements.
        while len(additional) < 2:
            additional.append("none")

        danger = (forced_block, additional[0], additional[1])
        return (food_state, danger)


    def get_body(self):
    	return self.snake_body
    
    def get_food(self):
    	return self.food_pos
    

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
        """head_x, head_y = self.snake_body[0]
        if head_x == 0 or head_x == self.frame_size_x - 10 or head_y == 0 or head_y == self.frame_size_y - 10:
            reward -= 20"""
        
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
        if not self.food_spawn:

            # With a 25% chance the food will appear in on e of the borders
            if random.random() < 0.00:
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
                
            self.food_pos = [x, y]

            while self.food_pos in self.snake_body: # Ensures that the food does not spawn inside the body
                self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_x//10)) * 10]
        self.food_spawn = True
        
        

