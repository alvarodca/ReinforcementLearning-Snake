import sys
import pygame
import numpy as np
from snake_env import SnakeGameEnv
from q_learning import QLearning

def main():
    # Parameters
    FRAME_SIZE_X = 300
    FRAME_SIZE_Y = 300
    test_episodes = 10          # Number of test episodes to run
    render_game = True          # Set to False to disable rendering during tests
    
    # Initialize environment and Q-learning agent
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body=True)
    number_states = 40          # As defined by your encode_state2 mapping (8 * 5)
    number_actions = 4
    ql = QLearning(n_states=number_states, n_actions=number_actions)
    
    # Load the Q-table from file
    try:
        ql.q_table = np.loadtxt("qtable.txt")
    except Exception as e:
        print("Error loading Q-table:", e)
        sys.exit(1)
    
    # Disable exploration for testing
    ql.epsilon = 0
    
    # Setup game window (if rendering is enabled)
    if render_game:
        pygame.init()
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        pygame.display.set_caption("Test Q-Table")
        fps_controller = pygame.time.Clock()
        BLACK = pygame.Color(0, 0, 0)
        GREEN = pygame.Color(0, 255, 0)
        RED = pygame.Color(255, 0, 0)
    
    total_rewards_all = 0
    # Define action mapping
    action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    # Run test episodes
    for ep in range(test_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False
        
        last_action = None  # Record last action taken before dying.
        last_state  = None  # Record last state before dying.
        
        while not game_over:
            enc_state = ql.encode_state2(state)
            action = np.argmax(ql.q_table[enc_state])
            last_state = state  # Store current state before taking the action
            last_action = action  # Store the selected action
            state, reward, game_over = env.step(action)
            total_reward += reward
            
            if render_game:
                game_window.fill(BLACK)
                snake_body = env.get_body()
                food_pos = env.get_food()
                for pos in snake_body:
                    pygame.draw.rect(game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
                pygame.draw.rect(game_window, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
                
                # Allow for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                pygame.display.flip()
                fps_controller.tick(20)  # Adjust test speed if needed
        
        # Print the final state and last action (as readable strings) before dying
        print(f"Test Episode {ep+1} Total Reward: {total_reward}")
        # Decode the last state: food_state could be a string or a tuple.
        food_state = last_state[0]
        if isinstance(food_state, tuple):
            food_state_str = f"{food_state[0]}-{food_state[1]}"
        else:
            food_state_str = food_state
        danger_str = last_state[1]
        last_action_str = action_map.get(last_action, str(last_action))
        print(f"Last state: Food = {food_state_str}, Danger = {danger_str} | Last Action Taken: {last_action_str}")
        total_rewards_all += total_reward

    avg_reward = total_rewards_all / test_episodes
    print("Average Reward over Test Episodes:", avg_reward)
    
    if render_game:
        pygame.quit()

if __name__ == "__main__":
    main()