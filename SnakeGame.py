"""
Snake Eater Game
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
from snake_env import SnakeGameEnv
from q_learning import QLearning
import pygame
import sys
import numpy as np

def main(training=True, difficulty=1000):
    # Window size
    FRAME_SIZE_X = 150
    FRAME_SIZE_Y = 150
    
    # Colors (R, G, B)
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)
    BLUE = pygame.Color(0, 0, 255)
    
    render_game = True # Show the game or not
    growing_body = True # Makes the body of the snake grow

    # Defining our states and actions
    number_states = 320
    number_actions = 4
    num_episodes = 500 # Episode we want for training, everytime an apple is  eaten or snake dies an episode is finished

    # Initialize the game window, environment and q_learning algorithm
    # Your code here.
    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)

    if training: 
        ql = QLearning(n_states=number_states, n_actions=number_actions)
    else: 
        ql = QLearning(n_states=number_states, n_actions=number_actions, epsilon=0)
    


    if render_game:
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        fps_controller = pygame.time.Clock()
    
    # Loading the table
    ql.load_q_table(filename="qtable_phase3.txt")
        
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False
        score = 0
        while not game_over:
            # Your code here.
            # Choose the best action for the state and possible actions from the q_learning algorithm
            # Call the environment step with that action and get next_state, reward and game_over variables



            # Obtaining the current state and encoding it
            print(f"--------EPISODE {episode+1}--------")
            state = env.get_state3()
            print("state",state)
            enc_state = ql.encode_state3(state)
            print("enc_state",enc_state)

            # Obtaining the directions and action taken
            directions = [0,1,2,3]
            action = ql.choose_action(enc_state, directions)
            print("action",action)
            nextState, reward, game_over = env.step(action)
            print("nextState",nextState)
            print(f"reward {reward}\n\n")
            # Saving the score to update it later
            if reward == 100: # Apple is eaten
                score += 100
            else:
                score -= 1

            if training:
                #update the q table using those variables.
                ql.update_q_table(state,action,reward,nextState)

            # Update the state and the total_reward.
            state = nextState # Updating state
            total_reward += reward # Updating reward
            
            # Render
            if render_game:
                game_window.fill(BLACK)
                snake_body = env.get_body()
                food_pos = env.get_food()
                for pos in snake_body:
                    pygame.draw.rect(game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
        
                pygame.draw.rect(game_window, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
            
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    
                pygame.display.flip()
                fps_controller.tick(difficulty)
        
        # Saving our table
        ql.save_q_table(filename="qtable_phase3.txt")
        # Saving out hyperparameters
        # ql.save_hyperparams(episode+1,total_reward)
        print(f"Episode {episode+1}, Total reward: {total_reward}, Snake length: {len(env.get_body())}")
        # Save both total reward and snake length (tab separated)
        with open("episode_rewards.txt", "a") as f:
           f.write(f"{score}\t{total_reward}\t{len(env.get_body())}\n")





if __name__ == "__main__":
    main(training=False, difficulty=15)

    
