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

def main():
    # Window size
    FRAME_SIZE_X = 300
    FRAME_SIZE_Y = 300
    
    # Colors (R, G, B)
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)
    BLUE = pygame.Color(0, 0, 255)
    
    difficulty = 10  # Adjust as needed
    render_game = True # Show the game or not
    growing_body = True # Makes the body of the snake grow
    training = True # Defines if it should train or not

    # Defining our states and actions
    number_states = 8
    number_actions = 4
    num_episodes = 5000 # Episode we want for training, everytime an apple is  eaten or snake dies an episode is finished

    # Initialize the game window, environment and q_learning algorithm
    # Your code here.
    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    ql = QLearning(n_states=number_states, n_actions=number_actions)  
    


    if render_game:
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        fps_controller = pygame.time.Clock()
    
    # Loading the table
    ql.load_q_table(filename="qtable.txt")
        
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False
        
        while not game_over:
            # Your code here.
            # Choose the best action for the state and possible actions from the q_learning algorithm
            # Call the environment step with that action and get next_state, reward and game_over variables



            # Obtaining the current state and encoding it
            state = env.get_state()
            print("state",state)
            enc_state = ql.encode_state(state)
            print("enc_state",enc_state)

            # Obtaining the directions and action taken
            directions = [0,1,2,3]
            action = ql.choose_action(enc_state, directions)
            print("action",action)
            nextState, reward, game_over = env.step(action)
            print("nextState",nextState)
            print(f"reward {reward}\n\n")


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
        ql.save_q_table()
        # Saving out hyperparameters
        #ql.save_hyperparams(episode+1,total_reward)
        print(f"Episode {episode+1}, Total reward: {total_reward}")

if __name__ == "__main__":
    main()
