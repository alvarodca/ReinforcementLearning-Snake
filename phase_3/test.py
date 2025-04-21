import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeGameEnv
from q_learning import QLearning

def test_agent(num_episodes=200, difficulty=10, frame_size=150):
    """
    Runs the test agent for num_episodes without learning, saves the total reward,
    snake length, and score for each episode, and returns these as lists.
    """
    pygame.init()
    env = SnakeGameEnv(frame_size, frame_size, growing_body=True)
    number_states = 320
    number_actions = 4
    # Create a QLearning agent with epsilon=0 to disable exploration.
    ql = QLearning(n_states=number_states, n_actions=number_actions, epsilon=0)
    ql.load_q_table("qtable_phase3.txt")
    
    # Optionally, you can create a game window if you wish to render.
    render_game = False
    if render_game:
        game_window = pygame.display.set_mode((frame_size, frame_size))
        fps_controller = pygame.time.Clock()
    
    rewards_list = []
    lengths_list = []
    scores_list = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        score = 0
        game_over = False
        
        while not game_over:
            enc_state = ql.encode_state3(state)
            # Get the action (greedy since epsilon=0).
            action = ql.choose_action(enc_state, [0,1,2,3])
            next_state, reward, game_over = env.step(action)
            
            # Update score (if an apple is eaten, reward==100; otherwise, penalize).
            if reward == 100:
                score += 100
            else:
                score -= 1
            
            total_reward += reward
            state = next_state
            
            # Render if needed.
            if render_game:
                game_window.fill(pygame.Color(0, 0, 0))
                snake_body = env.get_body()
                food_pos = env.get_food()
                for pos in snake_body:
                    pygame.draw.rect(game_window, pygame.Color(0, 255, 0), 
                                     pygame.Rect(pos[0], pos[1], 10, 10))
                pygame.draw.rect(game_window, pygame.Color(255, 0, 0), 
                                 pygame.Rect(food_pos[0], food_pos[1], 10, 10))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                pygame.display.flip()
                fps_controller.tick(difficulty)
        
        rewards_list.append(total_reward)
        lengths_list.append(len(env.get_body()))
        scores_list.append(score)
        print(f"Episode {episode+1}: Reward = {total_reward}, Length = {len(env.get_body())}, Score = {score}")
    
    # Save results to a file.
    with open("test_results.txt", "w") as f:
        for i in range(num_episodes):
            f.write(f"{scores_list[i]}\t{rewards_list[i]}\t{lengths_list[i]}\n")
    
    return rewards_list, lengths_list, scores_list

def plot_histograms(rewards, lengths, scores):
    """
    Plots a separate histogram for total rewards, snake lengths, and scores.
    Highlights the mean with a vertical line and shows its value.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14,4))
    
    # Plot Reward distribution.
    plt.subplot(1, 3, 1)
    plt.hist(rewards, bins=20, color='blue', edgecolor='black')
    mean_reward = np.mean(rewards)
    plt.axvline(mean_reward, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.2f}')
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.legend()
    
    # Plot Snake Length distribution.
    plt.subplot(1, 3, 2)
    plt.hist(lengths, bins=20, color='green', edgecolor='black')
    mean_length = np.mean(lengths)
    plt.axvline(mean_length, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.2f}')
    plt.xlabel("Snake Length")
    plt.ylabel("Frequency")
    plt.title("Snake Length Distribution")
    plt.legend()

    # Plot Score distribution.
    plt.subplot(1, 3, 3)
    plt.hist(scores, bins=20, color='red', edgecolor='black')
    mean_score = np.mean(scores)
    plt.axvline(mean_score, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution")
    plt.legend()
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    rewards, lengths, scores = test_agent(num_episodes=1000, difficulty=10000, frame_size=150)
    plot_histograms(rewards, lengths, scores)
    pygame.quit()