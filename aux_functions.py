import numpy as np
import matplotlib.pyplot as plt

def decode_state(state_index):
    """
    Decodes an integer state index (0 to 127) into a (food_state, danger) state tuple 
    according to the new encoding.
    
    Food_state is encoded as:
      0: "UP"
      1: "DOWN"
      2: "LEFT"
      3: "RIGHT"
      4: ("LEFT", "UP")
      5: ("RIGHT", "UP")
      6: ("LEFT", "DOWN")
      7: ("RIGHT", "DOWN")
      
    Danger is encoded as (using 16 possible codes):
      0: ("top", "none")
      1: ("top", "bottom")
      2: ("top", "left")
      3: ("top", "right")
      4: ("bottom", "none")
      5: ("bottom", "top")
      6: ("bottom", "left")
      7: ("bottom", "right")
      8: ("left", "none")
      9: ("left", "top")
      10: ("left", "bottom")
      11: ("left", "right")
      12: ("right", "none")
      13: ("right", "top")
      14: ("right", "bottom")
      15: ("right", "left")
      
    The overall encoding is:
        state_index = food_state_index * 16 + danger_code
    Total state space: 8 * 16 = 128.
    """
    # Decode food_state
    food_index = state_index // 16
    if food_index < 4:
        simple_map = ["UP", "DOWN", "LEFT", "RIGHT"]
        food_state = simple_map[food_index]
    else:
        diagonal_map = {
            4: ("LEFT", "UP"),
            5: ("RIGHT", "UP"),
            6: ("LEFT", "DOWN"),
            7: ("RIGHT", "DOWN")
        }
        food_state = diagonal_map.get(food_index, None)
    
    # Decode danger
    danger_code = state_index % 16
    danger_inv = {
        0: ("top", "none"),
        1: ("top", "bottom"),
        2: ("top", "left"),
        3: ("top", "right"),
        4: ("bottom", "none"),
        5: ("bottom", "top"),
        6: ("bottom", "left"),
        7: ("bottom", "right"),
        8: ("left", "none"),
        9: ("left", "top"),
        10: ("left", "bottom"),
        11: ("left", "right"),
        12: ("right", "none"),
        13: ("right", "top"),
        14: ("right", "bottom"),
        15: ("right", "left")
    }
    danger = danger_inv.get(danger_code, ("none", "none"))
    
    return (food_state, danger)


def plot_reward_and_length(filename="episode_rewards.txt", group_size=5):
    """
    Reads total reward and snake length for each episode from filename,
    groups every group_size episodes, computes the mean reward and mean snake 
    length for each group, then plots two subplots:
      - Mean reward progress with a fitted regression line.
      - Mean snake length progress with a fitted regression line.
    """
    rewards = []
    lengths = []
    
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        r = float(parts[0])
                        l = float(parts[1])
                        rewards.append(r)
                        lengths.append(l)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    n_groups = len(rewards) // group_size
    if n_groups == 0:
        print("Not enough data to form groups.")
        return

    grouped_rewards = [np.mean(rewards[i*group_size:(i+1)*group_size]) for i in range(n_groups)]
    grouped_lengths = [np.mean(lengths[i*group_size:(i+1)*group_size]) for i in range(n_groups)]
    x_values = np.arange(1, n_groups+1) * group_size

    slope_r, intercept_r = np.polyfit(x_values, grouped_rewards, 1)
    regression_line_r = slope_r * x_values + intercept_r

    slope_l, intercept_l = np.polyfit(x_values, grouped_lengths, 1)
    regression_line_l = slope_l * x_values + intercept_l

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(x_values, grouped_rewards, color='blue', label='Grouped Mean Reward')
    plt.plot(x_values, regression_line_r, color='red', linestyle='--', 
             label=f'Regression Line\n(slope={slope_r:.2f})')
    plt.xlabel(f"Episode (Grouped every {group_size} episodes)")
    plt.ylabel("Mean Total Reward")
    plt.title("Agent Reward Progress")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(x_values, grouped_lengths, color='green', label='Grouped Mean Length')
    plt.plot(x_values, regression_line_l, color='orange', linestyle='--', 
             label=f'Regression Line\n(slope={slope_l:.2f})')
    plt.xlabel(f"Episode (Grouped every {group_size} episodes)")
    plt.ylabel("Mean Snake Length")
    plt.title("Agent Snake Length Progress")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Now, our Q-table has 128 rows, one for each state index.
    n_states = 128   
    n_actions = 4    # Actions: e.g., UP, DOWN, LEFT, RIGHT
    filename = "qtable.txt"
    
    # Define action mapping.
    action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    try:
        q_table = np.loadtxt(filename)
    except Exception as e:
        print("Error loading Q-table:", e)
        return

    print("Best policy (state -> best action):")
    for state_index in range(n_states):
        state_decoded = decode_state(state_index)
        best_action_idx = np.argmax(q_table[state_index])
        best_action = action_map.get(best_action_idx, best_action_idx)
        # Check if the best action appears in the danger tuple (case insensitive)
        danger = state_decoded[1]
        warning = ""
        if any(best_action.lower() == d.lower() for d in danger if d != "none"):
            warning = " [WARNING: Action is blocked!]"
        print(f"State index {state_index:3} : {state_decoded} -> Best action: {best_action}{warning}")

if __name__ == "__main__":
    main()
    plot_reward_and_length()