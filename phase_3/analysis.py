import numpy as np
import matplotlib.pyplot as plt
import math


def decode_state(state_index):
    """
    Decodes an integer state index (0 to 319) into a (food_state, danger) state tuple
    according to the above encoding.
    
    Food_state is decoded as:
        0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT",
        4: ("LEFT", "UP"), 5: ("RIGHT", "UP"), 6: ("LEFT", "DOWN"), 7: ("RIGHT", "DOWN")
    
    Danger is decoded from a danger code (0 to 39) as follows:
    - If danger_code < 4, then 1-danger case: forced = danger_code mapped via forced_inv and add1 = add2 = "none".
    - If 4 <= danger_code < 16, then 2-dangers case.
        Let temp = danger_code - 4. Then f_idx = temp // 3 and candidate index = temp % 3.
        forced = forced_inv[f_idx] and then candidate list = [d for d in ["top","bottom","left","right"] if d != forced],
        so add1 = candidate_list[candidate index] and add2 = "none".
    - If 16 <= danger_code < 40, then 3-dangers case.
        Let temp = danger_code - 16. Then f_idx = temp // 6, remainder = temp % 6.
        forced = forced_inv[f_idx]. Then, candidate list = [d for d in ["top","bottom","left","right"] if d != forced].
        Let pos1 = remainder // 2 and pos2 = remainder % 2.
        Then add1 = candidate_list[pos1],
        and candidate_list2 = [d for d in candidate_list if d != add1],
        so add2 = candidate_list2[pos2].
    
    Returns (food_state, (forced, add1, add2)).
    """
    # Decode food_state.
    food_index = state_index // 40
    if food_index < 4:
        food_state = ["UP", "DOWN", "LEFT", "RIGHT"][food_index]
    else:
        diagonal_map = {
            4: ("LEFT", "UP"),
            5: ("RIGHT", "UP"),
            6: ("LEFT", "DOWN"),
            7: ("RIGHT", "DOWN")
        }
        food_state = diagonal_map.get(food_index, "UP")  # fallback

    danger_code = state_index % 40
    forced_inv = {0: "top", 1: "bottom", 2: "left", 3: "right"}
    if danger_code < 4:
        # 1-danger case
        f_idx = danger_code
        forced = forced_inv.get(f_idx, "top")
        add1, add2 = "none", "none"
    elif danger_code < 16:
        # 2-dangers case
        temp = danger_code - 4
        f_idx = temp // 3
        pos = temp % 3
        forced = forced_inv.get(f_idx, "top")
        candidates = [d for d in ["top", "bottom", "left", "right"] if d != forced]
        add1 = candidates[pos] if pos < len(candidates) else "none"
        add2 = "none"
    else:
        # 3-dangers case.
        temp = danger_code - 16
        f_idx = temp // 6
        remainder = temp % 6
        forced = forced_inv.get(f_idx, "top")
        candidates = [d for d in ["top", "bottom", "left", "right"] if d != forced]
        pos1 = remainder // 2
        pos2 = remainder % 2
        add1 = candidates[pos1] if pos1 < len(candidates) else "none"
        candidates2 = [d for d in candidates if d != add1]
        add2 = candidates2[pos2] if pos2 < len(candidates2) else "none"
        
    danger = (forced, add1, add2)
    return (food_state, danger)
    

def main():
    # Now, our Q-table has 128 rows—one for each state index (total state space of 8*100 = 800).
    n_states = 320  
    n_actions = 4    # Actions: UP, DOWN, LEFT, RIGHT
    filename = "qtable_phase3.txt"
    
    # Define action mapping.
    action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    try:
        q_table = np.loadtxt(filename)
    except Exception as e:
        print("Error loading Q-table:", e)
        return

    print("Best policy (state -> best actions):")
    for state_index in range(n_states):
        state_decoded = decode_state(state_index)
        row_actions = q_table[state_index]
        max_val = np.max(row_actions)
        # Identify all actions tied with the maximum value (using a tolerance for floating-point imprecision).
        best_indices = np.where(np.abs(row_actions - max_val) < 1e-6)[0]
        best_actions = [action_map.get(idx, str(idx)) for idx in best_indices]
        best_action_str = ", ".join(best_actions)
        
        # Check if any best action is in the danger tuple (ignoring "none"), case insensitive.
        danger = state_decoded[1]  # danger is a 3-element tuple
        blocked = any(any(best.lower() == d.lower() for d in danger if d != "none") for best in best_actions)
        warning = " [WARNING: Action is blocked!]" if blocked else ""
        
        print(f"\nState index {state_index:3} : {state_decoded} -> Best actions: {best_action_str}{warning}")



import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def process_data(filename="episode_rewards.txt", group_size=30):
    """
    Reads the rewards and lengths from the file; groups them; and returns
    the x-values (iteration numbers), grouped rewards, and grouped lengths.
    """
    rewards = []
    lengths = []
    
    try:
        with open(filename, "r") as f:
            for line in f:
                try:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        # parts[0] is reward, parts[2] is length.
                        r = float(parts[0])
                        l = float(parts[2])
                        rewards.append(r)
                        lengths.append(l)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None, None, None

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    n_groups = len(rewards) // group_size
    if n_groups == 0:
        print("Not enough data to form groups.")
        return None, None, None

    grouped_rewards = [np.mean(rewards[i*group_size:(i+1)*group_size]) for i in range(n_groups)]
    grouped_lengths = [np.mean(lengths[i*group_size:(i+1)*group_size]) for i in range(n_groups)]
    x_values = np.arange(1, n_groups+1) * group_size
    
    # Convert grouped lists to NumPy arrays.
    grouped_rewards = np.array(grouped_rewards)
    grouped_lengths = np.array(grouped_lengths)
    
    return x_values, grouped_rewards, grouped_lengths


def plot_segment(ax, x_vals, y_vals, seg_range, color):
    start_val, end_val = seg_range
    mask = (x_vals >= start_val) & (x_vals <= end_val)
    if np.sum(mask) >= 4:  # Ensure enough data points
        seg_x = x_vals[mask]
        seg_y = y_vals[mask]  # Now y_vals is a NumPy array.
        win_len = int(np.sum(mask))
        if win_len % 2 == 0:
            win_len = max(3, win_len - 1)
        smooth_y = savgol_filter(seg_y, window_length=win_len, polyorder=2)
        ax.plot(seg_x, smooth_y, color=color, linestyle='--',
                label=f'{start_val}-{end_val} Trend')


def plot_reward(filename="episode_rewards.txt", group_size=30):
    """
    Creates a plot for grouped rewards with vertical lines at 5000, 6000, and 7000,
    and smooth trend curves for defined segments.
    """
    x_vals, grouped_rewards, _ = process_data(filename, group_size)
    if x_vals is None:
        return

    segments = [(0, 5000), (5001, 6000), (6001, 7000), (7001, 7500)]
    colors = ['red', 'green', 'orange', 'purple']

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, grouped_rewards, color='blue', label='Grouped Mean Reward')
    for v in [5000, 6000, 7000]:
        plt.axvline(v, color='black', linestyle='--', linewidth=1)
    for i, seg in enumerate(segments):
        plot_segment(plt.gca(), x_vals, grouped_rewards, seg, colors[i])
    plt.xlabel(f"Episode (Grouped every {group_size} episodes)")
    plt.ylabel("Mean Total Reward")
    plt.title("Agent Reward Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_length(filename="episode_rewards.txt", group_size=30):
    """
    Creates a plot for grouped snake lengths with vertical lines at 5000, 6000, and 7000,
    and smooth trend curves for defined segments.
    """
    x_vals, _, grouped_lengths = process_data(filename, group_size)
    if x_vals is None:
        return

    segments = [(0, 5000), (5001, 6000), (6001, 7000), (7001, 7500)]
    colors = ['red', 'green', 'orange', 'purple']

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, grouped_lengths, color='green', label='Grouped Mean Length')
    for v in [5000, 6000, 7000]:
        plt.axvline(v, color='black', linestyle='--', linewidth=1)
    for i, seg in enumerate(segments):
        plot_segment(plt.gca(), x_vals, grouped_lengths, seg, colors[i])
    plt.xlabel(f"Episode (Grouped every {group_size} episodes)")
    plt.ylabel("Mean Snake Length")
    plt.title("Agent Snake Length Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    plot_reward(group_size=30)
    plot_length(group_size=30)