import numpy as np
from scipy import stats
import random
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

class FrozenLake:
    def __init__(self, size=4, slippery=False):
        self.size = size
        self.grid = np.array([
            ['S', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H'],
            ['F', 'F', 'F', 'H'],
            ['H', 'F', 'F', 'G']
        ])
        self.start = (0, 0)
        self.goal = (3, 3)
        self.holes = [(i, j) for i in range(size) for j in range(size) if self.grid[i, j] == 'H']
        self.slippery = slippery
        self.state = self.start
        self.n_states = size * size
        self.n_actions = 4  # 0: left, 1: down, 2: right, 3: up

    def reset(self):
        self.state = self.start
        return self.pos_to_state(self.state)

    def pos_to_state(self, pos):
        return pos[0] * self.size + pos[1]

    def state_to_pos(self, s):
        return (s // self.size, s % self.size)

    def get_deltas(self):
        return [(0, -1), (1, 0), (0, 1), (-1, 0)]  # left, down, right, up

    def step(self, action):
        pos = self.state
        deltas = self.get_deltas()
        if self.slippery:
            p = np.random.choice([0, -1, 1], p=[1/3, 1/3, 1/3])
            effective_action = (action + p) % 4
        else:
            effective_action = action
        dx, dy = deltas[effective_action]
        new_pos = (pos[0] + dx, pos[1] + dy)
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = pos
        cell = self.grid[new_pos]
        if cell == 'H':
            reward = 0
            done = True
        elif cell == 'G':
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        self.state = new_pos
        return self.pos_to_state(new_pos), reward, done

    def change_env(self):
        self.grid[2, 1] = 'H'
        self.holes = [(i, j) for i in range(self.size) for j in range(size) if self.grid[i, j] == 'H']

def sample_entropy(time_series, m=2, r=0.2):
    N = len(time_series)
    r = r * np.std(time_series)
    def _phi(m):
        x = np.array([time_series[i:i+m] for i in range(N-m+1)])
        C = np.sum([np.sum(np.all(np.abs(x - x[i]) <= r, axis=1)) - 1 for i in range(len(x))])
        return C / (N-m)
    A = _phi(m)
    B = _phi(m+1)
    if A == 0 or B == 0:
        return np.inf
    return -np.log(B / A)

def get_epsilon(agent_type, epsilon_fixed, vol_epsilon):
    if agent_type == 'fixed':
        return epsilon_fixed
    elif agent_type == 'random':
        return random.uniform(0.01, 0.5)
    elif agent_type == 'volitional':
        return vol_epsilon
    return epsilon_fixed

def run_trial(agent_type, alpha=0.5, gamma=0.99, epsilon_fixed=0.1, max_steps=100, convergence_threshold=0.8, window=5):
    env = FrozenLake()
    Q = np.zeros((env.n_states, env.n_actions))
    states, actions, pre_rewards = [], [], []
    # Pre-train
    for ep in range(500):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            if random.random() < epsilon_fixed:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = np.argmax(Q[state])
            new_state, reward, done = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            total_reward += reward
            state = new_state
            if done:
                break
        pre_rewards.append(total_reward)
        if len(pre_rewards) >= window and np.mean(pre_rewards[-window:]) >= convergence_threshold:
            break
    # Post-Om training
    env.change_env()
    post_rewards, post_states, post_actions = [], [], []
    episodes_to_converge = 0
    performance_history = []
    vol_high_epsilon = False
    vol_epsilon = epsilon_fixed if agent_type == 'volitional' else None
    for ep in range(50):  # Limit to 50 episodes for analysis
        state = env.reset()
        total_reward = 0
        episode_states, episode_actions = [], []
        epsilon = get_epsilon(agent_type, epsilon_fixed, vol_epsilon)
        for step in range(max_steps):
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = np.argmax(Q[state])
            episode_states.append(state)
            episode_actions.append(action)
            new_state, reward, done = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action])
            total_reward += reward
            state = new_state
            if done:
                break
        post_rewards.append(total_reward)
        post_states.append(episode_states)
        post_actions.append(episode_actions)
        episodes_to_converge += 1
        if agent_type == 'volitional':
            performance_history.append(total_reward)
            if len(performance_history) >= 5:
                recent_avg = np.mean(performance_history[-5:])
                if recent_avg < 0.2 and not vol_high_epsilon:
                    vol_epsilon = 1.0
                    vol_high_epsilon = True
            if vol_high_epsilon:
                vol_epsilon *= 0.95
                if vol_epsilon < epsilon_fixed:
                    vol_epsilon = epsilon_fixed
                    vol_high_epsilon = False
        if len(post_rewards) >= window and np.mean(post_rewards[-window:]) >= convergence_threshold:
            break
    # Compute mutual information
    flat_states = np.concatenate([np.array(s) for s in post_states if s])
    flat_actions = np.concatenate([np.array(a) for a in post_actions if a])
    mi = mutual_info_score(flat_states, flat_actions) if len(flat_states) == len(flat_actions) and len(flat_states) > 0 else np.nan
    # Compute Sample Entropy per episode
    episode_sampen = []
    for ep in range(len(post_actions)):
        actions_ep = np.array(post_actions[ep]) if post_actions[ep] else np.array([])
        episode_sampen.append(sample_entropy(actions_ep) if len(actions_ep) >= 10 else np.nan)
    return episodes_to_converge, post_rewards, mi, episode_sampen

def run_simulations(num_trials=50):
    types = ['fixed', 'random', 'volitional']
    results = {t: {'episodes': [], 'rewards': [], 'mi': [], 'sampen': []} for t in types}
    for t in types:
        for trial in range(num_trials):
            eps, rews, mi, sampen = run_trial(t)
            results[t]['episodes'].append(eps)
            results[t]['rewards'].append(rews)
            results[t]['mi'].append(mi)
            results[t]['sampen'].append(sampen)
    stats_dict = {t: {'mean_eps': np.mean(results[t]['episodes']), 'sd_eps': np.std(results[t]['episodes']),
                      'mean_mi': np.nanmean(results[t]['mi']), 'mean_sampen': np.nanmean([np.nanmean(s) for s in results[t]['sampen']])} for t in types}
    t_stat, p_val = stats.ttest_ind(results['volitional']['episodes'], results['fixed']['episodes'])
    max_len = max(len(r) for t in types for r in results[t]['rewards'])
    avg_rewards = {t: np.nanmean([np.pad(np.array(r), (0, max_len - len(r)), mode='constant', constant_values=np.nan) for r in results[t]['rewards']], axis=0) for t in types}
    avg_sampen = {t: np.nanmean([np.pad(np.array(s), (0, max_len - len(s)), mode='constant', constant_values=np.nan) for s in results[t]['sampen']], axis=0) for t in types}
    return stats_dict, p_val, avg_rewards, avg_sampen

# Run simulation
sim_results, p_value, avg_rewards, avg_sampen = run_simulations()

# Save results to CSV
import pandas as pd
results_df = pd.DataFrame({
    'Agent': ['Fixed', 'Random', 'Volitional'],
    'Mean_Episodes': [sim_results[t]['mean_eps'] for t in ['fixed', 'random', 'volitional']],
    'SD_Episodes': [sim_results[t]['sd_eps'] for t in ['fixed', 'random', 'volitional']],
    'Mean_MI': [sim_results[t]['mean_mi'] for t in ['fixed', 'random', 'volitional']],
    'Mean_SampEn': [sim_results[t]['mean_sampen'] for t in ['fixed', 'random', 'volitional']]
})
results_df.to_csv('simulation_results.csv', index=False)

# Generate Figures
episodes = range(50)
plt.figure(figsize=(8, 6))
plt.plot(episodes, avg_rewards['volitional'][:50], label='Volitional Agent', color='blue')
plt.plot(episodes, avg_rewards['fixed'][:50], label='Fixed Agent', color='red')
plt.plot(episodes, avg_rewards['random'][:50], label='Random Agent', color='green')
plt.xlabel('Episodes Post-Om Event')
plt.ylabel('Mean Reward')
plt.title('Mean Reward per Episode Post-Om Event (50 Trials)')
plt.legend()
plt.grid(True)
plt.savefig('figure_5.1.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(episodes, avg_sampen['volitional'][:50], label='Volitional Agent', color='blue')
plt.plot(episodes, avg_sampen['fixed'][:50], label='Fixed Agent', color='red')
plt.plot(episodes, avg_sampen['random'][:50], label='Random Agent', color='green')
plt.xlabel('Episodes Post-Om Event')
plt.ylabel('Mean Sample Entropy')
plt.title('Mean Sample Entropy of Action Sequences Post-Om Event (50 Trials)')
plt.legend()
plt.grid(True)
plt.savefig('figure_5.2.png', dpi=300)
plt.close()
