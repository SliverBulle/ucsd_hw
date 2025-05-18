import numpy as np
import random
import matplotlib.pyplot as plt

# 状态 ID 转换函数
def get_state_id(row, col):
    return row * 2 + col

# 状态转移函数
def transition(s, a):
    row, col = s
    if row == 4:  # terminal state
        return s
    if a == 0:  # down 1
        return (min(row + 1, 4), col)
    elif a == 1:  # down 2
        return (min(row + 2, 4), col)
    elif a == 2:  # cross and down 1
        return (min(row + 1, 4), 1 - col)
    elif a == 3:  # stay
        return (row, col)

# 奖励函数
def reward(s, a, s_prime):
    if s_prime[0] == 4:  # reach target (4,0) or (4,1)
        return 100
    return -1  # penalty for each step move

# Q-learning parameters
gamma = 0.9  # discount factor
num_episodes = 1000  # number of training episodes
epsilon = 1.0  # initial exploration rate
epsilon_min = 0.01  # minimum exploration rate
epsilon_decay = 0.995  # exploration rate decay
start_states = [(0, 0), (0, 1)]  # initial state set

# initialize Q table and visit count
Q = np.zeros((10, 4))  # 10 states, 4 actions
visit = np.zeros((10, 4))  # record the number of visits to each state-action pair
history = [[] for _ in range(4)]  # record the history of Q values for D0=(0,0)
D0_id = get_state_id(0, 0)  # the state ID of D0=(0,0) is 0

# Q-learning training loop
for episode in range(num_episodes):
    s = random.choice(start_states)  # randomly choose the initial state
    s_id = get_state_id(s[0], s[1])

    while True:
        # ε-greedy policy to select action
        if random.random() < epsilon:
            a = random.randint(0, 3)  # random exploration
        else:
            a = np.argmax(Q[s_id])  # exploit the optimal action

        # execute action and get next state and reward
        s_prime = transition(s, a)
        s_prime_id = get_state_id(s_prime[0], s_prime[1])
        r = reward(s, a, s_prime)

        # update Q value
        visit[s_id][a] += 1
        alpha = 1.0 / visit[s_id][a]  # learning rate based on visit count
        if s_prime[0] == 4:  # terminal state
            target = r
        else:
            target = r + gamma * np.max(Q[s_prime_id])
        Q[s_id][a] += alpha * (target - Q[s_id][a])

        # record the history of Q values for D0
        if s_id == D0_id:
            history[a].append(Q[s_id][a])

        # update current state
        s = s_prime
        s_id = s_prime_id

        # check if the terminal state is reached
        if s[0] == 4:
            break

    # decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# output the Q values for D0
print("Q values for D0 (0,0):")
actions = ['down 1', 'down 2', 'cross and down 1', 'stay']
for a in range(4):
    print(f"Q(D0, {actions[a]}): {Q[D0_id][a]:.1f}")

# plot the Q value update curve
plt.figure(figsize=(10, 6))
for a in range(4):
    visits = range(1, len(history[a]) + 1)
    plt.plot(visits, history[a], label=f'Q(D0, {actions[a]})')
plt.xlabel('Number of visits to (D0, a)')
plt.ylabel('Q value')
plt.legend()
plt.title('Q value changes with the number of visits to D0 (0,0)')
plt.savefig('/root/code/ucsd_hw/257_hw/hw3/figure/q3_plot.png')
plt.show()