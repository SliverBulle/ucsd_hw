import numpy as np
import matplotlib.pyplot as plt

# the true distribution of the coins
p1 = 0.8  # the probability of coin 1 getting 1
p2 = 0.4  # the probability of coin 2 getting 1

# simulate coin tossing
def simulate_coin(coin_idx, num_samples=1):
    if coin_idx == 1:
        return np.random.binomial(1, p1, num_samples)
    else:
        return np.random.binomial(1, p2, num_samples)

# 1. Explore-then-commit strategy using N = ⌈0.2T⌉
def explore_then_commit_1(T):
    N = int(np.ceil(0.2 * T))  # exploration phase length
    
    # initialize
    rewards = np.zeros(T)
    coin1_rewards = []
    coin2_rewards = []
    
    # exploration phase: average try each coin
    for t in range(min(N, T)):
        if t % 2 == 0:  # even round try coin 1
            reward = simulate_coin(1)[0]
            coin1_rewards.append(reward)
        else:  # odd round try coin 2
            reward = simulate_coin(2)[0]
            coin2_rewards.append(reward)
        rewards[t] = reward
    
    # calculate the average reward of each coin
    if len(coin1_rewards) > 0 and len(coin2_rewards) > 0:
        mean_coin1 = np.mean(coin1_rewards)
        mean_coin2 = np.mean(coin2_rewards)
        
        # choose the coin with the higher average reward
        best_coin = 1 if mean_coin1 >= mean_coin2 else 2
    else:
        best_coin = 1  # default choose coin 1
    
    # exploitation phase: only use the best coin
    for t in range(N, T):
        reward = simulate_coin(best_coin)[0]
        rewards[t] = reward
    
    return np.cumsum(rewards)

# 2. Explore-then-commit strategy using N = ⌈(1/2)T^(2/3)(log T)^(1/3)⌉
def explore_then_commit_2(T):
    N = int(np.ceil(0.5 * (T**(2/3)) * (np.log(T)**(1/3))))  # exploration phase length
    
    # initialize
    rewards = np.zeros(T)
    coin1_rewards = []
    coin2_rewards = []
    
    # exploration phase: average try each coin
    for t in range(min(N, T)):
        if t % 2 == 0:  # even round try coin 1
            reward = simulate_coin(1)[0]
            coin1_rewards.append(reward)
        else:  # odd round try coin 2
            reward = simulate_coin(2)[0]
            coin2_rewards.append(reward)
        rewards[t] = reward
    
    # calculate the average reward of each coin
    if len(coin1_rewards) > 0 and len(coin2_rewards) > 0:
        mean_coin1 = np.mean(coin1_rewards)
        mean_coin2 = np.mean(coin2_rewards)
        
        # choose the coin with the higher average reward
        best_coin = 1 if mean_coin1 >= mean_coin2 else 2
    else:
        best_coin = 1  # default choose coin 1
    
    # exploitation phase: only use the best coin
    for t in range(N, T):
        reward = simulate_coin(best_coin)[0]
        rewards[t] = reward
    
    return np.cumsum(rewards)

# 3. ε-greedy strategy using ε=0.2
def epsilon_greedy(T, epsilon=0.2):
    # initialize
    rewards = np.zeros(T)
    coin1_rewards = []
    coin2_rewards = []
    coin1_pulls = 0
    coin2_pulls = 0
    
    # force each coin to pull at least once
    reward = simulate_coin(1)[0]
    rewards[0] = reward
    coin1_rewards.append(reward)
    coin1_pulls += 1
    
    if T > 1:
        reward = simulate_coin(2)[0]
        rewards[1] = reward
        coin2_rewards.append(reward)
        coin2_pulls += 1
    
    # for the remaining rounds
    for t in range(2, T):
        # calculate the average reward of each coin
        mean_coin1 = np.mean(coin1_rewards) if coin1_pulls > 0 else 0
        mean_coin2 = np.mean(coin2_rewards) if coin2_pulls > 0 else 0
        
        # choose the best coin with probability 1-ε, and choose a random coin with probability ε
        if np.random.random() > epsilon:
            # choose the coin with the highest reward
            coin = 1 if mean_coin1 >= mean_coin2 else 2
        else:
            # choose a random coin
            coin = np.random.choice([1, 2])
        
        # pull the selected coin and get the reward
        reward = simulate_coin(coin)[0]
        rewards[t] = reward
        
        # update the record
        if coin == 1:
            coin1_rewards.append(reward)
            coin1_pulls += 1
        else:
            coin2_rewards.append(reward)
            coin2_pulls += 1
    
    return np.cumsum(rewards)

# 4. Upper Confidence Bound strategy
def ucb(T):
    # initialize
    rewards = np.zeros(T)
    coin1_rewards = []
    coin2_rewards = []
    coin1_pulls = 0
    coin2_pulls = 0
    
        # force each coin to pull at least once
    reward = simulate_coin(1)[0]
    rewards[0] = reward
    coin1_rewards.append(reward)
    coin1_pulls += 1
    
    if T > 1:
        reward = simulate_coin(2)[0]
        rewards[1] = reward
        coin2_rewards.append(reward)
        coin2_pulls += 1
    
    # for the remaining rounds
    for t in range(2, T):
        # calculate the UCB value of each coin
        mean_coin1 = np.mean(coin1_rewards) if coin1_pulls > 0 else 0
        mean_coin2 = np.mean(coin2_rewards) if coin2_pulls > 0 else 0
        
        ucb1 = mean_coin1 + np.sqrt(2 * np.log(t) / coin1_pulls) if coin1_pulls > 0 else float('inf')
        ucb2 = mean_coin2 + np.sqrt(2 * np.log(t) / coin2_pulls) if coin2_pulls > 0 else float('inf')
        
        # select the coin with the highest UCB value
        coin = 1 if ucb1 >= ucb2 else 2
        
        # pull the selected coin and get the reward
        reward = simulate_coin(coin)[0]
        rewards[t] = reward
        
        # update the record
        if coin == 1:
            coin1_rewards.append(reward)
            coin1_pulls += 1
        else:
            coin2_rewards.append(reward)
            coin2_pulls += 1
    
    return np.cumsum(rewards)

# 主函数：运行多次模拟并绘制结果
def main():
    # 设置T的范围
    T_values = np.logspace(1, 3, 20).astype(int)  # 从10到1000的对数刻度
    
    # 存储不同策略的平均收益
    num_simulations = 50  # 模拟次数
    etc1_rewards = np.zeros((num_simulations, len(T_values)))
    etc2_rewards = np.zeros((num_simulations, len(T_values)))
    eg_rewards = np.zeros((num_simulations, len(T_values)))
    ucb_rewards = np.zeros((num_simulations, len(T_values)))
    
    # 运行多次模拟
    for sim in range(num_simulations):
        for i, T in enumerate(T_values):
            etc1_rewards[sim, i] = explore_then_commit_1(T)[-1]
            etc2_rewards[sim, i] = explore_then_commit_2(T)[-1]
            eg_rewards[sim, i] = epsilon_greedy(T)[-1]
            ucb_rewards[sim, i] = ucb(T)[-1]
            
        print(f"completed {sim+1}/{num_simulations}")
    
    # calculate the average reward
    avg_etc1_rewards = np.mean(etc1_rewards, axis=0)
    avg_etc2_rewards = np.mean(etc2_rewards, axis=0)
    avg_eg_rewards = np.mean(eg_rewards, axis=0)
    avg_ucb_rewards = np.mean(ucb_rewards, axis=0)
    
    # plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(T_values, avg_etc1_rewards, 'o-', label='Explore-then-commit N=⌈0.2T⌉')
    plt.plot(T_values, avg_etc2_rewards, 's-', label='Explore-then-commit N=⌈(1/2)T^(2/3)(log T)^(1/3)⌉')
    plt.plot(T_values, avg_eg_rewards, '^-', label='ε-greedy (ε=0.2)')
    plt.plot(T_values, avg_ucb_rewards, 'd-', label='Upper Confidence Bound')
    
    # add the reference line of the optimal strategy (always choose the best coin)
    plt.plot(T_values, 0.8 * np.array(T_values), '--', color='gray', label='Optimal (always choose coin 1)')
    
    plt.xscale('log')
    plt.xlabel('round T')
    plt.ylabel('total reward J(T)')
    plt.title('comparison of total rewards of different bandit strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/root/code/ucsd_hw/257_hw/hw3/figure/q6.png')
    plt.show()

if __name__ == "__main__":
    main()