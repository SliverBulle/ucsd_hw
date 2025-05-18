import numpy as np
import matplotlib.pyplot as plt

#   
states = ['s1', 's2']
actions = ['a1', 'a2', 'a3']
rewards = {'s1': -10, 's2': 10}
gamma = 0.6

# define the transition probability

P = {
    's1': {
        'a1': {'s1': 0.2, 's2': 0.8},
        'a2': {'s1': 0.4, 's2': 0.6},
        'a3': {'s1': 0.0, 's2': 1.0}
    },
    's2': {
        'a1': {'s1': 0.1, 's2': 0.9},
        'a2': {'s1': 0.0, 's2': 0.0},   
        'a3': {'s1': 0.5, 's2': 0.5}
    }
}

def bellman_update(V):

    V_new = {}
    
    for s in states:
        max_value = float('-inf')
        
        for a in actions:
            action_value = 0
            
            # if the transition probability of the current state and action exists
            if a in P[s]:
                for next_s, prob in P[s][a].items():
                    # calculate the expected reward of the next state
                    action_value += prob * (rewards[next_s] + gamma * V[next_s])
            
            max_value = max(max_value, action_value)
        
        V_new[s] = max_value
    
    return V_new

def value_iteration(V_init, epsilon=0.1):

    V = V_init.copy()
    iterations = 0
    trajectory = [list(V.values())]  # record the trajectory of the value
    
    while True:
        V_new = bellman_update(V)
        iterations += 1
        
        # calculate the maximum change
        max_diff = max(abs(V_new[s] - V[s]) for s in states)
        
        # record the current value
        trajectory.append(list(V_new.values()))
        
        # update the value function
        V = V_new.copy()
        
        # check convergence condition: if the maximum change is less than epsilon * (1-gamma) / gamma, then converge
        # according to the theory of value iteration, the distance between the value function and the optimal value function is no more than epsilon
        if max_diff < epsilon * (1 - gamma):
            break
    
    return V, iterations, np.array(trajectory)

# initial value vector
V_A = {'s1': 0, 's2': 0}
V_B = {'s1': 100, 's2': 100}

# run value iteration, starting from two different initial points
V_final_A, iterations_A, trajectory_A = value_iteration(V_A)
V_final_B, iterations_B, trajectory_B = value_iteration(V_B)

print(f"从初始点 V_A = (0, 0) 开始:")
print(f"收敛后的值: {V_final_A}")
print(f"迭代次数: {iterations_A}")

print(f"\n从初始点 V_B = (100, 100) 开始:")
print(f"收敛后的值: {V_final_B}")
print(f"迭代次数: {iterations_B}")

# 绘制值迭代轨迹
plt.figure(figsize=(12, 5))

# the first initial point
plt.subplot(1, 2, 1)
plt.plot(trajectory_A[:, 0], trajectory_A[:, 1], 'bo-', label='Value Iteration Path')
plt.scatter(trajectory_A[0, 0], trajectory_A[0, 1], c='red', s=100, label='Initial Value $V_A$')
plt.scatter(trajectory_A[-1, 0], trajectory_A[-1, 1], c='green', s=100, label='Converged Value')
plt.xlabel('V(s1)')
plt.ylabel('V(s2)')
plt.title(f'Value Iteration from $V_A$ = (0, 0)\nConverged in {iterations_A} iterations')
plt.grid(True)
plt.legend()


# the second initial point
plt.subplot(1, 2, 2)
plt.plot(trajectory_B[:, 0], trajectory_B[:, 1], 'bo-', label='Value Iteration Path')
plt.scatter(trajectory_B[0, 0], trajectory_B[0, 1], c='red', s=100, label='Initial Value $V_B$')
plt.scatter(trajectory_B[-1, 0], trajectory_B[-1, 1], c='green', s=100, label='Converged Value')
plt.xlabel('V(s1)')
plt.ylabel('V(s2)')
plt.title(f'Value Iteration from $V_B$ = (100, 100)\nConverged in {iterations_B} iterations')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('/root/code/ucsd_hw/257_hw/hw3/figure/part1_B.png')
plt.close()
