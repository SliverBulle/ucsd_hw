import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class SafeDrivingEnvironment:
    def __init__(self):
        # 定义环境参数
        self.lanes = [4, 5]  # 垂直车道的列索引
        
        # 定义特殊位置
        self.start_positions = [(0, lane) for lane in self.lanes]
        self.d0_position = (2, 4)  # D0位置
        
        # 危险区域D1-D4（行和列的交叉点）
        self.danger_zones = {
            "D1": (4, 4),
            "D2": (4, 5),
            "D3": (6, 4),
            "D4": (6, 5)
        }
        
        # 行人行
        self.l_row = 4  # L-Row的行索引
        self.r_row = 6  # R-Row的行索引
        
        # 停车位置P1和P2
        self.parking_positions = [(8, 4), (8, 5)]  # P1, P2
        
        # 定义动作空间
        self.actions = {
            0: "不移动",
            1: "向下移动1格",
            2: "向下移动2格",
            3: "换道并向下移动1格"
        }
        
        # 定义奖励
        self.collision_penalty = -100  # 碰撞惩罚
        self.goal_reward = 100  # 到达停车位奖励
        self.step_penalty = -1  # 每步的小惩罚
        
        # 定义折扣因子
        self.gamma = 0.9
        
    def reset(self):
        # 随机选择一个起始位置
        self.car_position = random.choice(self.start_positions)
        
        # 初始化行人位置（随机）
        self.l_pedestrian = (self.l_row, random.randint(0, 7))
        self.r_pedestrian = (self.r_row, random.randint(0, 7))
        
        return self._get_state()
    
    def _get_state(self):
        # 状态表示为(car_row, car_col, l_ped_col, r_ped_col)
        return (self.car_position[0], self.car_position[1], 
                self.l_pedestrian[1], self.r_pedestrian[1])
    
    def get_specific_init_state(self):
        # 根据图中所示设置特定的初始状态
        # 汽车在D0位置，行人位置如图中所示
        self.car_position = self.d0_position
        # 根据图中所示，L-Row的行人在E2位置，R-Row的行人在E1位置
        self.l_pedestrian = (self.l_row, 5)  # E2对应的列索引
        self.r_pedestrian = (self.r_row, 2)  # E1对应的列索引
        return self._get_state()
    
    def step(self, action):
        # 获取当前汽车位置
        car_row, car_col = self.car_position
        
        # 执行动作
        if action == 0:  # 不移动
            new_car_position = (car_row, car_col)
            car_occupies = [new_car_position]
        elif action == 1:  # 向下移动1格
            new_car_position = (car_row + 1, car_col)
            car_occupies = [new_car_position]
        elif action == 2:  # 向下移动2格
            new_car_position = (car_row + 2, car_col)
            # 当车辆以2格速度移动时，中间格也被占用
            car_occupies = [(car_row + 1, car_col), new_car_position]
        elif action == 3:  # 换道并向下移动1格
            new_col = self.lanes[0] if car_col == self.lanes[1] else self.lanes[1]
            new_car_position = (car_row + 1, new_col)
            car_occupies = [new_car_position]
        
        # 检查是否越界
        if new_car_position[0] < 0 or new_car_position[0] > 8 or \
           new_car_position[1] < 0 or new_car_position[1] > 7:
            return self._get_state(), self.collision_penalty, True
        
        # 更新汽车位置
        self.car_position = new_car_position
        
        # 移动行人（1或2格，各50%概率）
        l_ped_move = random.choice([1, 2])
        r_ped_move = random.choice([1, 2])
        
        # L-Row行人向左移动（环绕式）
        self.l_pedestrian = (self.l_row, (self.l_pedestrian[1] - l_ped_move) % 8)
        
        # R-Row行人向右移动（环绕式）
        self.r_pedestrian = (self.r_row, (self.r_pedestrian[1] + r_ped_move) % 8)
        
        # 检查是否与行人碰撞
        for pos in car_occupies:
            if (pos[0] == self.l_row and pos[1] == self.l_pedestrian[1]) or \
               (pos[0] == self.r_row and pos[1] == self.r_pedestrian[1]):
                return self._get_state(), self.collision_penalty, True
        
        # 检查是否到达停车位
        if self.car_position in self.parking_positions:
            return self._get_state(), self.goal_reward, True
        
        # 继续游戏
        return self._get_state(), self.step_penalty, False

class QLearning:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, 
                 min_exploration_rate=0.01, exploration_decay_rate=0.001):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        
        # 初始化Q表
        # 状态为(car_row, car_col, l_ped_col, r_ped_col)
        # 动作有4种
        self.q_table = {}
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state][action]
    
    def update_q_value(self, state, action, reward, next_state):
        # 获取当前Q值
        current_q = self.get_q_value(state, action)
        
        # 计算最大下一个Q值
        next_max_q = np.max([self.get_q_value(next_state, a) for a in range(4)])
        
        # 计算新的Q值
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q)
        
        # 更新Q表
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        self.q_table[state][action] = new_q
        
    def choose_action(self, state):
        # 探索或利用
        if random.random() < self.exploration_rate:
            return random.randint(0, 3)
        else:
            # 获取状态对应的Q值
            if state not in self.q_table:
                self.q_table[state] = np.zeros(4)
            
            # 返回最大Q值对应的动作
            return np.argmax(self.q_table[state])
    
    def train(self, num_episodes=50000, max_steps_per_episode=100):
        for episode in tqdm(range(num_episodes)):
            # 重置环境
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, done = self.env.step(action)
                
                # 更新Q值
                self.update_q_value(state, action, reward, next_state)
                
                # 更新状态
                state = next_state
                steps += 1
            
            # 更新探索率
            self.exploration_rate = max(
                self.min_exploration_rate, 
                self.exploration_rate * np.exp(-self.exploration_decay_rate * episode)
            )
    
    def train_for_specific_state(self, specific_state, num_episodes=2000, max_steps_per_episode=50):
        """为特定状态进行更集中的训练"""
        print("为特定状态进行额外训练...")
        
        for _ in tqdm(range(num_episodes)):
            # 设置环境为特定状态
            self.env.car_position = (specific_state[0], specific_state[1])
            self.env.l_pedestrian = (self.env.l_row, specific_state[2])
            self.env.r_pedestrian = (self.env.r_row, specific_state[3])
            
            state = specific_state
            done = False
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, done = self.env.step(action)
                
                # 更新Q值
                self.update_q_value(state, action, reward, next_state)
                
                # 更新状态
                state = next_state
                steps += 1
                
                # 如果达到了终止状态，重置回特定状态再继续
                if done:
                    self.env.car_position = (specific_state[0], specific_state[1])
                    self.env.l_pedestrian = (self.env.l_row, specific_state[2])
                    self.env.r_pedestrian = (self.env.r_row, specific_state[3])
                    state = specific_state
                    done = False

# 主程序
def main():
    # 设置随机种子以便结果可复现
    np.random.seed(42)
    random.seed(42)
    
    # 创建环境
    env = SafeDrivingEnvironment()
    
    # 创建Q学习代理
    agent = QLearning(env, learning_rate=0.1, discount_factor=0.9, 
                      exploration_rate=1.0, exploration_decay_rate=0.001)
    
    # 训练代理
    print("训练Q学习代理...")
    agent.train(num_episodes=50000)
    
    # 获取图中所示的特定状态
    specific_state = env.get_specific_init_state()
    
    # 计算并显示D0位置（图中所示状态）的Q值
    print(f"\nD0位置的Q值（行人位置如图所示 - 状态: {specific_state}）:")
    if specific_state in agent.q_table:
        for action in range(4):
            print(f"动作 '{env.actions[action]}' 的Q值: {agent.q_table[specific_state][action]:.4f}")
    else:
        print("没有为该状态学习Q值，正在为此状态进行额外训练...")
        
        # 为特定状态进行额外训练
        agent.train_for_specific_state(specific_state)
        
        # 再次显示Q值
        print(f"\n额外训练后D0位置的Q值:")
        for action in range(4):
            print(f"动作 '{env.actions[action]}' 的Q值: {agent.q_table[specific_state][action]:.4f}")
    
    # 找出最佳动作
    best_action = np.argmax(agent.q_table[specific_state])
    print(f"\n最佳动作: {env.actions[best_action]}")

if __name__ == "__main__":
    main() 