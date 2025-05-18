import heapq

def alien_invasion(n, edges, c):
    """
    :param n: int, number of cities
    :param edges: List of tuples, each tuple is (u, v, l)
    :param c: int, energy cost for teleportation
    :return: int, minimum total energy cost
    """
    # 构建邻接表
    graph = [[] for _ in range(n + 1)]
    for u, v, l in edges:
        graph[u].append((v, l))
        graph[v].append((u, l))
    
    # 初始化
    visited = [False] * (n + 1)
    min_heap = []
    total_cost = 0
    
    # 维护已建传送站的集合
    teleport_set = set()
    
    # 从城市1开始，建立传送站
    teleport_set.add(1)
    visited[1] = True
    total_cost += c  # 第一次传送成本
    
    # 将城市1的所有连接加入堆
    for neighbor, l in graph[1]:
        heapq.heappush(min_heap, (min(l, c), neighbor))
    
    while len(teleport_set) < n:
        if not min_heap:
            break
        cost, city = heapq.heappop(min_heap)
        if not visited[city]:
            visited[city] = True
            teleport_set.add(city)
            total_cost += cost
            # 将新加入的城市的所有连接加入堆
            for neighbor, l in graph[city]:
                if not visited[neighbor]:
                    heapq.heappush(min_heap, (min(l, c), neighbor))
    
    return total_cost

# 示例用法
if __name__ == "__main__":
    # 输入示例
    n = 5  # 城市数量
    edges = [
        (1, 2, 4),
        (1, 3, 2),
        (2, 3, 1),
        (2, 4, 5),
        (3, 4, 8),
        (4, 5, 3)
    ]
    c = 3  # 传送能量成本
    print(alien_invasion(n, edges, c))  # 输出最小总能量成本