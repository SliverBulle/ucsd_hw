from collections import deque
import unittest

def modified_dijkstra(graph, s, t):
    n = len(graph)
    dist = [float('inf')] * n
    dist[s] = 0
    Q1, Q2 = deque([s]), deque()

    while Q1 or Q2:
        if Q1:
            u = Q1.popleft()
        else:
            u = Q2.popleft()

        for v, weight in graph[u]:
            if weight == 1 and dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                Q1.append(v)
            elif weight == 2 and dist[v] > dist[u] + 2:
                dist[v] = dist[u] + 2
                Q2.append(v)

    return dist[t]

class TestModifiedDijkstra(unittest.TestCase):
    def test_simple_path(self):
        graph = [
            [(1, 1), (2, 2)],  # 节点 0 的邻居
            [(0, 1), (2, 1)],  # 节点 1 的邻居
            [(0, 2), (1, 1)]   # 节点 2 的邻居
        ]
        self.assertEqual(modified_dijkstra(graph, 0, 2), 2)

    def test_longer_path(self):
        graph = [
            [(1, 1), (2, 2)],  # 节点 0 的邻居
            [(0, 1), (3, 1)],  # 节点 1 的邻居
            [(0, 2), (3, 1)],  # 节点 2 的邻居
            [(1, 1), (2, 1), (4, 2)],  # 节点 3 的邻居
            [(3, 2)]  # 节点 4 的邻居
        ]
        self.assertEqual(modified_dijkstra(graph, 0, 4), 4)

    def test_unreachable_node(self):
        graph = [
            [(1, 1)],  # 节点 0 的邻居
            [(0, 1)],  # 节点 1 的邻居
            []  # 节点 2 没有邻居
        ]
        self.assertEqual(modified_dijkstra(graph, 0, 2), float('inf'))

    def test_all_weight_one(self):
        graph = [
            [(1, 1), (2, 1)],  # 节点 0 的邻居
            [(0, 1), (3, 1)],  # 节点 1 的邻居
            [(0, 1), (3, 1)],  # 节点 2 的邻居
            [(1, 1), (2, 1)]   # 节点 3 的邻居
        ]
        self.assertEqual(modified_dijkstra(graph, 0, 3), 2)

    def test_all_weight_two(self):
        graph = [
            [(1, 2), (2, 2)],  # 节点 0 的邻居
            [(0, 2), (3, 2)],  # 节点 1 的邻居
            [(0, 2), (3, 2)],  # 节点 2 的邻居
            [(1, 2), (2, 2)]   # 节点 3 的邻居
        ]
        self.assertEqual(modified_dijkstra(graph, 0, 3), 4)

if __name__ == '__main__':
    unittest.main()