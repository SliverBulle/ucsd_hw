class PickingTeams:
    def __init__(self, n, m, a, b, friendships):
        self.n = n  # Number of people
        self.m = m  # Number of friendship pairs
        self.a = a  # Cost to assign to team A
        self.b = b  # Cost to assign to team B
        self.friendships = friendships  # List of (u, v, c) tuples

        self.graph = [[] for _ in range(n + 2)]  # Graph represented as adjacency list
        self.S = n  # Source node index
        self.T = n + 1  # Sink node index

    def add_edge(self, u, v, capacity):
        # Add forward edge
        forward = {'to': v, 'capacity': capacity, 'rev': len(self.graph[v])}
        self.graph[u].append(forward)
        # Add backward edge
        backward = {'to': u, 'capacity': 0, 'rev': len(self.graph[u]) - 1}
        self.graph[v].append(backward)

    def build_graph(self):
        # Connect source and sink
        for i in range(self.n):
            self.add_edge(self.S, i, self.a[i])  # Source to person
            self.add_edge(i, self.T, self.b[i])  # Person to sink

        # Connect friendships
        for u, v, c in self.friendships:
            self.add_edge(u, v, c)
            self.add_edge(v, u, c)

    def bfs_level(self, level):
        from collections import deque
        queue = deque()
        queue.append(self.S)
        level[self.S] = 0
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge['capacity'] > 0 and level[edge['to']] == -1:
                    level[edge['to']] = level[u] + 1
                    queue.append(edge['to'])
        return level[self.T] != -1

    def dfs_flow(self, u, flow, level, ptr):
        if u == self.T:
            return flow
        while ptr[u] < len(self.graph[u]):
            edge = self.graph[u][ptr[u]]
            v = edge['to']
            if edge['capacity'] > 0 and level[v] == level[u] + 1:
                pushed = self.dfs_flow(v, min(flow, edge['capacity']), level, ptr)
                if pushed > 0:
                    edge['capacity'] -= pushed
                    self.graph[v][edge['rev']]['capacity'] += pushed
                    return pushed
            ptr[u] += 1
        return 0

    def max_flow(self):
        flow = 0
        level = [-1] * (self.n + 2)
        while True:
            level = [-1] * (self.n + 2)
            if not self.bfs_level(level):
                break
            ptr = [0] * (self.n + 2)
            while True:
                pushed = self.dfs_flow(self.S, float('inf'), level, ptr)
                if pushed == 0:
                    break
                flow += pushed
        return flow, level

    def min_cut(self):
        self.build_graph()
        max_flow, level = self.max_flow()
        # BFS to find reachable vertices from S
        from collections import deque
        visited = [False] * (self.n + 2)
        queue = deque()
        queue.append(self.S)
        visited[self.S] = True
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge['capacity'] > 0 and not visited[edge['to']]:
                    visited[edge['to']] = True
                    queue.append(edge['to'])
        # Assign teams based on visited
        team_A = []
        team_B = []
        for i in range(self.n):
            if visited[i]:
                team_A.append(i)
            else:
                team_B.append(i)
        return team_A, team_B, max_flow, level

    def solve(self):
        team_A, team_B, total_cost, level = self.min_cut()
        return team_A, team_B, total_cost, level


def main():
    import sys

    # 示例输入格式
    # 第一行: n m
    # 第二行: a1 a2 ... an
    # 第三行: b1 b2 ... bn
    # 接下来的 m 行: u v c

    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))
    friendships = []
    for _ in range(m):
        u, v, c = map(int, sys.stdin.readline().split())
        friendships.append((u, v, c))

    picker = PickingTeams(n, m, a, b, friendships)
    team_A, team_B, total_cost, level = picker.solve()

    print("团队 A 成员:", team_A)
    print("团队 B 成员:", team_B)
    print("总成本:", total_cost)
    print("层级:", level)

if __name__ == "__main__":
    main()