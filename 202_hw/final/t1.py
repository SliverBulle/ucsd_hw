def maximize_weight(points, L):
    """
    :param points: List of tuples, where each tuple is (i, yi, wi)
    :param L: float, the minimum required difference in y-coordinate
    :return: Maximum total weight
    """
    n = len(points)
    # 按照点的索引排序，确保 i < j 时对应的索引关系
    points.sort(key=lambda x: x[0])
    
    # 初始化 DP 表
    dp = [[0] * 2 for _ in range(n + 1)]
    dp[1] = [points[0][2], 0]
    dp[2] = [points[1][2], 0]
    for i in range(1, n + 1):
        #print(i)
        _, y_i, w_i = points[i-1]
        
        # 枚举之前的点 j
        for j in range(1, i):
            _, y_j, _ = points[j-1]
            if abs(y_i - y_j) >= L:
                if i - j > 1:
                    dp[i][0] = max(dp[i][0], max(dp[j][0], dp[j][1]) + w_i)
                    print("i=",i,"j=",j,"y_i-y_j=",y_i-y_j,"dp[i][0]=",dp[i][0])
                elif i - j == 1:
                    dp[i][1] = dp[j][0] + w_i
                    print("i=",i,"j=",j,"y_i-y_j=",y_i-y_j,"dp[i][1]=",dp[i][1],"dp[j][0]=",dp[j][0])
    max_weight = 0
    for i in range(1, n + 1):
        max_weight = max(max_weight, max(dp[i][0], dp[i][1]))
    return max_weight


# 示例用法
if __name__ == "__main__":
    # 点的格式为 (索引, y坐标, 权重)
    points = [
        (1, 1.0, 1),
        (2, 4.0, 10),
        (3, 3.0, 2),
        (4, 6.0, 5.0),
        (5, 5.0, 30.5)
    ]
    L = 2.0
    max_total_weight = maximize_weight(points, L)
    print(f"最大总权重为: {max_total_weight}")