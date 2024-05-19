import matplotlib.pyplot as plt

# 数据
n = [10, 20, 30, 40, 50, 60, 70]
cpu_time = [0.005653, 0.010124, 0.275588, 1.804581, 11.08303, 27.66342, 700.1828]
gpu_time = [0.748869, 0.832359, 1.535779, 2.048215, 2.650429, 4.292189, 65.10023]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制CPU和GPU时间的对数图
plt.plot(n, cpu_time, marker='o', label='CPU Time', color='b')
plt.plot(n, gpu_time, marker='o', label='GPU Time', color='r')

# 设置对数刻度
#plt.yscale('log')

# 添加标题和标签
plt.title('CPU vs GPU Computation Time')
plt.xlabel('Problem Size (n)')
plt.ylabel('Time (seconds, log scale)')
plt.legend()

# 显示图形
plt.grid(True, which="both", ls="--")
plt.show()
