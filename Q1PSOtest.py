import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # 导入该函数是为了绘制3D图
import matplotlib as mpl
import time
time_start_1 = time.time()

# 数据导入
data_x = [60, 60, 20, 20, 80, 80, 0, 0]
data_y = [15, 35, 35, 15, 0, 50, 50, 0]
data_zuobiao = [[], [], [], [], [], [], [], []]
for i in range(len(data_x)):
    data_zuobiao[i] = (data_x[i], data_y[i])

# 导入边的起始点和终止点标号
Line = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [4, 7]])
data_xjk = np.zeros((7, 2))
data_yjk = np.zeros((7, 2))
for j in range(Line.shape[0]):
    for k in range(Line.shape[1]):
        data_xjk[j][k] = data_x[Line[j][k]]
        data_yjk[j][k] = data_y[Line[j][k]]
data_total = (data_xjk, data_yjk)
print(data_xjk)
print(data_total[0])
print(data_total[0][0])
print(data_total[0][0][0])

# 起始点设置
def startp(xjk,yjk,cij,pjk):
    X = np.zeros((cij.shape[0]))
    Y = np.zeros((cij.shape[0]))
    for i in range(cij.shape[0]):
        for j in range(xjk.shape[0]):
            for k in range(xjk.shape[1]):
                X[i] += xjk[j][k] * cij[i][j] * pjk[k][0][j]
                Y[i] += yjk[j][k] * cij[i][j] * pjk[k][0][j]
    data_Pi = [X,Y]
    return data_Pi


# 终止点设置
def startq(xjk,yjk,cij,qjk):
    XX = np.zeros((cij.shape[0]))
    YY = np.zeros((cij.shape[0]))
    for i in range(cij.shape[0]):
        for j in range(xjk.shape[0]):
            for k in range(xjk.shape[1]):
                XX[i] += xjk[j][k] * cij[i][j] * qjk[k][j]
                YY[i] += yjk[j][k] * cij[i][j] * qjk[k][j]
    data_Qi = [XX,YY]
    return data_Qi



# 提供坐标计算距离

def distance(X1, X2):
    dis = np.sqrt((X1[0] - X2[0]) ** 2 + (X1[1] - X2[1]) ** 2)  # 欧式距离计算
    return dis



# cc的变换
def switch_line(shunxv_line,w,c1,c2):
    num_line = len(shunxv_line)
    n = w
    pp = random.uniform(0, sum([w, c1, c2]))
    if pp <= n:
        random.shuffle(shunxv_line)
    else:
        switch_num = 5
        switch = random.randint(0,switch_num)
        for s in range(switch):
            k = random.randint(0, num_line - 1)
            kk = random.randint(k, num_line - 1)
            shunxv_line[k], shunxv_line[kk] = shunxv_line[kk], shunxv_line[k]
    return shunxv_line


# 计算cij
def ctocij(cc):
    c = np.zeros((len(cc), len(cc)))
    for i in range(c.shape[0]):
        c[i][cc[i]] = 1
    cij = copy.deepcopy(c)
    return cij

# 顺序的初始化
num_line = 7
shunxv_line = list(range(0, num_line))
random.shuffle(shunxv_line)
PB = shunxv_line
print(PB)

cc = [4, 5, 1, 0, 3, 2, 6]
cij = ctocij(cc)


# pjk转化为qjk
def pjktoqjk(pjk):
    qjk = np.zeros((len(pjk),pjk[0][0].shape[0]))
    for j in range(pjk[0][0].shape[0]):
        for k in range(len(pjk)):
            qjk[k][j] =1 - pjk[k][0][j]
    return qjk

#qjk = pjktoqjk(pjk)

# 离散粒子群
# 定义适应度函数
def funcp(pjk,cij):
    data_P = startp(data_xjk, data_yjk, cij, pjk)
    data_Q = startq(data_xjk, data_yjk, cij, pjktoqjk(pjk))
    distance1 = 0
    distance2 = 0
    distance3 = 0
    for i in range(data_P[0].shape[0]-1):
        distance1 += distance([data_P[0][i + 1], data_P[1][i + 1]], [data_Q[0][i], data_Q[1][i]])
        distance2 = distance([data_P[0][0], data_P[1][0]], data_zuobiao[4])
        distance3 = distance(data_zuobiao[4], [data_Q[0][-1], data_Q[1][-1]])
    dis_zong = distance1 + distance2 + distance3 - 32
    return dis_zong


# 把无效解转化为有效解
def callback(x):
    for j in range(len(x[0][0])):
        d = 0
        for k in range(len(x)):
            d += x[k][0][j]
        if d == 2:
            n = random.randint(0,10)
            pp = 5
            if n > pp:
                x[0][0][j] = 0
            else:
                x[1][0][j] = 0
        elif d == 0:
            n = random.randint(0,10)
            pp = 5
            if n > pp:
                x[0][0][j] = 1
            else:
                x[1][0][j] = 1
    return x


# 设置字体和设置负号
matplotlib.rc("font", family="KaiTi")
matplotlib.rcParams["axes.unicode_minus"] = False
# 初始化种群，群体规模，每个粒子的速度和规模
N = 60 # 种群数目
D = 14 # 维度
T = 2000 # 最大迭代次数
c1 = 0.4 # 个体学习因子
c2 = 0.4 # 群体学习因子
w_max = 0.5 # 权重系数最大值
w_min = 0.2 # 权重系数最小值
x_max = 9 # 每个维度最大取值范围，如果每个维度不一样，那么可以写一个数组，下面代码依次需要改变
x_min = 0 # 同上
v_max = 10 # 每个维度粒子的最大速度
v_min = -10 # 每个维度粒子的最小速度
c1_max = 0.8 # c2系数最大值
c1_min = 0.3 # c1系数最小值
c2_max = 0.3 # c2系数最大值
c2_min = 0.8 # c2系数最小值
# 顺序的初始化
num_line = 7
shunxv_line = list(range(0, num_line))
random.shuffle(shunxv_line)
PB = shunxv_line
print(PB)

# 初始化种群个体
x = np.random.randint(0,2,[N, D]) # 初始化每个粒子的位置
v = (v_max - v_min) * np.random.rand(N,D) + v_min # 初始化每个粒子的速度
vx = np.zeros_like(v)

# 初始化每个粒子的适应度值
p = x # 用来存储每个粒子的最佳位置
p_best = np.ones(N) # 用来存储每个粒子的适应度值
knn = [[x[1,0:7]],[x[1,7:14]]]
print(knn[0][0])
print(knn[0][0][0])
print(knn[0][0].shape[0])

for i in range(N):
    p_best[i] = funcp([[x[i,0:7]],[x[i,7:14]]],ctocij(PB))
#     p[i,:] = x[j,:]

g_best = 400# 设置全局最优值
gb = np.ones(T)  # 用来存储每依次迭代的最优值
x_best = np.ones(D)# 存储最优粒子的取值
# 初始化全局最优位置与最优值
for i in range(N):
    if p_best[i] < g_best:
        g_best = p_best[i]
        x_best = x[i,:].copy()


for i in range(T):
    for j in range(N):
        # 更新每个个体最优值和最优位置
        if p_best[j] > funcp(callback([[x[j,0:7]],[x[j,7:14]]]),ctocij(PB)):
            p_best[j] = funcp(callback([[x[j,0:7]],[x[j,7:14]]]),ctocij(PB))
            p[j, :] = x[j, :].copy()
        # 更新全局最优位置和最优值
        if p_best[j] < g_best:
            g_best = p_best[j]
            x_best = x[j, :].copy()
            PB_best = PB.copy()
        # 计算动态惯性权重
        w = w_max - (w_max - w_min) * i / T
        c1 = c1_max - (c1_max - c1_min) * i / T
        c2 = c2_max - (c2_max - c2_min) * i / T
        # 更新速度, 因为位置需要后面进行概率判断更新
        v[j, :] = w * v[j, :] + c1 * np.random.rand(1) * (p_best[j] - x[j, :]) + c2 * np.random.rand(1) * (
                    x_best - x[j, :])
        # 边界条件处理
        for jj in range(D):
            if (v[j, jj] > v_max) or (v[j, jj] < v_min):
                v[j, jj] = v_min + np.random.rand(1) * (v_max - v_min)
        # 进行概率计算并且更新位置
        vx[j, :] = 1 / (1 + np.exp(-v[j, :]))
        for ii in range(D):
            x[j, ii] = 1 if vx[j, ii] > np.random.rand(1) else 0
        # 更新顺序
        PB = switch_line(PB,w,c1,c2)
    gb[i] = g_best

print("最优值为", gb[T - 1],"最优位置为",format(x_best),"最佳顺序为",PB_best)

#结果输出
def result_out(x_best, PB_best, L):
    result_L = copy.deepcopy(L+1)
    result_x = np.zeros((int(len(PB_best)),int(len(x_best)/len(PB_best))))
    for k in range(int(len(x_best) / len(PB_best))):
        for j in range(len(PB_best)):
            result_x[j][k] = x_best[j+k*len(PB_best)]

    for j in range(len(PB_best)):
        if result_x[j][0] == 1:
            result_L[j][0], result_L[j][1] = result_L[j][0], result_L[j][1]
        else:
            result_L[j][0], result_L[j][1] = result_L[j][1], result_L[j][0]

    result = [[]] * len(PB_best)
    for i in range(len(PB_best)):
        result[i] = result_L[PB_best[i]]

    print(result_x)
    print(L)
    print(result_L)
    print(result)
    return result


print(result_out(x_best,PB_best,Line))

# code
time_end_1 = time.time()
print("运行时间："+str(time_end_1 - time_start_1)+"秒")

#迭代曲线图
plt.plot(range(T),gb)
plt.xlabel("迭代次数")
plt.ylabel("适应度值")
plt.title("适应度进化曲线")
plt.show()