'''
Вам даны два набора точек в двумерном пространстве.
- Найдите центры масс этих наборов,
соедините их прямой, постройте срединный перпендикуляр.
- Постройте проекции точек на срединный перпендикуляр.
- Подпишите координаты центров наборов точек.

X1 = np.random.multivariate_normal([4,4],[[0.1,-0.4],[-0.4,3]], size = 10)
plt.scatter(X1[:,0], X1[:,1])

X2 = np.random.multivariate_normal([8,4],[[0.2,0.4],[0.4,1]], size = 10)
plt.scatter(X2[:,0], X2[:,1])
'''


import numpy as np
import matplotlib.pyplot as plt

X1 = np.random.multivariate_normal([4, 4], [[0.1, -0.4], [-0.4, 3]], size=10)
X2 = np.random.multivariate_normal([8, 4], [[0.2, 0.4], [0.4, 1]], size=10)


#1
# C1 = ((3+4+5)/3, (5+4+3)/3) = (4, 4)
C1 = np.mean(X1, axis=0)
C2 = np.mean(X2, axis=0)


#2
# вектор между центрами и середина отрезка
V = C2 - C1
M = (C1 + C2) / 2
# направление срединного перпендикуляра
V_perp = np.array([-V[1], V[0]])
V_perp_norm = V_perp / np.linalg.norm(V_perp) # нормируем до 1

t = np.linspace(-4, 4, 100)
perp_x = M[0] + t * V_perp_norm[0]
perp_y = M[1] + t * V_perp_norm[1]


#3
def project_points(X, M, u):
    vec = X - M  # вектор от середины к точке
    proj_scalar = np.dot(vec, u) # расстояние до тени
    return M + np.outer(proj_scalar, u) # координаты тени

proj_X1 = project_points(X1, M, V_perp_norm)
proj_X2 = project_points(X2, M, V_perp_norm)


#4
plt.figure(figsize=(10, 8))
plt.scatter(X1[:, 0], X1[:, 1], color='purple', label='Набор X1', s=50)
plt.scatter(X2[:, 0], X2[:, 1], color='orange', label='Набор X2', s=50)
# центры масс
plt.plot([C1[0], C2[0]], [C1[1], C2[1]], 'k--', label='Отрезок C1-C2')
# срединный перпендикуляр
plt.plot(perp_x, perp_y, 'g-', linewidth=2, label='Срединный перпендикуляр')
# проекции
plt.scatter(proj_X1[:, 0], proj_X1[:, 1], color='purple', marker='x', s=80, label='Проекции X1')
plt.scatter(proj_X2[:, 0], proj_X2[:, 1], color='orange', marker='x', s=80, label='Проекции X2')
# подписи
plt.annotate(f'C1 ({C1[0]:.2f}, {C1[1]:.2f})', xy=C1, xytext=(C1[0]-1.5, C1[1]+0.8),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
plt.annotate(f'C2 ({C2[0]:.2f}, {C2[1]:.2f})', xy=C2, xytext=(C2[0]+0.5, C2[1]+0.8),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

plt.legend()
plt.grid(True, linestyle=':')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.title('Центры масс, срединный перпендикуляр и проекции точек')
plt.axis('equal')
plt.show()