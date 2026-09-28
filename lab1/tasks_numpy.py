'''
Задания подразумевают, что писать циклы нельзя, необходимо использовать способности библиотеки numpy.

- Реализуйте функцию, возвращающую максимальный элемент в массиве X среди элементов,
перед которыми стоит нулевой.

- Реализуйте функцию, принимающую на вход матрицу X
и некоторое число value и возвращающую ближайший к числу ЭЛЕМЕНТ МАТРИЦЫ.

- Дана функция, обрабатывающая список объектов формы (p, k).
А у вас есть матрица (n,m,k)
Вычислите функцию относительно этой матрицы.
Код функции менять нельзя, нужно преобразовать матрицу.
Результат должен быть формы (n,m).  

- У вас есть массив точек в виде матрицы размерности (n, d), n - количество точек,
d - размерность каждой точки. Вычислите попарные расстояние между всеми точками - матрицу (n,n).
Используется евклидово расстояние.
Попробуйте воспользоваться трюком с расширением размерности матрицы.
'''

import numpy as np

'''задание 1'''
def max_el(X):
    mask = X[:-1] == 0
    cands = X[1:][mask]
    
    if len(cands) == 0:
        return np.nan
        
    return np.max(cands)

print('задание 1')
X = np.array([1, 0, 5, 2, 0, 8, 3, 0, 1])
print(*X)
print('максимальный элемент после 0:', max_el(X)) # 8




'''задание 2'''
def nearest_num(X, value):
    min_num = np.argmin(np.abs(X - value)) # индекс минимального близкого
    i, j = np.unravel_index(min_num, X.shape)
    return X[i, j]

print('Задание 2')
X = np.array([[10, 20, 30], 
              [40, 55, 60]])
value = 50
print("Ближайший элемент:", nearest_num(X, value)) # 55




    
'''задание 3'''
# на вход двумерный массив (p, k)
def fun(x):
    return np.sum(x, axis=1) + np.trace(x @ x.T)

# матрица (n, m, k)
data = np.random.normal(0, 5, size=(10,20,5))

def transform_data(data):
    n, m, k = data.shape
    # из 3 в 2
    data_reshaped = data.reshape(n * m, k)
    # 1
    result = fun(data_reshaped)
    # 1 в 2
    return result.reshape(n, m)


print('Задание 3')
data = np.random.normal(0, 5, size=(10, 20, 5))
result = transform_data(data)
print("Исходная форма:", data.shape)
print("Форма результата:", result.shape) # (10, 20)






'''задание 4'''
'''
X = [[0, 0], [1, 0]]
[[0.0, 1.0],
 [1.0, 0.0]]
'''
def distances(X):
    
    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distances

print('Задание 4')
n, d = 4, 2
X = np.array([[0, 0], 
              [1, 0], 
              [0, 1], 
              [1, 1]])
dist_matrix = distances(X)
print("Матрица расстояний:\n", np.round(dist_matrix, 2))