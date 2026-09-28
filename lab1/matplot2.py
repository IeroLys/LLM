'''
- Сравните три вида применения функции к массиву поэлементно:
с помощью обычного python-цикла, при помощи np.vectorize
и при помощи специальной реализации функции,
используя средства numpy.

- Отобразите на графике зависимость времени работы
от размера входного массива. 
Размер массива удобно взять возрастающим по экспоненте.
Пример использования np.vectorize и реализации средствами numpy
был приведён на занятии.

- Сделайте правильные подписи к осям картинки.
'''

import numpy as np
import matplotlib.pyplot as plt
import time

def fun_python(x):
    return abs(x) * max(x, 5)

# np.vectorize
vectorized_fun = np.vectorize(fun_python)

# NumPy версия
def fun_np(x):
    return np.abs(x) * np.fmax(x, 5)


sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
times_python = []
times_vectorize = []
times_numpy = []

for size in sizes:
    arr = np.random.uniform(-10, 10, size) # создаём массив нужного размера
    
    # цикл
    start = time.perf_counter()
    res_py = [fun_python(x) for x in arr]
    times_python.append(time.perf_counter() - start)
    
    # np.vectorize
    start = time.perf_counter()
    res_vec = vectorized_fun(arr)
    times_vectorize.append(time.perf_counter() - start)
    
    # NumPy
    start = time.perf_counter()
    res_np = fun_np(arr)
    times_numpy.append(time.perf_counter() - start)


plt.figure(figsize=(10, 6))
plt.plot(sizes, times_python, marker='o', label='Python-цикл', linewidth=2)
plt.plot(sizes, times_vectorize, marker='s', label='np.vectorize', linewidth=2)
plt.plot(sizes, times_numpy, marker='^', label='NumPy', linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Размер входного массива', fontsize=12)
plt.ylabel('Время выполнения (логарифмическая шкала)', fontsize=12)
plt.title('Сравнение производительности поэлементных операций', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", linestyle='--', alpha=0.7)
plt.show()