import numpy as np

arr = np.array([0, 3, 0, 0, 7, 2])
mymask = arr[:-1] == 0
cands = arr[1:]
nums_to_sum = cands[mymask]
# print(np.max(nums_to_sum))

X = [[5, 15], [25, 35]]
value = 20
X2 = np.array(X)
X2 = np.abs(X2 - value)
min_num = np.argmin(X2)
#print(min_num)

# на вход двумерный массив (p, k)
def fun(x):
    return np.sum(x, axis=1) + np.trace(x @ x.T)

# матрица (n, m, k)
data = np.random.normal(0, 5, size=(10,20,5))
print(data)