import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

# 1. Определение функции и генерация данных
def true_function(x):
    return x**2 + np.sin(3 * x)

np.random.seed(42)
# Тренировочная выборка: 10 точек + шум
x_train = np.random.uniform(-3, 3, 10)
y_train = true_function(x_train) + np.random.normal(0, 1.5, 10)

# Тестовая выборка: 500 точек + шум
x_test = np.random.uniform(-3, 3, 500)
y_test = true_function(x_test) + np.random.normal(0, 1.5, 500)

X_train = x_train.reshape(-1, 1)
X_test = x_test.reshape(-1, 1)

# 2. Полиномиальные признаки 8-й степени и их масштабирование
degree = 8
poly = PolynomialFeatures(degree=degree, include_bias=False)
scaler = StandardScaler()

X_train_poly = scaler.fit_transform(poly.fit_transform(X_train))
X_test_poly = scaler.transform(poly.transform(X_test))

# 3. Перебор коэффициента регуляризации alpha
alphas = np.logspace(-4, 3, 100)
lasso_errors, ridge_errors = [], []
lasso_weights, ridge_weights = [], []

for alpha in alphas:
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_train_poly, y_train)
    lasso_errors.append(mean_squared_error(y_test, lasso.predict(X_test_poly)))
    lasso_weights.append(lasso.coef_)
    
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_poly, y_train)
    ridge_errors.append(mean_squared_error(y_test, ridge.predict(X_test_poly)))
    ridge_weights.append(ridge.coef_)

# 4. Визуализация
plt.figure(figsize=(14, 5))

# График 1: Ошибка от alpha
plt.subplot(1, 2, 1)
plt.plot(alphas, lasso_errors, label='Lasso (Test MSE)', color='orange')
plt.plot(alphas, ridge_errors, label='Ridge (Test MSE)', color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Коэффициент регуляризации ($\\alpha$)')
plt.ylabel('Ошибка (MSE) на тесте')
plt.title('Зависимость ошибки от коэффициента регуляризации')
plt.legend()
plt.grid(True, which="both", ls="--")

# График 2: Веса от alpha
plt.subplot(1, 2, 2)
for i in range(degree):
    plt.plot(alphas, [w[i] for w in lasso_weights], color='orange', alpha=0.6)
    plt.plot(alphas, [w[i] for w in ridge_weights], color='blue', alpha=0.6, ls='--')
    
plt.xscale('log')
plt.xlabel('Коэффициент регуляризации ($\\alpha$)')
plt.ylabel('Значение веса')
plt.title('Зависимость весов моделей от $\\alpha$')
plt.axhline(0, color='black', linewidth=1)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()