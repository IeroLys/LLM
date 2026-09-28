import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Загрузка данных
df = pd.read_csv('lab2/hw2_polynomial_data.csv')
X = df[['x0', 'x1', 'x2']].values
y = df['y'].values

# 2. Генерация полиномиальных признаков (степень 3, так как степень 2 дает низкое качество R^2 ~ 0.23)
poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(['x0', 'x1', 'x2'])

# 3. Масштабирование признаков (критически важно для корректной работы L1-регуляризации)
scaler = StandardScaler()
X_poly_scaled = X_poly.copy()
X_poly_scaled[:, 1:] = scaler.fit_transform(X_poly[:, 1:]) # bias не масштабируем

# 4. Подбор параметра alpha для Lasso
alphas = np.logspace(-3, 1, 100)
best_alpha, best_mse, best_coef = 1.0, np.inf, None

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_poly_scaled, y)
    n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-4)
    mse = mean_squared_error(y, lasso.predict(X_poly_scaled))
    
    # Ищем баланс: малое число признаков (5-9) и минимальная ошибка
    if 5 <= n_nonzero <= 9 and mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_coef = lasso.coef_

# 5. Выделение важных признаков
mask = np.abs(best_coef) > 1e-4
selected_names = feature_names[mask]
selected_features_unscaled = X_poly[:, mask]

# 6. Финальное обучение обычной линейной регрессии на отобранных признаках
# (fit_intercept=False, так как столбец '1' уже включен в признаки)
final_model = LinearRegression(fit_intercept=False)
final_model.fit(selected_features_unscaled, y)

print(f"Оптимальное alpha: {best_alpha:.4e}")
print(f"Количество значимых мономов: {np.sum(mask)}")
print(f"Итоговая MSE: {mean_squared_error(y, final_model.predict(selected_features_unscaled)):.4f}")
print(f"R^2: {1 - mean_squared_error(y, final_model.predict(selected_features_unscaled)) / np.var(y):.4f}\n")

print("Итоговая формула зависимости:")
formula_parts = []
for name, coef in zip(selected_names, final_model.coef_):
    if name == '1':
        formula_parts.append(f"{coef:.4f}")
    else:
        formula_parts.append(f"{coef:.4f} * {name}")
    print(f"  {name}: {coef:.4f}")

print("\ny = " + " + ".join(formula_parts))