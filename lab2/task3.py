import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Загрузка данных
df_strange = pd.read_csv('lab2/hw2_strange_data.csv')
X_s = df_strange[['x']].values
y_s = df_strange['y'].values

# 2. Обучение линейной регрессии
model = LinearRegression()
model.fit(X_s, y_s)
y_pred = model.predict(X_s)

print(f"Коэффициент наклона: {model.coef_[0]:.4f}")
print(f"Свободный член: {model.intercept_:.4f}")
print(f"MSE: {np.mean((y_s - y_pred)**2):.4f}")

# 3. Визуализация
plt.figure(figsize=(8, 5))
plt.scatter(X_s, y_s, alpha=0.7, label='Данные')
plt.plot(X_s, y_pred, color='red', linewidth=2, label=f'Линейная регрессия (MSE={np.mean((y_s - y_pred)**2):.1f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Диаграмма рассеяния "странных" данных')
plt.legend()
plt.grid(True)
plt.show()