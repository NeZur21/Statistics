from math import sqrt
import pandas as pd
from Lab4.main4 import diskr_ser, correlation
import matplotlib.pyplot as plt
import numpy as np
import math

types = ["y = ax + b", "y = ax^b", "y = ab^x", "y = a + b/x", "y = 1/(ax + b)", "y = x/(ax + b)", "y = a ln x + b"]

X = [x for x, f in diskr_ser]
Y = [f for x, f in diskr_ser]
# print(X)
# print(Y)

x_mean = [0] * 7
y_mean = [0] * 7

x_mean[0], x_mean[2], x_mean[4] = [round((X[0] + X[-1]) / 2, 1)] * 3
x_mean[1], x_mean[6] = [round(sqrt(X[0] * X[-1]), 1)] * 2
x_mean[3], x_mean[5] = [round((2 * X[0] * X[-1]) / (X[0] + X[-1]), 1)] * 2

y_mean[0], y_mean[3], y_mean[6] = [round((Y[0] + Y[-1]) / 2, 1)] * 3
y_mean[1], y_mean[2] = [round(sqrt(Y[0] * Y[-1]), 1)] * 2
y_mean[4], y_mean[5] = [round((2 * Y[0] * Y[-1]) / (Y[0] + Y[-1]), 1)] * 2

y_s = [0] * 7

for i in range(len(X) - 1):
    if X[i] <= x_mean[0] < X[i + 1]:
        a = i
        a1 = i + 1
    elif X[i] <= x_mean[1] < X[i + 1]:
        b = i
        b1 = i + 1
    elif X[i] <= x_mean[3] < X[i + 1]:
        c = i
        c1 = i + 1

y_s[0], y_s[2], y_s[4] = [round((Y[a] + (Y[a1] - Y[a]) / (X[a1] - X[a]) * (x_mean[0] - X[a])), 2)] * 3
y_s[1], y_s[6] = [round((Y[b] + (Y[b1] - Y[b]) / (X[b1] - X[b]) * (x_mean[1] - X[b])), 2)] * 2
y_s[3], y_s[5] = [round((Y[c] + (Y[c1] - Y[c]) / (X[c1] - X[c]) * (x_mean[3] - X[c])), 2)] * 2

d_s = []
for i in range(len(y_s)):
    d_s.append(abs(y_mean[i] - y_s[i]))

d_s_p = []
for i in range(len(d_s)):
    d_s_p.append(round((d_s[i] / sum(Y) * 100), 2))

min_d_s = min(d_s)
zavis = d_s.index(min_d_s)

result = []
for i in range(len(d_s)):
    result.append({'Структура': types[i], 'x_mean': x_mean[i], 'y_mean': y_mean[i], 'y_s': y_s[i], 'd_s': d_s[i], 'd_s_p %': str(d_s_p[i]) + '%'})

S = sum(Y)
t = 0.02 * S



print(f'Минимальное : {min_d_s}')
print(f'Зависимость min delta_s: {types[zavis]}')
print(pd.DataFrame(result))

plt.figure(figsize=(8, 4))
plt.plot(X, Y, "o-", color="blue", label="Исходные данные")
plt.xlabel("x")
plt.ylabel("y")
plt.title("График экспериментальных данных")
plt.grid(True)

plt.legend()
plt.show()

def approx(X,Y, func):

    x_interp = np.arange(min(X), max(X) + 1, 1)
    # print(x_interp)

    u = [func(val) for val in X] # math.log(val)
    n = len(X)
    sum_u = sum(u)
    sum_y = sum(Y)
    sum_uy = sum(u[i] * Y[i] for i in range(n))
    sum_u2 = sum(u[i] ** 2 for i in range(n))

    a = (sum_uy - (sum_u * sum_y) / n) / (sum_u2 - (sum_u ** 2) / n)
    b = (sum_y / n) - a * (sum_u / n)

    # print(f"Коэффициенты аппроксимации: a = {a}, b = {b}")

    new_y = []
    for xi in x_interp:
        if xi in X:
            new_y.append(Y[list(X).index(xi)])
        else:
            # print(math.log(xi), xi)
            new_y.append(a * func(xi) + b) # math.log(val)
    # Оценка параметров линейной модели

    print(f"Модель: y = {a:.4f} * x + {b:.4f}")

    if func == math.log:
        func = np.log

    X = np.array(X, dtype=float)
    Y_pred = a * func(X) + b   # np.log(X)
    # -0.5677283540348753

    SS_res = np.sum((Y - Y_pred)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - SS_res / SS_tot

    print(f"R^2 = {R2:.4f}")
    print(f"√R^2 = {np.sqrt(R2):.4f}")
    print(f'rxy = {correlation:.4f}')
    return a, b, sum_y, new_y, R2

a, b, sum_y, new_y, R2 = approx(X,Y, abs)

# print("Обновленный массив y:", new_y)

percent_2 = 0.02 * sum_y  # 2% от суммы частот
y_diff = []
max_diff = float('inf')
attempt = 0

new_y_copy = new_y.copy()

while max_diff > percent_2:
    attempt += 1
    y_diff = [abs(new_y_copy[i] - new_y_copy[i - 1]) for i in range(1, len(new_y_copy))]

    print(f"Попытка {attempt}, последовательные разности: {y_diff}")

    max_diff = max(y_diff)
    print(f"Максимальная разность = {max_diff} ({'больше' if max_diff > percent_2 else 'меньше'} 2%)")

    new_y_copy = y_diff.copy()

print(f"Минимальная разность ≤ 2%: {max_diff}")
print(f"Показатель степени аппроксимирующего многочлена: {attempt}")

a1, b1, sum_y1, new_y1, R21 = approx(X,Y, math.log)

plt.figure(figsize=(10, 6))

plt.scatter(X, Y, color="blue", label="Исходные данные", s=60)

x_fit = np.linspace(min(X), max(X), 200)
y_fit = a * x_fit + b # np.log(x_fit)
plt.plot(x_fit, y_fit, color="red", linewidth=2, label=f"y = {a:.4f} x + {b:.4f}")

x_fit1 = np.linspace(min(X), max(X), 200)
y_fit1 = a1 * np.log(x_fit1) + b1 # np.log(x_fit)
plt.plot(x_fit1, y_fit1, color="blue", linewidth=2, label=f"y = {a1:.4f} x + {b1:.4f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Корреляционное поле с аппроксимацией")
plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
plt.legend()

plt.text(min(X) + 0.5, max(Y) - 0.5, f"R² = {R2:.4f}", fontsize=12, color="darkred")
plt.text(min(X) + 0.5, max(Y) - 100, f"R² = {R21:.4f}", fontsize=12, color="darkblue")


plt.show()

