import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

ages = np.loadtxt('../Москва_2021.txt')
# ЛР 2.1: Выборочное наблюдение

# Надёжность γ = 0.95 → z = 1.96
# Точность δ = 3 года
sigma = np.std(ages, ddof=0)  # ddof=0
z = stats.norm.ppf(0.975)  # 1.96
delta = 3
n = math.ceil((z * sigma / delta) ** 2)  # Округление вверх для целого n
print(f"Объём выборки n: {n}")

# Генерация 36 выборок размера n с возвращением (with replacement)
num_samples = 36
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(ages, size=n, replace=True)
    mean = np.mean(sample)
    sample_means.append(mean)

sample_means = np.array(sample_means)

min_mean = np.min(sample_means)
max_mean = np.max(sample_means)
left = math.floor(min_mean)  # Округление вниз
right = math.ceil(max_mean)  # Округление вверх
bins = np.arange(left, right + 1, 1)  # Интервалы длиной 1 год

# Гистограмма
plt.figure(figsize=(10, 6))
hist, edges, _ = plt.hist(sample_means, bins=bins, density=True, alpha=0.6, color='b', edgecolor='black')
plt.title('Гистограмма относительных частот выборочных средних')
plt.xlabel('Возраст (годы)')
plt.ylabel('Относительная частота')
plt.grid(True)
plt.show()

# Вывод относительных частот
rel_freq = hist  # density=True даёт относительные частоты (нормализованные)
print("Интервалы и относительные частоты:")
for i in range(len(hist)):
    print(f"[{edges[i]:.0f}, {edges[i+1]:.0f}): {rel_freq[i]:.4f}")

# ЛР 2.2: Статистические оценки

mids = (edges[:-1] + edges[1:]) / 2
freq = hist * len(sample_means)  # Абсолютные частоты
# Точечные оценки методом моментов:
a_hat = np.sum(mids * freq) / np.sum(freq) # Берём середины столов графика и усредняем и, учитывая кол-во наблюдений
# Разброс центров столбов гистограммы
var_hat = np.sum(freq * (mids - a_hat)**2) / np.sum(freq)
sigma_hat = np.sqrt(var_hat)
print(f"Оценка a: {a_hat:.2f}")
print(f"Оценка sigma: {sigma_hat:.2f}")

# Построение кривой Гаусса поверх гистограммы
x = np.linspace(left, right, 100)
pdf = stats.norm.pdf(x, loc=a_hat, scale=sigma_hat)  # Плотность нормального
plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=bins, density=True, alpha=0.6, color='b', edgecolor='black')
plt.plot(x, pdf, 'r-', lw=2, label='Кривая Гаусса')
plt.title('Гистограмма с аппроксимирующей кривой Гаусса')
plt.xlabel('Возраст (годы)')
plt.ylabel('Относительная частота / Плотность')
plt.legend()
plt.grid(True)
plt.show()

# Доверительный интервал для мат. ожидания на основе одной выборки
sample = np.random.choice(ages, size=n, replace=True)  # Одна выборка
sample_mean = np.mean(sample)  # Точечная оценка
sample_std = np.std(sample, ddof=1)  # Выборочное std (ddof=1)
df = n - 1  # Степени свободы
t_quantile = stats.t.ppf(0.975, df)  # Квантиль Стьюдента для 95%
margin = t_quantile * sample_std / np.sqrt(n)  # Точность (полуширина)
ci_lower = sample_mean - margin
ci_upper = sample_mean + margin

print(f"Точечная оценка (среднее выборки): {sample_mean:.2f}")
print(f"Доверительный интервал: ({ci_lower:.2f}, {ci_upper:.2f})")
print(f"Точность оценки (margin): {margin:.2f}")
print(f"Квантиль распределения Стьюдента: {t_quantile:.4f}")