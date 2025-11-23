from Lab1.main1 import diskr_ser, interval_ser, mid_diskr, d1
import numpy as np
import math
from scipy.stats import t
import matplotlib.pyplot as plt
import scipy.stats as stats

X = [x for x, f in diskr_ser]
Y = [f for x, f in diskr_ser]

X_mean = np.mean(X)
Y_mean = np.mean(Y)

def cov(X, Y):
    c = 0
    for x, y in diskr_ser:
        c += (x - X_mean) * (y - Y_mean)
    return  c / (len(X))

sko_X = np.std(X, ddof=0)
sko_Y = np.std(Y, ddof=0)

def cof_cov(cov, sko_X, sko_Y):
    return cov / (sko_X * sko_Y)

covariation = cov(X, Y)
correlation = cof_cov(covariation, sko_X, sko_Y)

a = 0.05

def t_nabl(corrlation, n):
    return (corrlation * math.sqrt(n - 2)) / math.sqrt(1 - correlation ** 2)

nabl = t_nabl(correlation, len(X))

t_crit = t.ppf(1 - a / 2, df=len(X)-2)

x_i = []
d_i = []
for (a, b), f in interval_ser:
    mean = 0
    sum_x2 = 0
    freq = 0
    for d, f1 in diskr_ser:
        if a <= d < b:
            mean += d * f1
            sum_x2 += d ** 2 * f1
            freq += f1
    if f > 0:
        x_i.append(mean / freq)
        d_i.append((sum_x2 / freq) - ((mean / freq) ** 2))

def in_group(inter, d_i):
    frequency = [f for (a, b), f in inter]
    n = sum(frequency)

    return sum(d_i[i] * frequency[i] / n for i in range(len(frequency)))

ing = in_group(interval_ser, d_i)

def mej_group(inter, x_i):
    frequency = [f for (a, b), f in inter]
    n = sum(frequency)

    return sum((x_i[i] - mid_diskr) ** 2 * frequency[i] / n for i in range(len(frequency)))

mejg = mej_group(interval_ser, x_i)

obsh = ing + mejg

def spearman(diskr_ser):
    p = 0
    X = stats.rankdata([a for (a, f) in diskr_ser], method='average')
    Y = stats.rankdata([f for (a, f) in diskr_ser], method='average')
    for i in range(len(X)):
        p += (X[i] - Y[i]) ** 2
    return 1 - ((6 * p) / (len(diskr_ser) ** 3 - len(diskr_ser)))

spear_cof = spearman(diskr_ser)

print('Среднее значение возрастов', X_mean)
print('Среднее значение частот', Y_mean)

print(f'Коэффициент ковариации {covariation:.4f}')

print('Среднее квадратичное отклонение возраста', sko_X)
print('Среднее квадратичное отклонение частоты', sko_Y)

print('Линейный коэффициент корреляции', correlation)

print('r < -0.5: с возрастом частота преступлений снижается')

print('Наблюдаемое значение', nabl)
print('Критическое значение', t_crit)

if nabl < t_crit:
    print('Гипотеза H₀ отвергается - коэффициент корреляции в генеральной совокупности не равен нулю')
else:
    print('H₀ не отвергается - корреляция статистически незначима, значит частота и возраст связаны линейной корреляции')

#print('Средние групп:', x_i)
#print('Дисперсии групп:', d_i)
print('Внутригрупповая дисперсия', ing)
print('Межгрупповая дисперсия', mejg)
print(f'Общая дисперсия {obsh:.10f}')
print(f'Дисперсия из 1 лабы {d1:.10f}')
print('Корреляционное отношение', math.sqrt(mejg / obsh))
print('Коэффициент Спирмана', spear_cof)

plt.figure(figsize = (15, 6))
plt.scatter(X, Y, label='Данные')
plt.title('Корреляционное поле')
plt.xlabel('Возраст преступников')
plt.ylabel('Частота преступлений')
plt.xticks(range(min(X), max(X) + 1))
plt.grid(True)
plt.show()
