from Lab1.main1 import interval_ser as interval_row
from Lab1.main1 import sigma2, mid_intr, d1, disp_d, diskr
from Lab2.main2 import frequencies as interval_mid
from Lab2.main2 import sigma, _x, selection_1, selection_2
from scipy.stats import chi2 as chi2_crit
from scipy.stats import f, norm
import math

interval_mid = [[[a, b], f * 36] for [a, b], f in interval_mid]

def freq(interval_ser):
    f = 0
    for (a, b), fr in interval_ser:
        f += fr
    return f

def chi(interval, E):
    chi2 = 0
    for i in range(len(interval)):
        ((a, b), O) = interval[i]
        Ei = E[i]
        chi2 += (O - Ei) ** 2 / Ei
    return chi2

def crit(interval, alpha, chi2):
    m = 2
    df = len(interval) - 1 - 2
    chi_crit = chi2_crit.ppf(1 - alpha, df)
    if chi2 <= chi_crit:
        print("H0 принимается: распределение нормальное")
    else:
        print("H0 отвергается: распределение не нормальное")

def fisher(sigma1, sigma2):
    return max(sigma1, sigma2) ** 2 / min(sigma1, sigma2) ** 2

intervals = []

for (a, b), freqs in interval_row:
    intervals.append([[a, b], freqs])

interval_row = intervals

interval_row[0][0][0] = -float('inf')
interval_row[-1][0][1] = float('inf')

print('ИНТЕРВАЛЬНЫЙ РЯД')

n = freq(interval_row)

E = [n * (norm.cdf((b-mid_intr) / sigma2) - norm.cdf((a-mid_intr) / sigma2)) for (a, b), f in interval_row]

print('Ожидаемые частоты')
for i in E:
    print(i)

print()

print("H0: распределение возраста — нормальное")
print("H1: распределение возраста — не нормальное\n")
chi2_value  = chi(interval_row, E)

crit(interval_row, 0.05, chi2_value )

print('Наблюдаемое значение Хи2')
print(chi2_value )

print('Критическое значение')
print(chi2_crit.ppf(1 - 0.05, len(interval_row) - 2 - 1))

print()

print('СРЕДНИЕ ВОЗРАСТЫ')
print('Частота')
n = freq(interval_mid)
print(n, end='\n')

E = [n * (norm.cdf(b, _x, sigma) - norm.cdf(a, _x, sigma)) for (a, b), f in interval_mid]
print('Ожидаемые частоты')
for i in E:
    print(i)
print()
chi2_value = chi(interval_mid, E)

crit(interval_row, 0.05, chi2_value )
print("H0: распределение среднего возраста — нормальное")
print("H1: распределение среднего возраста — не нормальное\n")
print()
print('Наблюдаемое значение Хи2')
print(chi2_value )

print('Критическое значение')
print(chi2_crit.ppf(1 - 0.05, len(interval_mid) - 2 - 1))

print()

alpha = 0.05
sel1 = selection_2[0]
sel2 = selection_2[1]
sigma_1 = disp_d(sorted(diskr(sel1).items()), False)
sigma_2 = disp_d(sorted(diskr(sel2).items()), False)
n1 = len(sel1)
n2 = len(sel2)
if sigma_2 > sigma_1:
    n1, n2 = n2, n1
print("H0: D1 = D2")
print("H1: D1 > D2\n")
print(f"Степени свободы: k₁ = {n1 - 1}, k₂ = {n2 - 1}\n")
print()
f_crit = f.ppf(1 - alpha, n1 - 1, n2 - 1)
print(f"s1² = {sigma_1:.6f}")
print(f"s2² = {sigma_2:.6f}")
print()
print('H0: D1 = D2')
print('H1: D1 > D2')
if fisher(sigma_1, sigma_2) > f_crit:
    print(f'Отвергаем H0 - различие дисперсий статистически значимо: критическое Fкрит = {f_crit}, наблюдаемое значение F = {fisher(sigma_1, sigma_2)}')
else:
    print(f'Принимаем H0 - различие дисперсий статистически незначимо: критическое Fкрит = {f_crit}, наблюдаемое значение F = {fisher(sigma_1, sigma_2)}')

f_crit_low = f.ppf(alpha / 2, n1 - 1, n2 - 1)
f_crit_high = f.ppf(1- alpha / 2, n1 - 1, n2 - 1)

print('H0: D1 = D2')
print('H1: D1 ≠ D2')
if fisher(sigma_1, sigma_2) < f_crit_low or fisher(sigma_1, sigma_2) > f_crit_high:
    print(f'Отвергаем H0 дисперсии различаются: критические значения: {f_crit_high}, наблюдаемое значение F = {fisher(sigma_1, sigma_2)}')
else:
    print(f'Принимаем H0 различие дисперсий статистически незначимо: критические значения: {f_crit_high}, наблюдаемое значение F = {fisher(sigma_1, sigma_2)}')

print()

disp = disp_d(sorted(diskr(selection_1).items()), False)

x_2 = ((len(selection_1) - 1) * disp ) / d1

x_2_crit = chi2_crit.ppf(1 - alpha, len(selection_1) - 1)

print('Дисперсия генеральной совокупности')
print(d1)

print('Дисперсия выборки')
print(disp, end='\n')
print()

print('H0: D(x) = s^2')
print('H1: D(x) > s^2')

if x_2 > x_2_crit:
    print(f'Отвергаем H0 генеральная дисперсия больше заданной: критическое Xи2крит {x_2_crit}, наблюдаемое Хи2 {x_2}')
else:
    print(f'Принимаем H0 различие незначимо: критическое Xи2крит {x_2_crit}, наблюдаемое Хи2 {x_2}')

x_2_crit_low = chi2_crit.ppf(alpha / 2, len(selection_1) - 1)
x_2_crit_high = chi2_crit.ppf(1 - alpha / 2, len(selection_1) - 1)

print()
print('H0: D(x) = s^2')
print('H1: D(x) ≠ s^2')
if x_2 < x_2_crit_low or x_2 > x_2_crit_high:
    print(f'Отвергаем H0: критические: {x_2_crit_low}; {x_2_crit_high}, наблюдаемое Хи2 {x_2}')
else:
    print(f'Принимаем H0: критическое: {x_2_crit_low}; {x_2_crit_high}, наблюдаемое Хи2 = {x_2}')

print()
print('H0: D(x) = s^2')
print('H1: D(x) < s^2')
x_2_crit = chi2_crit.ppf(alpha, len(selection_1) - 1)
if x_2 < x_2_crit:
    print(f'Отвергаем H0 генеральная дисперсия меньше заданной: критическое Xи2крит {x_2_crit}, наблюдаемое Хи2 = {x_2}')
else:
    print(f'Принимаем H0 различие незначимо: критическое Xи2крит {x_2_crit}, наблюдаемое Хи2 = {x_2}')