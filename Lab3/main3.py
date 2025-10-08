from Lab1.main1 import interval_ser as interval_row
from Lab1.main1 import sigma1, mid_intr
from Lab2.main2 import frequencies as interval_mid
from Lab2.main2 import sigma, _x
from scipy.stats import chi2 as chi2_crit
from scipy.stats import norm

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
    df = n - 1 - 2
    chi_crit = chi2_crit.ppf(1 - alpha, df)
    if chi2 <= chi_crit:
        print("H0 принимается: распределение нормальное")
    else:
        print("H0 отвергается: распределение не нормальное")

print('ИНТЕГРАЛЬНЫЙ РЯД')

n = freq(interval_row)

E = [n * (norm.cdf(b, mid_intr, sigma1) - norm.cdf(a, mid_intr, sigma1)) for (a, b), f in interval_row]

print('Ожидаемые частоты')
for i in E:
    print(i)

chi2_value  = chi(interval_row, E)

crit(interval_row, 0.05, chi2_value )

print('Хи2')
print(chi2_value )

print('Критическое значение')
print(chi2_crit.ppf(1 - 0.05, n - 2 - 1))

print()

print('СРЕДНИЕ ВОЗРАСТЫ')
print('Частота')
n = freq(interval_mid)
print(n)

E = [n * (norm.cdf(b, _x, sigma) - norm.cdf(a, _x, sigma)) for (a, b), f in interval_mid]
print('Ожидаемые частоты')
for i in E:
    print(i)

chi2_value = chi(interval_mid, E)

crit(interval_row, 0.05, chi2_value )

print('Хи2')
print(chi2_value )

print('Критическое значение')
print(chi2_crit.ppf(1 - 0.05, n - 2 - 1))