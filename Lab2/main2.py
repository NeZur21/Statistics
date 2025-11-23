import matplotlib.pyplot as plt
from Lab1.main1 import sigma1
from math import sqrt, e, pi, floor, ceil
import random
import scipy

file = open('../Москва_2021.txt')
lines1 = list(map(lambda x: x.strip(), file.readlines()))

acc = 3
alpha = 0.05
t = 1.96
n = ceil(t ** 2 * sigma1 ** 2 / acc ** 2)


def selection(n, digits, k):
    list1= [[] for i in range(k)]
    for i in range(k):
        for j in range(n):
            list1[i].append(int(digits[random.randint(0, len(digits) - 1)]))
    return list1

def average(digits):
    list1 = []
    for select in digits:
        list1.append(sum(select) / len(select))
    return list1

def intrevals(digits):
    start = floor(min(digits))
    end = ceil(max(digits))
    list1 = [[i, i + 1] for i in range(start, end + 1)]
    return list1

def freq(digits, lines):
    list1 = [[i, 0] for i in digits]
    for i in list1:
        for digit in lines:
            if i[0][0] <= int(digit) < i[0][1]:
                i[1] += 1
        i[1] /= 36
        i[1] = i[1]
    return list1

def diagram(d, norm):
    starts = [start for (start, end), freq in d]
    widths = [end - start for (start, end), freq in d]
    frequencies = [freq for (start, end), freq in d]

    plt.bar(starts, frequencies, width=widths, align='edge', label='Гистограмма')

    ticks = [str(start) for start in starts]
    ticks.append(str(40))
    starts.append(40)
    plt.xticks(starts, ticks)

    points = [x for x, f in norm]
    frequencies1 = [f for x, f in norm]

    plt.plot(points, frequencies1, 'r', label='Нормальное распределение')

    plt.legend()
    plt.grid(True)
    plt.show()

def av(digits):
    x = 0
    for [a, b], f in digits:
        x += (a + b) / 2 * f
    return x

def sigma_inter(digits, x):
    sigma = 0
    for [a, b], f in digits:
        sigma += (a - x) ** 2 * f
    sigma = sqrt(sigma)
    return sigma

def normal(digits, sigma, _x, step):
    start = digits[0][0][0]
    end = digits[-1][0][1]

    list1 = []
    a = start
    while a <= end:
        y = (1 / (sigma * sqrt(2 * pi))) * e ** (-(a - _x) ** 2 / (2 * sigma ** 2))
        list1.append([a, y])
        a += step

    return list1

def confidence(digits):
    n = len(digits)
    mid = sum(digits) / n
    s = sqrt((sum((x - mid) ** 2 for x in digits) / (n - 1)))
    t_value = scipy.stats.t.ppf(0.975, n - 1)
    margin_error = t_value * s / (n ** 0.5)
    lower = mid - margin_error
    upper = mid + margin_error
    return (float(lower), float(upper)), mid, s, margin_error

selections = selection(n, lines1.copy(), 36)

averages = average(selections)

interval_row = intrevals(averages)

frequencies = freq(interval_row, averages)

_x = av(frequencies)

sigma = sigma_inter(frequencies, _x)

norm = normal(frequencies, sigma, _x, 1)
norm_graph = normal(frequencies, sigma, _x, 0.1)

conf = confidence(random.choice(selections))

selection_1 = random.choice(selections)
selection_2 = [random.choice(selections), random.choice(selections)]

cov = -4032.3032

if __name__ == "__main__":
    print('Длин выборки')
    print(n)
    #print('Значения выборок')
    #for i in selections:
    #    print(i)



    print('Средние выборок')
    print(averages, end='\n\n')
    print('Интервальный ряд')
    print(interval_row, end='\n\n')
    print('Интервальный ряд с частотами')
    print(frequencies, end='\n\n')

    print('Значения для выборки')
    print('Критерий Стьдента')
    print(scipy.stats.t.ppf(1 - alpha / 2, len(random.choice(selections)) - 1), end='\n\n')
    print('СКО')
    print(conf[2], end='\n\n')
    print('Точность')
    print(conf[3], end='\n\n')
    print('Доверительный интервал')
    print(f'{conf[0][0]} {conf[0][1]}', end='\n\n')
    print('Средняя доверительного интервала')
    print(f'{(conf[0][0] + conf[0][1]) / 2}', end='\n\n')

    diagram(frequencies, norm_graph)

