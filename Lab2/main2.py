import matplotlib.pyplot as plt
from Lab1.main1 import sigma1, mid
from math import sqrt, e, pi, floor, ceil, exp
import random

file = open('../Москва_2021.txt')
lines1 = list(map(lambda x: x.strip(), file.readlines()))

acc = 3
t = 1.96

n = round(t ** 2 * sigma1 ** 2 / acc ** 2)

def selection(n, digits):
    list1= [[] for i in range(36)]
    for i in range(36):
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
        i[1] = round(i[1], 3)
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

    plt.plot(points, frequencies1, 'r', label='Полигон частот')

    plt.legend()
    plt.grid(True)
    plt.show()

def normal(digits):
    list1 = [[a, 0] for [a, b], f in digits]
    _x = 0
    for [a, b], f in digits:
        _x += (a + b) / 2 * f
    sigma = 0
    for [a, b], f in digits:
        sigma += (a - _x) ** 2 * f
    sigma = sqrt(sigma)
    for x in range(len(list1)):
        list1[x][1] = (1 / (sigma * sqrt(2 * pi))) * e ** (-(list1[x][0] - _x) ** 2 / (2 * sigma ** 2))

    return list1

def confidence(digits):
    n = len(digits)
    mid = sum(digits) / n
    s = sqrt((sum((x - mid) ** 2 for x in digits) / (n - 1)))
    t_value = 2.262
    margin_error = t_value * s / (n ** 0.5)
    lower = mid - margin_error
    upper = mid + margin_error
    return (lower, upper)

selections = selection(n, lines1.copy())

averages = average(selections)

interval_row = intrevals(averages)

frequencies = freq(interval_row, averages)

norm = normal(frequencies)

conf = confidence(random.choice(selections))

print(norm)
#print('Значения выборок')
#for i in selections:
#    print(i)

print('Средние выборок')
print(averages)
print('Интервальный ряд')
print(interval_row)
print('Интервальный ряд с частотами')
print(frequencies)
print('Нормальное распределение')
print(norm)
print('Доверительный интервал')
print(conf)

diagram(frequencies, norm)


