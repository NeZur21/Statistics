from collections import defaultdict, Counter
import math
import matplotlib.pyplot as plt

file = open('Lab1/Москва_2021.txt')
lines1 = map(lambda x: x.strip(), file.readlines())
file = open('Lab1/Москва_2021.txt')
lines2 = map(lambda x: x.strip(), file.readlines())

def diskr(lines):
    d = defaultdict(int)
    for i in lines:
        if i != '\n':
            d[int(i)] += 1
    return d

def interval(chast):
    ryad = defaultdict(int)
    start = chast[0][0]
    end = chast[0][0] + 8
    max_value = max(chast, key=lambda x: x[0])[0]
    sorted_chast = sorted(chast, key=lambda x: x[0])
    intervals = []

    while start <= max_value:
        intervals.append((start, end))
        start = end + 1
        end = start + 8
    for interval in intervals:
        elements_in_interval = [el for el in sorted_chast if interval[0] <= el[0] <= interval[1]]
        ryad[interval] = sum(el[1] for el in elements_in_interval)
    return ryad

def mid_d(d):
    x = 0
    for i in d:
        x += i[1] * i[0]
    x = x / sum(i[1] for i in d)
    return x

def mid_i(d):
    x = 0
    for i in d:
        x += ((i[0][0] + i[0][1]) / 2  + 0.5) * i[1]
    x = x / sum(i[1] for i in d)
    return x

def disp_d(d):
    return sum((x[0] - mid_d(d)) ** 2 * x[1] for x in d) / sum(i[1] for i in d)
def disp_i(d):
    return sum((((x[0][0] + x[0][1] + 1) / 2) - mid_i(d)) ** 2 * x[1] for x in d) / sum(i[1] for i in d)

def variation_d(d, sigma):
    return sigma / mid_d(d)

def variation_i(d, sigma):
    return sigma / mid_i(d)
def moda(d):
    counts = Counter(d)
    max_freq = max(d, key=lambda x: x[1])
    return max_freq

def midian(d):
    n = len(d)
    mid = n // 2
    if n % 2 == 0:
        return (d[mid - 1] + d[mid]) / 2
    else:
        return d[mid]

def diagram(d):
    midpoints = [(start + end) / 2 for (start, end), freq in d]

    frequencies = [freq for (start, end), freq in d]
    plt.bar(midpoints, frequencies, width=9, label='Гистограмма')

    plt.legend()
    plt.grid(True)
    plt.show()

def polygon(d):
    points = [i for i, f in d]
    frequencies = [f for i, f in d]
    plt.plot(points, frequencies, label='Полигон частот')
    plt.legend()
    plt.grid(True)
    plt.show()

def assimetr(d):
    n = sum(i for a, i in d)
    return (1 / n) * sum((x - mid_d(d) / sigma1) ** 3 for x, f in d)

diskr_ser = sorted(diskr(lines1).items())
interval_ser = sorted(interval(diskr_ser).items())

print('ДИСКРЕТНОЕ')

print(*diskr_ser, sep='\n', end=' значения\n')

print(mid_d(diskr_ser), end=' средняя\n')

d1 = disp_d(diskr_ser)
d2 = disp_i(interval_ser)
print(d1, end=' дисперсия\n')
sigma1 = math.sqrt(d1)
sigma2 = math.sqrt(d2)
print(sigma1, end=' средне квадратичное\n')
print(variation_d(diskr_ser, sigma1) * 100, end=' коэф вариации\n')
print(moda(diskr_ser), end=' мода\n')
print(midian(sorted(lines2)), end=' медиана\n')
ma = max(diskr_ser, key=lambda x: x[0])[0]
mi = min(diskr_ser, key=lambda x: x[0])[0]
print(ma - mi, end=' размах\n')
print(assimetr(diskr_ser), end=' ассиметрия\n')

print('\nИНРЕВАЛЬНОЕ')
print(*interval_ser, sep='\n', end=' значения\n')
print(mid_i(interval_ser), end=' средняя\n')
print(d2, end=' дисперсия\n')
print(sigma2, end=' средне квадратичное\n')
print(variation_i(interval_ser, sigma2) * 100, end=' коэф вариации\n')

diagram(interval_ser)
polygon(diskr_ser)



