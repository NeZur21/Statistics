from collections import defaultdict, Counter
import math
import matplotlib.pyplot as plt

file = open('Москва_2021.txt')
lines1 = sorted(map(lambda x: x.strip(), file.readlines()))
lines3 = lines1[:]
file = open('Москва_2021.txt')
lines2 = map(lambda x: x.strip(), file.readlines())

def diskr(lines):
    d = defaultdict(int)
    for i in lines:
        if i != '\n':
            d[int(i)] += 1
    return d

def interval(chast):
    chast = chast[:]
    ryad = defaultdict(int)
    digits = [d for d, f in chast]
    x = len(chast) // 7 + 1
    for i in range(14, 7 * x + 14 + 1):
        if i not in digits:
            chast.append((i, 0))
    chast = sorted(chast)
    print(chast)
    #while len(chast) % 7 != 0:
        #chast.append((chast[-1][0] + 1, 0))
    i = 0
    freq = 0
    while len(ryad) != 7:
        ryad[(chast[i][0], chast[i + x][0])] = sum(chast[i + j][1] for j in range(9))
        freq = sum(chast[i + j][1] for j in range(9)) + freq
        i += x
    return ryad

def mid_d(d):
    x = 0
    for i in d:
        x += i[1] * i[0]
    x = x / sum(i[1] for i in d)
    return x

def mid_i(d):
    x = 0
    a = 0
    b = 1
    for i in d:
            x += (((i[0][0] + i[0][1]))/ 2) * i[1]
    x = x / sum(i[1] for i in d)
    return x

def disp_d(d):
    return sum((x[0] - mid_d(d)) ** 2 * x[1] for x in d) / sum(i[1] for i in d)
def disp_i(d):
    return sum((((x[0][0] + x[0][1]) / 2) - mid_i(d)) ** 2 * x[1] for x in d) / sum(i[1] for i in d)

def variation_d(d, sigma):
    return sigma / mid_d(d)

def variation_i(d, sigma):
    return sigma / mid_i(d)
def moda_d(d):
    counts = Counter(d)
    max_freq = max(d, key=lambda x: x[1])
    return max_freq

def moda_i(d):
    ma = max(d, key=lambda x: x[1])
    l = ma[0][0]
    freq = [f for (start, end), f in d]
    freq_1 = freq[d.index(ma) - 1]
    freq__1 = freq[d.index(ma) + 1]
    freqm = freq[d.index(ma)]
    return l + (freqm - freq_1) / ((freqm - freq_1) + (freqm - freq__1)) * 9

def midian(d):
    n = len(d)
    mid = n // 2
    if n % 2 == 0:
        return (d[mid - 1] + d[mid]) / 2
    else:
        return d[mid]

def diagram(d):
    starts = [start for (start, end), freq in d]
    widths = [end - start for (start, end), freq in d]
    frequencies = [freq for (start, end), freq in d]

    plt.bar(starts, frequencies, width=widths, align='edge', label='Гистограмма')

    ticks = [str(start) for start in starts]
    ticks.append(str(77))
    starts.append(77)
    plt.xticks(starts, ticks)


    plt.legend()
    plt.grid(True)
    plt.show()

def polygon(d):
    points = [i for i, f in d]
    frequencies = [f for i, f in d]
    plt.plot(points, frequencies, label='Полигон частот')

    start = d[0][0]
    end = d[-1][0]
    ticks = [str(start), int(end)]
    plt.xticks([start, end], ticks)

    plt.legend()
    plt.grid(True)
    plt.show()

def assimetr(d1):
    n = sum(i for a, i in d1)
    s = math.sqrt(sum(f * (x - mid_d(d1)) ** 2 for x, f in d1) / (n - 1))
    M = sum(f * (x - mid_d(d1) ) ** 3 for x, f in d1)

    return n / ((n - 1) * (n - 2)) * (M / s ** 3)
#    return (1 / n) * sum((((x - mid_d(d1)) / (sigma1) ** 3)  for x, f in d1)

def excess(d):
    n = sum(i for a, i in d)
    M = sum(f * (x - mid_d(d)) ** 4 for x, f in d) / n
    return M / (sigma1 ** 4) - 3

def sigma_3(d, lines):
    a = mid_d(d) - 3 * sigma1
    b = mid_d(d) + 3 * sigma1

    check = []

    for i in lines:
        if a <= int(i) <= b:
            check.append(i)

    return len(check) / len(lines) * 100

def comm(d):
    list1 = []
    f = 0
    for i in d:
        list1.append((i[0], i[1] + f))
        f += i[1]
    return list1

def func(d):
    x = [i for i, f in d]
    f = [f for i, f in d]

    plt.step(x, f, where='post', label='F(x)')
    plt.xlabel('X')
    plt.ylabel('F(X)')
    plt.title('Статистическая функция распределения')

    start = d[0][0]
    end = d[-1][0]
    ticks = [str(start), int(end)]
    plt.xticks([start, end], ticks)

    plt.grid(True)
    plt.legend()
    plt.show()

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
print(moda_d(diskr_ser), end=' мода и частота\n')
print(midian(sorted(lines2)), end=' медиана\n')
ma = max(diskr_ser, key=lambda x: x[0])[0]
mi = min(diskr_ser, key=lambda x: x[0])[0]
print(ma - mi, end=' размах\n')

print('\nИНРЕВАЛЬНОЕ')
print(*interval_ser, sep='\n', end=' значения\n')
print(mid_i(interval_ser), end=' средняя\n')
print(d2, end=' дисперсия\n')
print(sigma2, end=' средне квадратичное\n')
print(variation_i(interval_ser, sigma2) * 100, end=' коэф вариации\n')
print(moda_i(interval_ser), end=' мода\n')
print(interval_ser[-1][0][1] - interval_ser[0][0][0], end=' размах\n')

print()

print(assimetr(diskr_ser), end=' ассиметрия\n')
print(excess(diskr_ser), end=' эксцесс\n')

print(sigma_3(diskr_ser, lines3), end=' правило трех сигм соблюдено \n')

print(comm(diskr_ser))

diagram(interval_ser)
polygon(diskr_ser)
func(comm(diskr_ser))


