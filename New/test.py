from Lab1.main1 import interval_ser as interval_row
from Lab1.main1 import sigma2, mid_intr, d1, disp_d, diskr, sko
from Lab2.main2 import frequencies as interval_mid
from Lab2.main2 import sigma, _x, selection_1, selection_2
from scipy.stats import chi2 as chi2_crit
from scipy.stats import f, norm
#from Lab3.main3 import chi
import math

n = 32423

E = []

for (a, b), f in interval_row:
    if a == 14:
        print(-float('inf'), b)
        print((a-mid_intr) / sigma2, (b-mid_intr) / sigma2)
        print(n * (norm.cdf((b-mid_intr) / sigma2) - norm.cdf((-float('inf')-mid_intr) / sigma2)))
        E.append(n * (norm.cdf((b-mid_intr) / sigma2) - norm.cdf((-float('inf')-mid_intr) / sigma2)))
    elif b == 77:
        print(a, float('inf'))
        print((a - mid_intr) / sigma2, (b - mid_intr) / sigma2)
        print(n * (norm.cdf((float('inf') - mid_intr) / sigma2) - norm.cdf((a - mid_intr) / sigma2)))
        E.append(n * (norm.cdf((float('inf') - mid_intr) / sigma2) - norm.cdf((a - mid_intr) / sigma2)))
    else:
        print(a, b)
        print((a - mid_intr) / sigma2, (b - mid_intr) / sigma2)
        print(n * (norm.cdf((b - mid_intr) / sigma2) - norm.cdf((a - mid_intr) / sigma2)))
        E.append(n * (norm.cdf((b - mid_intr) / sigma2) - norm.cdf((a - mid_intr) / sigma2)))

#print(chi(interval_row, E), '--')
