from math import sqrt
import pandas as pd
from Lab4.main4 import diskr_ser
from Lab5.main5 import attempt
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import f, t

X = [x for x, f in diskr_ser]
Y = [f for x, f in diskr_ser]

def mnk(m, x, y):
    x = np.array(x)
    n = x.shape[0]
    y = np.array(y).reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        x = np.column_stack([x ** i for i in range(1, m + 1)])

    C = np.hstack([np.ones((n, 1)), x])
    CtC = C.T @ C
    B = np.linalg.inv(CtC) @ C.T @ y

    y_pred = C @ B
    res = y - y_pred
    RSS = np.sum(res ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    ESS = TSS - RSS
    R2 = 1 - RSS / TSS
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - m - 1)
    s = np.sqrt(RSS / (n - m -1))

    return {
        "B": B,
        "n": n,
        'm': m,
        "y_pred": y_pred,
        "res": res,
        "RSS": RSS,
        "R2": R2,
        "R2_adj": R2_adj,
        "s": s,
        "C": C,
        "ESS": ESS
    }
res5 = mnk(attempt, X, Y)

# 1. Форматируем вывод коэффициентов
coeffs_str = ', '.join([f"w{i}" for i in range(len(res5['B']))])
print(f"Найденные весовые коэффициенты ({coeffs_str}): {res5['B']}")

# 2. Форматируем уравнение регрессии
equation_parts = [f"{res5['B'][0][0]:.2f}"]
for i in range(1, len(res5['B'])):
    equation_parts.append(f"{res5['B'][i][0]:.6f} * x{i}")

equation_str = " + ".join(equation_parts)
print(f"Уравнение регрессии: y = {equation_str}")
print(f'Коэффициент детерминации R2 = {res5['R2']}')
print(f'Скорректированный коэффициент детерминации R2adj = {res5['R2_adj']}')

res3 = mnk(3, X, Y)

# 1. Форматируем вывод коэффициентов
coeffs_str = ', '.join([f"w{i}" for i in range(len(res3['B']))])
print(f"Найденные весовые коэффициенты ({coeffs_str}): {res3['B']}")

# 2. Форматируем уравнение регрессии
equation_parts = [f"{res3['B'][0][0]:.2f}"]
for i in range(1, len(res3['B'])):
    equation_parts.append(f"{res3['B'][i][0]:.6f} * x{i}")

equation_str = " + ".join(equation_parts)
print(f"Уравнение регрессии: y = {equation_str}")
print(f'Коэффициент детерминации R2 = {res3['R2']}')
print(f'Скорректированный коэффициент детерминации R2adj = {res3['R2_adj']}')

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Исходные точки', color='black')
plt.plot(X, res5['y_pred'], color='red', label='Полином 5 степени')
plt.plot(X, res3['y_pred'], color='blue', label='Полином 3 степени')
plt.legend()
plt.grid(True)
plt.xlabel('Возраст')
plt.ylabel('Частоты')
plt.show()

data = {
    'year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'x1': [4922.4, 4130.7, 4137.4, 3889.4, 4263.9, 4243.5, 3966.5, 3657.0, 3461.2, 4316.0, 3624.6],
    'x2': [23369, 26629, 29792, 32495, 34030, 36709, 39167, 43724, 47867, 51344, 57244],
    'x3': [1865.9, 1807.9, 1749.5, 1690.0, 1577.0, 1444.5, 1304.6, 1208.6, 1126.7, 1102.8, 1077.7],
    'x4': [24951.2, 14648.1, 39558.7, 32365.0, 46568.8, 27929.6, 37218.5, 51418.1, 116166.5, 126304.8, 159875.4],
    'x5': [12.7, 10.7, 10.8, 11.3, 13.4, 13.2, 12.9, 12.6, 12.3, 12.1, 11.0],
    'x6': [669.4, 644.1, 668.0, 693.7, 611.6, 608.3, 611.4, 583.9, 620.7, 564.7, 644.2],
    'y': [2404.8, 2302.2, 2206.2, 2190.6, 2388.5, 2160.1, 2058.5, 1991.5, 2024.3, 2044.2, 2004.4]
}

df = pd.DataFrame(data)

X_multi = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
Y_multi = df[['y']].values.reshape(-1, 1)

n, m = X_multi.shape

res1 = mnk(m, X_multi, Y_multi)

# 1. Форматируем вывод коэффициентов
coeffs_str = ', '.join([f"w{i}" for i in range(len(res1['B']))])
print(f"Найденные весовые коэффициенты ({coeffs_str}): {res1['B']}")

# 2. Форматируем уравнение регрессии
equation_parts = [f"{res1['B'][0][0]:.2f}"]
for i in range(1, len(res1['B'])):
    equation_parts.append(f"{res1['B'][i][0]:.6f} * x{i}")

equation_str = " + ".join(equation_parts)
print(f"Уравнение регрессии: y = {equation_str}")

df_factors = df.drop('year', axis=1)
correlation_matrix = df_factors.corr()

print("## 1. Матрица парных корреляций (ρ)")
print(correlation_matrix.round(4))
print("-" * 50)

df = pd.DataFrame(data)
df_factors = df.drop('year', axis=1)  # Все переменные для корреляции


def calculate_pearson_corr(X_list, Y_list):
    n = len(X_list)
    if n == 0 or n != len(Y_list):
        return 0

    # Расчет средних значений (Чистый Python: используем sum())
    mean_x = sum(X_list) / n
    mean_y = sum(Y_list) / n

    # Числитель (Ковариация: Sum[(xi - mean_x) * (yi - mean_y)])
    numerator = sum([(x - mean_x) * (y - mean_y) for x, y in zip(X_list, Y_list)])

    # Знаменатель (Произведение стандартных отклонений)
    std_x_sq = sum([(x - mean_x) ** 2 for x in X_list])
    std_y_sq = sum([(y - mean_y) ** 2 for y in Y_list])

    denominator = math.sqrt(std_x_sq * std_y_sq)

    if denominator == 0:
        return 0

    return numerator / denominator


def build_correlation_matrix_manual(dataframe):
    columns = dataframe.columns.tolist()
    matrix = pd.DataFrame(index=columns, columns=columns)

    for i in range(len(columns)):
        for j in range(i, len(columns)):
            col1 = columns[i]
            col2 = columns[j]

            # Преобразуем Series в список для передачи в функцию
            corr_val = calculate_pearson_corr(dataframe[col1].tolist(), dataframe[col2].tolist())

            # Матрица симметрична
            matrix.loc[col1, col2] = corr_val
            matrix.loc[col2, col1] = corr_val

    return matrix.astype(float)


# 2. Построение матрицы парных корреляций
correlation_matrix_manual = build_correlation_matrix_manual(df_factors)

print("## 1. Матрица парных корреляций (Ручной расчет)")
print(correlation_matrix_manual.round(4))
print("-" * 50)

print("\nИсключение факторов с |r| < 0.5 относительно y:")

target = 'y'  # столбец с зависимой переменной
features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']  # независимые переменные

selected_features = []
for factor in features:
    r_val = df[[factor, target]].corr().iloc[0, 1]
    if abs(r_val) >= 0.5:
        print(f"  Сохраняем {factor}: r = {r_val:.4f}")
        selected_features.append(factor)
    else:
        print(f"  Исключаем {factor}: r = {r_val:.4f}")
print(f"\nВыбранные факторы для стартовой модели: {selected_features}")


def dispersion_analysis(results_mnk):
    """Возвращает факторную и остаточную дисперсии, F-критерий и p-значение."""
    n = results_mnk['n']
    m = results_mnk['m']
    ESS = results_mnk['ESS']
    RSS = results_mnk['RSS']

    # Степени свободы
    df_ess = m  # df1
    df_rss = n - m - 1  # df2

    # Дисперсии (Средние квадраты)
    factor_dispersion = ESS / df_ess  # MS_ESS
    residual_dispersion = RSS / df_rss  # MS_RSS

    # F-критерий
    F_stat = factor_dispersion / residual_dispersion

    # Вероятность (p-значение)
    p_value = f.sf(F_stat, df_ess, df_rss)

    return {
        "Факторная дисперсия": factor_dispersion,
        "Остаточная дисперсия": residual_dispersion,
        "F-критерий": F_stat,
        "p-значение (F)": p_value
    }


# 1. Результат
anova_results = dispersion_analysis(res1)


def coefficient_significance_analysis(results_mnk):
    """
    Возвращает статистические параметры значимости для каждого коэффициента B.
    """
    C = results_mnk['C']
    B = results_mnk['B']
    n = results_mnk['n']
    m = results_mnk['m']
    RSS = results_mnk['RSS']

    # 1. Средний квадрат остатков (Остаточная дисперсия)
    MS_RSS = RSS / (n - m - 1)

    # 2. Ковариационная матрица коэффициентов
    # Cov(B) = MS_RSS * (C^T * C)^-1
    CtC_inv = np.linalg.pinv(C.T @ C)
    Cov_B = MS_RSS * CtC_inv

    # 3. Стандартные ошибки (s_j)
    # s_j = sqrt(диагональные элементы Cov_B)
    std_errors = np.sqrt(np.diag(Cov_B)).reshape(-1, 1)

    # 4. T-критерий
    T_stats = B / std_errors

    # Степени свободы для T-критерия
    df_t = n - m - 1

    # 5. p-значение (двусторонний тест)
    # Используем sf (Survival function) и умножаем на 2
    p_values = t.sf(np.abs(T_stats), df_t) * 2

    # Параметры theta_j (в данном случае это просто B_j)
    theta_params = B

    return {
        "B": theta_params.flatten(),  # Параметры (theta_j)
        "std_errors": std_errors.flatten(),  # Стандартные ошибки (s_j)
        "T_stats": T_stats.flatten(),  # Значения T-критерия (T_j)
        "p_values": p_values.flatten()  # Вероятности (p-значения)
    }


# 2. Результат
t_test_results = coefficient_significance_analysis(res1)


def general_model_analysis(results_mnk):
    """
    Возвращает R2, R_adj^2, множественный R и стандартную ошибку s.
    """
    R2 = results_mnk['R2']
    R2_adj = results_mnk['R2_adj']
    s = results_mnk['s']

    # Множественный коэффициент корреляции R = sqrt(R^2)
    R_multiple = np.sqrt(R2)

    return {
        "R^2": R2,
        "R_adj^2": R2_adj,
        "R_multiple": R_multiple,
        "s (Стандартная ошибка)": s
    }


def print_step_results(step_num, factors, results, t_test):
    """Выводит результаты для текущего шага."""
    anova_res = dispersion_analysis(results)
    metrics_res = general_model_analysis(results)

    print(f"\n ШАГ {step_num}: МОДЕЛЬ С ФАКТОРАМИ {factors}")

    # 3. Общий анализ
    print("### 3. Общий анализ регрессионной модели")
    print(
        f"R²: {metrics_res['R^2']:.4f}, R_adj²: {metrics_res['R_adj^2']:.4f}, R: {metrics_res['R_multiple']:.4f}, s: {metrics_res['s (Стандартная ошибка)']:.2f}")

    # 1. Дисперсионный анализ
    print("\n### 1. Дисперсионный анализ (F-критерий)")
    print(f"F-критерий: {anova_res['F-критерий']:.2f}, p-значение (F): {anova_res['p-значение (F)']:.6f}")

    # 2. Анализ статистической значимости
    print("\n### 2. Анализ статистической значимости коэффициентов (T-критерий)")
    coeffs_df = pd.DataFrame({
        'Фактор': ['B0 (Intercept)'] + factors,
        'p-значение': t_test['p_values'],
        'Значение B': t_test['B']
    })
    coeffs_df['Значимость (p <= 0.1)'] = coeffs_df['p-значение'].round(1) <= 0.1
    print(coeffs_df.round(4).to_string(index=False))
    print("-----------------------------------------------------------------")

    return coeffs_df


def stepwise_regression(X_initial, Y, initial_factor_names, alpha=0.1):
    """
    Проводит шаговый регрессионный анализ, исключая факторы с p > alpha.
    """
    current_factors = initial_factor_names[:]  # Копируем список
    step = 0

    while True:
        step += 1

        if not current_factors:
            print("\nКритическая ошибка: В модели не осталось факторов.")
            break

        X_current = X_initial[current_factors].values

        # 1. Расчет МНК и метрик
        results = mnk(len(current_factors), X_current, Y)
        t_test = coefficient_significance_analysis(results)

        # 2. Вывод результатов для текущего шага
        coeffs_df = print_step_results(step, current_factors, results, t_test)

        # 3. Поиск наименее значимого фактора (исключаем B0, так как его не удаляют)
        # Начинаем со второго элемента (индекс 1)
        insignificant_factors = coeffs_df[coeffs_df['p-значение']> alpha]

        if insignificant_factors.empty:
            print(f"\nПроцесс завершен: Все оставшиеся факторы ({current_factors}) значимы (p <= {alpha}).")
            break

        # Исключаем фактор с наибольшим p-значением
        factor_to_remove_row = insignificant_factors.sort_values(by='p-значение', ascending=False).iloc[0]
        factor_to_remove = factor_to_remove_row['Фактор']

        print(
            f"\nИсключаем фактор {factor_to_remove} (p={factor_to_remove_row['p-значение']:.4f}), так как p > {alpha}.")
        current_factors.remove(factor_to_remove)
    return results, current_factors


# ------------------- ЗАПУСК АНАЛИЗА -------------------
final_results, final_factors = stepwise_regression(
    df, # Передаем сам DataFrame
    Y_multi, # Передаем Y-вектор для расчета (11 строк)
    ['x1', 'x2', 'x3', 'x4'],
    alpha=0.1
)

# Финальное уравнение
B_final = final_results['B'].flatten()
final_equation = f"y = {B_final[0]:.4f} "
for i, factor in enumerate(final_factors):
    sign = "+" if B_final[i + 1] >= 0 else "-"
    final_equation += f"{sign} {abs(B_final[i + 1]):.4f} * {factor} "

print("\nОКОНЧАТЕЛЬНАЯ МОДЕЛЬ")
print(f"Финальные факторы: {final_factors}")
print(f"Уравнение регрессии: {final_equation.strip()}")
print(f"R² финальной модели: {final_results['R2']:.4f}")