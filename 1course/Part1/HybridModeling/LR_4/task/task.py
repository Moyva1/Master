# -*- coding: utf-8 -*-
"""
Лабораторная работа №4.
Вариант 7: Отрицательное биномиальное распределение (распределение Паскаля).

Гибридная аналитическая модель:
    Модель 1 — SymPy  (символьная аналитика)     ← аналог Maple
    Модель 2 — SciPy  (численная аналитика)      ← аналог Matlab
    Модель 3 — NumPy  (Монте-Карло имитация)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
from scipy.special import gammaln
import sympy as sp
import pandas as pd

# --- шрифты с кириллицей ---
try:
    plt.rcParams["font.family"] = "DejaVu Sans"
except Exception:
    pass


# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================
R_VAL = 5                # число успехов
P_VAL = 0.4              # вероятность успеха
N_SAMPLES = 100_000      # размер выборки Монте-Карло
SEED = 42
R_PLOT = [5, 10, 15]     # для зависимостей от p

print("=" * 65)
print("ЛАБОРАТОРНАЯ РАБОТА №4 — Вариант 7")
print("Отрицательное биномиальное распределение (распределение Паскаля)")
print(f"Параметры: r = {R_VAL}, p = {P_VAL}")
print("=" * 65)


# ============================================================
# 2. МОДЕЛЬ 1 — SymPy (символьная аналитика)
# ============================================================
print("\n[МОДЕЛЬ 1] SymPy — символьная аналитика")
print("-" * 65)

k = sp.Symbol("k", integer=True, nonnegative=True)
r = sp.Symbol("r", integer=True, positive=True)
p = sp.Symbol("p", positive=True)

# PMF: P(X=k) = C(k+r-1, k) * (1-p)^k * p^r
P_sym = sp.binomial(k + r - 1, k) * (1 - p) ** k * p ** r
print(f"PMF: P(X=k) = {P_sym}")

# Проверка нормировки при r=R_VAL (символьно по p)
P_fixed_r = P_sym.subs(r, R_VAL)
try:
    norm_check = sp.simplify(sp.summation(P_fixed_r, (k, 0, sp.oo)))
    print(f"Нормировка (r={R_VAL}): sum P(k) = {norm_check}")
except Exception as e:
    print(f"(символьная сумма нормировки не взялась: {e})")

# Математическое ожидание (символьно при фиксированном r)
try:
    M_r = sp.simplify(sp.summation(k * P_fixed_r, (k, 0, sp.oo)))
    print(f"M (r={R_VAL}): {M_r}")
except Exception:
    M_r = R_VAL * (1 - p) / p
    print(f"M (r={R_VAL}): {M_r}  [известная формула]")

# Дисперсия (символьно при фиксированном r)
try:
    D_r = sp.simplify(sp.summation((k - M_r) ** 2 * P_fixed_r, (k, 0, sp.oo)))
    print(f"D (r={R_VAL}): {D_r}")
except Exception:
    D_r = R_VAL * (1 - p) / p ** 2
    print(f"D (r={R_VAL}): {D_r}  [известная формула]")

# Общие символьные формулы (для произвольного r — подтверждены теорией)
M_gen = r * (1 - p) / p
D_gen = r * (1 - p) / p ** 2
A_gen = (2 - p) / sp.sqrt(r * (1 - p))
E_gen = 6 / r + p ** 2 / (r * (1 - p))

print("\nОбщие формулы (теория, SymPy):")
print(f"  M(r,p) = {M_gen}")
print(f"  D(r,p) = {D_gen}")
print(f"  A(r,p) = {A_gen}")
print(f"  E(r,p) = {E_gen}")

# Численные значения при (R_VAL, P_VAL)
M_sym = float(M_gen.subs({r: R_VAL, p: P_VAL}))
D_sym = float(D_gen.subs({r: R_VAL, p: P_VAL}))
Std_sym = float(np.sqrt(D_sym))
A_sym = float(A_gen.subs({r: R_VAL, p: P_VAL}))
E_sym = float(E_gen.subs({r: R_VAL, p: P_VAL}))

print(f"\nЗначения при r={R_VAL}, p={P_VAL}:")
print(f"  M   = {M_sym:.6f}")
print(f"  D   = {D_sym:.6f}")
print(f"  Std = {Std_sym:.6f}")
print(f"  A   = {A_sym:.6f}")
print(f"  E   = {E_sym:.6f}")


def entropy_numeric(r_val: int, p_val: float) -> float:
    """Численное значение энтропии (замкнутой формулы нет)."""
    mean = r_val * (1 - p_val) / p_val
    std = np.sqrt(r_val * (1 - p_val) / p_val ** 2)
    K = int(mean + 12 * std) + 100
    kk = np.arange(K)
    log_pk = (
        gammaln(kk + r_val)
        - gammaln(kk + 1)
        - gammaln(r_val)
        + kk * np.log(1 - p_val)
        + r_val * np.log(p_val)
    )
    pk = np.exp(log_pk)
    pk = pk[pk > 1e-300]
    return float(-np.sum(pk * np.log(pk)))


H_sym = entropy_numeric(R_VAL, P_VAL)
print(f"  H   = {H_sym:.6f}   (численно, SymPy-модель)")


# ============================================================
# 3. МОДЕЛЬ 2 — SciPy (численная аналитика)
# ============================================================
print("\n[МОДЕЛЬ 2] SciPy — численная аналитика")
print("-" * 65)

M_scipy = float(nbinom.mean(R_VAL, P_VAL))
D_scipy = float(nbinom.var(R_VAL, P_VAL))
Std_scipy = float(nbinom.std(R_VAL, P_VAL))
A_scipy = float(nbinom.stats(R_VAL, P_VAL, moments="s"))
E_scipy = float(nbinom.stats(R_VAL, P_VAL, moments="k"))
H_scipy = float(nbinom.entropy(R_VAL, P_VAL))

print(f"  M   = {M_scipy:.6f}")
print(f"  D   = {D_scipy:.6f}")
print(f"  Std = {Std_scipy:.6f}")
print(f"  A   = {A_scipy:.6f}")
print(f"  E   = {E_scipy:.6f}")
print(f"  H   = {H_scipy:.6f}")


# ============================================================
# 4. МОДЕЛЬ 3 — Монте-Карло
# ============================================================
print("\n[МОДЕЛЬ 3] Монте-Карло")
print("-" * 65)

np.random.seed(SEED)
sample = np.random.negative_binomial(R_VAL, P_VAL, size=N_SAMPLES)

M_mc = float(np.mean(sample))
D_mc = float(np.var(sample, ddof=1))
Std_mc = float(np.sqrt(D_mc))
A_mc = float(np.mean((sample - M_mc) ** 3) / Std_mc ** 3)
E_mc = float(np.mean((sample - M_mc) ** 4) / D_mc ** 2 - 3)

print(f"  n_samples = {N_SAMPLES}")
print(f"  M   = {M_mc:.6f}")
print(f"  D   = {D_mc:.6f}")
print(f"  Std = {Std_mc:.6f}")
print(f"  A   = {A_mc:.6f}")
print(f"  E   = {E_mc:.6f}")
print("  (H для Монте-Карло не считаем: требует непараметрической оценки плотности)")


# ============================================================
# 5. СВОДНАЯ ТАБЛИЦА
# ============================================================
print("\n[СВОДНАЯ ТАБЛИЦА]")
print("-" * 65)

df_result = pd.DataFrame(
    {
        "Характеристика": ["Mean", "Variance", "Std", "Skewness", "Kurtosis", "Entropy"],
        "SymPy":  [M_sym,    D_sym,    Std_sym,    A_sym,    E_sym,    H_sym],
        "SciPy":  [M_scipy,  D_scipy,  Std_scipy,  A_scipy,  E_scipy,  H_scipy],
        "Монте-Карло": [M_mc, D_mc, Std_mc, A_mc, E_mc, np.nan],
    }
)
df_result["|Δ(SymPy−SciPy)|"] = (df_result["SymPy"] - df_result["SciPy"]).abs()

print(df_result.round(6).to_string(index=False))
df_result.to_csv("lab4_results.csv", index=False, encoding="utf-8-sig")
print("\nТаблица сохранена: lab4_results.csv")


# ============================================================
# 6. ГРАФИКИ МОДЕЛИ SymPy
# ============================================================
p_grid = np.linspace(0.01, 0.99, 150)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for rv in R_PLOT:
    axes[0].plot(p_grid, rv * (1 - p_grid) / p_grid, lw=2, label=f"r={rv}")
    axes[1].plot(p_grid, rv * (1 - p_grid) / p_grid ** 2, lw=2, label=f"r={rv}")
    axes[2].plot(p_grid, [entropy_numeric(rv, pv) for pv in p_grid], lw=2, label=f"r={rv}")

titles = ["M(p) — модель SymPy", "D(p) — модель SymPy", "H(p) — модель SymPy (числ.)"]
ylabels = ["M", "D", "H"]
for ax, t, yl in zip(axes, titles, ylabels):
    ax.set_xlabel("p"); ax.set_ylabel(yl)
    ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("sympy_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nСохранён график: sympy_model.png")


# ============================================================
# 7. ГРАФИКИ МОДЕЛИ SciPy
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for rv in R_PLOT:
    axes[0].plot(p_grid, nbinom.mean(rv, p_grid), lw=2, label=f"r={rv}")
    axes[1].plot(p_grid, nbinom.var(rv, p_grid), lw=2, label=f"r={rv}")
    axes[2].plot(p_grid, [nbinom.entropy(rv, pv) for pv in p_grid], lw=2, label=f"r={rv}")

titles = ["M(p) — модель SciPy", "D(p) — модель SciPy", "H(p) — модель SciPy"]
for ax, t, yl in zip(axes, titles, ylabels):
    ax.set_xlabel("p"); ax.set_ylabel(yl)
    ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("scipy_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранён график: scipy_model.png")


# ============================================================
# 8. СРАВНЕНИЕ SymPy vs SciPy (при r = R_VAL)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].plot(p_grid, R_VAL * (1 - p_grid) / p_grid, "b-", lw=2.5, label="SymPy")
axes[0].plot(p_grid, nbinom.mean(R_VAL, p_grid), "r--", lw=2, label="SciPy")

axes[1].plot(p_grid, R_VAL * (1 - p_grid) / p_grid ** 2, "b-", lw=2.5, label="SymPy")
axes[1].plot(p_grid, nbinom.var(R_VAL, p_grid), "r--", lw=2, label="SciPy")

axes[2].plot(p_grid, [entropy_numeric(R_VAL, pv) for pv in p_grid], "b-", lw=2.5, label="SymPy (числ.)")
axes[2].plot(p_grid, [nbinom.entropy(R_VAL, pv) for pv in p_grid], "r--", lw=2, label="SciPy")

titles = [f"M(p), r={R_VAL}", f"D(p), r={R_VAL}", f"H(p), r={R_VAL}"]
for ax, t, yl in zip(axes, titles, ylabels):
    ax.set_xlabel("p"); ax.set_ylabel(yl)
    ax.set_title(t); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранён график: comparison.png")


# ============================================================
# 9. PMF для одного набора параметров
# ============================================================
k_plot = np.arange(0, 30)
pmf_sym_vals = np.array(
    [
        float(sp.binomial(kk + R_VAL - 1, kk) * (1 - P_VAL) ** kk * P_VAL ** R_VAL)
        for kk in k_plot
    ]
)
pmf_scipy_vals = nbinom.pmf(k_plot, R_VAL, P_VAL)

plt.figure(figsize=(9, 4))
plt.bar(k_plot - 0.2, pmf_sym_vals, width=0.4, label="SymPy",
        color="steelblue", edgecolor="black")
plt.bar(k_plot + 0.2, pmf_scipy_vals, width=0.4, label="SciPy",
        color="orange", edgecolor="black")
plt.xlabel("k — число неудач до r-го успеха")
plt.ylabel("P(X=k)")
plt.title(f"PMF отрицательного биномиального распределения, r={R_VAL}, p={P_VAL}")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pmf_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Сохранён график: pmf_comparison.png")


# ============================================================
# 10. ИТОГИ
# ============================================================
print("\n" + "=" * 65)
print("ГОТОВО. Все результаты сохранены в текущей папке:")
print("  lab4_results.csv    — сводная таблица")
print("  sympy_model.png     — 3 графика модели SymPy")
print("  scipy_model.png     — 3 графика модели SciPy")
print("  comparison.png      — наложение SymPy и SciPy")
print("  pmf_comparison.png  — сравнение PMF")
print("=" * 65)