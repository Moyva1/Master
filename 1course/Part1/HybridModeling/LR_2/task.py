import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Интенсивности переходов
lmb01, lmb03 = 0.34, 0.20
lmb12        = 0.70
lmb20        = 0.60
lmb34, lmb35 = 0.10, 0.85
lmb40        = 0.80          
lmb52, lmb56 = 0.30, 0.70
lmb62        = 1.00

def system(t, p):
    p0, p1, p2, p3, p4, p5, p6 = p
    dp0 = lmb20*p2 + lmb40*p4 - (lmb01 + lmb03)*p0
    dp1 = lmb01*p0 - lmb12*p1
    dp2 = lmb12*p1 + lmb52*p5 + lmb62*p6 - lmb20*p2
    dp3 = lmb03*p0 - (lmb34 + lmb35)*p3
    dp4 = lmb34*p3 - lmb40*p4
    dp5 = lmb35*p3 - (lmb52 + lmb56)*p5
    dp6 = lmb56*p5 - lmb62*p6
    return [dp0, dp1, dp2, dp3, dp4, dp5, dp6]

p0 = [1, 0, 0, 0, 0, 0, 0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1500)

sol = solve_ivp(system, t_span, p0, t_eval=t_eval,
                method='RK45', rtol=1e-8, atol=1e-10)

legend = ['p0 нормальная работа', 'p1 анализ производительности',
          'p2 профилактика', 'p3 анализ отказа',
          'p4 невосстанавливаемое состояние', 'p5 ремонт',
          'p6 восстановление']

plt.figure(figsize=(11, 6))
for i in range(7):
    plt.plot(sol.t, sol.y[i], label=legend[i])
plt.xlabel('Время t')
plt.ylabel('Вероятность p(t)')
plt.title('Вероятности состояний')
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()

print("Вероятности на t = 10:")
for i, name in enumerate(legend):
    print(f"  {name}: {sol.y[i, -1]:.6f}")
print("  Сумма:", sol.y[:, -1].sum())

# Проверка сохранения нормировки на всём интервале
total = sol.y.sum(axis=0)
print(f"\nПроверка нормировки: min = {total.min():.10f}, max = {total.max():.10f}")

# Проверка устойчивости
A = np.array([
    [-(lmb01+lmb03),  0,        lmb20,  0,          lmb40,  0,        0     ],
    [ lmb01,         -lmb12,    0,      0,          0,      0,        0     ],
    [ 0,              lmb12,   -lmb20,  0,          0,      lmb52,    lmb62 ],
    [ lmb03,          0,        0,     -(lmb34+lmb35), 0,   0,        0     ],
    [ 0,              0,        0,      lmb34,     -lmb40,  0,        0     ],
    [ 0,              0,        0,      lmb35,      0,    -(lmb52+lmb56), 0],
    [ 0,              0,        0,      0,          0,      lmb56,   -lmb62 ]
])
eig = np.linalg.eigvals(A)
print("\nСобственные числа (Re):")
for e in sorted(eig.real, reverse=True):
    print(f"  {e:.4f}")
print("max Re =", max(eig.real))