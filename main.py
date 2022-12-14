import numpy as np
from numpy import exp, pi, cos, sin
from numpy.lib import emath
import matplotlib.pyplot as plt


# N - число слоев
def calc_r(angle):
    N = 8

    # Dielectric conductibity of first layer

    epsilon0 = 1

    # 1 layer
    epsilon1 = (1.45 + 1j * 2 * 10**(-4))**2
    # 2 layer
    epsilon2 = (2.08 + 1j * 2 * 10**(-4))**2
    # last layer
    epsilon3 = 1.5**2

    d10 = 180 * 10**(-9)
    d1 = 210 * 10**(-9)
    d2 = 70 * 10**(-9)
    d_last = 150 * 10**(-6)

    lamb = 600 * 10 * (-9)

    k0 = 2 * pi / lamb

    k01x = k0 * cos(angle)
    k01z = k0 * sin(angle)

    def multiply_matrix(a, b):
        t11 = a[0][0] * b[0][0] + a[0][1] * b[1][0]
        t12 = a[0][0] * b[0][1] + a[0][1] * b[1][1]
        t21 = a[1][0] * b[0][0] + a[1][1] * b[1][0]
        t22 = a[1][0] * b[0][1] + a[1][1] * b[1][1]
        return [[t11, t12], [t21, t22]]

    def calc_k0_n(k0, epsil_n):
        return k0 * emath.sqrt(epsil_n)

    epsilons_all = [epsilon0]
    d_all = []
    for i in range(N):
        if i % 2 == 0:
            epsilons_all.append(epsilon1)
            d_all.append(d1)
        else:
            epsilons_all.append(epsilon2)
            d_all.append(d2)
    d_all[0] = d10
    epsilons_all.append(epsilon3)
    # d_all.append(d_last)

    # print(epsilons_all)
    # print(d_all)

    k_full = [k0]
    for i in range(N + 1):
        k_full.append(calc_k0_n(k_full[i], epsilons_all[i + 1]))

    # print(k_full)
    kz_full = [k01z]
    for i in range(N + 1):
        kz_full.append(emath.sqrt(k_full[i + 1]**2 - k01x ** 2))

    # print(kz_full)

    def calc_T_nn(epsil_n0, epsil_n1, kz_n0, kz_n1):
        t11 = kz_n0 * epsil_n1 + kz_n1 * epsil_n0
        t12 = -kz_n0 * epsil_n1 + kz_n1 * epsil_n0
        t11_1 = t11 / (2 * kz_n1 * epsil_n0)
        t12_1 = t12 / (2 * kz_n1 * epsil_n0)
        return [[t11_1, t12_1], [t12_1, t11_1]]

    def calc_T_dn(kz_n, d_n):
        t11 = exp(1j * kz_n * d_n)
        t22 = exp(-1j * kz_n * d_n)
        return [[t11, 0], [0, t22]]

    T_tot = [[1, 0], [0, 1]]
    for i in range(N + 1):
        epsilons_all_rev = epsilons_all[::-1]
        kz_full_rev = kz_full[::-1]
        d_all_rev = d_all[::-1]
        T_i_nn = calc_T_nn(epsilons_all_rev[i + 1], epsilons_all_rev[i], kz_full_rev[i + 1], kz_full_rev[i])
        if i < len(d_all):
            T_i_d = calc_T_dn(kz_full_rev[i + 1], d_all_rev[i])
            T_tot = multiply_matrix(T_tot, T_i_nn)
            T_tot = multiply_matrix(T_tot, T_i_d)
        else:
            T_tot = multiply_matrix(T_tot, T_i_nn)

    r = -1 * T_tot[1][0] / T_tot[1][1]
    return r


print(calc_r(30 * 2 * pi / 360))

angles = np.linspace(0, pi / 2, 100)
r_list = calc_r(angles)
fig, ax = plt.subplots()
ax.plot(angles, abs(r_list))
plt.show()