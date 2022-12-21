import numpy as np
from numpy import exp, pi, cos, sin
from numpy.lib import emath
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# N - число слоев
def calc_r(angle, lamb, periods, type, k01x):
    N = 2*periods

    # Dielectric conductibity of first layer

    epsilon0 = 1

    # 1 layer
    epsilon1 = (1.45 + 1j * 2 * 10 ** (-4)) ** 2
    # 2 layer
    epsilon2 = (2.08 + 1j * 2 * 10 ** (-4)) ** 2
    # last layer
    epsilon3 = 1.5 ** 2

    d10 = 180 * 10 ** (-9)
    d1 = 210 * 10 ** (-9)
    d2 = 70 * 10 ** (-9)
    d_last = 150 * 10 ** (-6)

    k0 = 2 * pi / (lamb*10**(-6))

    if type == 'angle':
        k01x = k0 * sin(angle)

    k01z = emath.sqrt((k0*epsilon0)**2-k01x**2)

    def multiply_matrix(a, b):
        t11 = a[0][0] * b[0][0] + a[0][1] * b[1][0]
        t12 = a[0][0] * b[0][1] + a[0][1] * b[1][1]
        t21 = a[1][0] * b[0][0] + a[1][1] * b[1][0]
        t22 = a[1][0] * b[0][1] + a[1][1] * b[1][1]
        return [[t11, t12], [t21, t22]]

    kz1 = emath.sqrt((k0 * emath.sqrt(epsilon1)) ** 2 - k01x ** 2)
    kz2 = emath.sqrt((k0 * emath.sqrt(epsilon2)) ** 2 - k01x ** 2)
    kz3 = emath.sqrt((k0 * emath.sqrt(epsilon3)) ** 2 - k01x ** 2)

    def calc_T_nn(epsil_n0, epsil_n1, kz_n0, kz_n1):
        t11 = kz_n0 * epsil_n1 + kz_n1 * epsil_n0
        t12 = -1 * kz_n0 * epsil_n1 + kz_n1 * epsil_n0
        t11_1 = t11 / (2 * kz_n1 * epsil_n0)
        t12_1 = t12 / (2 * kz_n1 * epsil_n0)
        return [[t11_1, t12_1], [t12_1, t11_1]]

    def calc_T_dn(kz_n, d_n):
        t11 = exp(1j * kz_n * d_n)
        t22 = exp(-1j * kz_n * d_n)
        return [[t11, 0], [0, t22]]

    T_tot = [[1, 0], [0, 1]]
    Td01 = calc_T_dn(kz1, d10)
    Td1 = calc_T_dn(kz1, d1)
    Td2 = calc_T_dn(kz2, d2)

    T01 = calc_T_nn(epsilon0, epsilon1, k01z, kz1)
    T12 = calc_T_nn(epsilon1, epsilon2, kz1, kz2)
    T21 = calc_T_nn(epsilon2, epsilon1, kz2, kz1)
    T23 = calc_T_nn(epsilon2, epsilon3, kz2, kz3)

    T_tot = multiply_matrix(T_tot, T23)
    T_tot = multiply_matrix(T_tot, Td2)
    for i in range(N - 2):
        if i % 2 == 0:
            T_tot = multiply_matrix(T_tot, T12)
            T_tot = multiply_matrix(T_tot, Td1)
        else:
            T_tot = multiply_matrix(T_tot, T21)
            T_tot = multiply_matrix(T_tot, Td2)
    T_tot = multiply_matrix(T_tot, T12)
    T_tot = multiply_matrix(T_tot, Td01)
    T_tot = multiply_matrix(T_tot, T01)


    # for i in range(N + 1):
    #     epsilons_all_rev = epsilons_all[::-1]
    #     kz_full_rev = kz_full[::-1]
    #     d_all_rev = d_all[::-1]
    #     T_i_nn = calc_T_nn(epsilons_all_rev[i + 1], epsilons_all_rev[i], kz_full_rev[i + 1], kz_full_rev[i])
    #     if i < len(d_all):
    #         T_i_d = calc_T_dn(kz_full_rev[i + 1], d_all_rev[i])
    #         T_tot = multiply_matrix(T_tot, T_i_nn)
    #         T_tot = multiply_matrix(T_tot, T_i_d)
    #     else:
    #         T_tot = multiply_matrix(T_tot, T_i_nn)

    r = -1 * T_tot[1][0] / T_tot[1][1]
    return r

margins = {
    "left"   : 0.10,
    "bottom" : 0.140,
    "right"  : 0.990,
    "top"    : 1
}
matplotlib.rcParams.update({'font.size': 18})
plt.rcParams['text.usetex'] = True

#Dependence from angle:
fig1 = plt.figure(1)
ax1 = plt.gca()
fig1.subplots_adjust(**margins)
#Wave length in mkm
lamb=0.8

#Periods in 1D photonic crystall
P=4

Angles = np.linspace(0, pi / 2, 1000)
R_list=calc_r(Angles, lamb, P, 'angle', 0)
plt.plot(Angles, np.imag(R_list), label=r'$Im(R^{p})$', color='firebrick', lw='1.5')
ax1.set_xlabel(r'Angle (rad)')

plt.legend()
plt.grid()
plt.savefig(f"Plot_Im_R(angle)_1DPC_Period_{P}.pdf", format="pdf", bbox_inches="tight")
plt.show()

pd.concat([pd.DataFrame(Angles), pd.DataFrame(np.real(R_list)), pd.DataFrame(np.imag(R_list))], axis=1).to_csv('R_1DPC.csv', index=False)


#Dependence from lambda:
fig2=plt.figure(2)
ax2=plt.gca()
fig2.subplots_adjust(**margins)
Angle=30
Lambs=np.linspace(0.1, 0.8, 1000)
R_list=calc_r(Angle*pi/180, Lambs, P, 'angle', 0)
plt.plot(Lambs, R_list*np.conj(R_list), label=r'$|R^{p}|^2$', color='royalblue', lw='1.5')
ax2.set_xlabel(r'$\lambda \;\mu m$')
plt.legend()
plt.grid()
plt.savefig(f"Plot_R(lamb)_1DPC_Period_{P}.pdf", format="pdf", bbox_inches="tight")
plt.show()
#
#
# Plot graph |R_p^2| from Array of k_x
# kx array starts from: a*k1
a = 0

# kx array ends with: b*k1
b = 4

# Amount of dots between a and b:
resolution = 500

# Lambda of wave:
lamb = 0.6

X1 = np.linspace(a, b, resolution)
fig3 = plt.figure(3)
fig3.subplots_adjust(**margins)
ax3 = plt.gca()
angle = 0
k1 = 2 * pi / (lamb * 10 ** (-6))
X = X1 * k1

plt.plot(X, calc_r(0, lamb, P, 'eva', X) * np.conj(calc_r(0, lamb, P, 'eva', X)),
         label=r'$|R^{p}|^{2}$ for 1DPC', color='royalblue', lw=1.5)
#plt.plot(X, np.abs(calc_r(0, lamb, P, 'eva', X)),
#         label=r'$|R^{p}|$ for 1DPC', color='royalblue', lw=1.5)
plt.legend()
plt.grid()
ax3.set_xlabel(r'$k_{1x}$')
ax3.set_xlim(xmin=a*k1)
plt.hlines(y=0, xmin=0, xmax=k1, color='g', linestyle='-', linewidth=3)
plt.hlines(y=0, xmin=k1, xmax=5 * k1, color='r', linestyle='-', linewidth=3)
# ax2.set_ylim(ymin=0)

plt.xticks(np.arange(a*k1, b * k1 + 1, k1))
ax3.text((a+0.05) * k1, -0.25, 'propagating', style='italic')
ax3.text(3 * k1, -0.25, 'evanescent', style='italic')
plt.xticks(plt.xticks()[0], [r"$" + format(r / k1, ".2g") + r"k_1$" for r in plt.xticks()[0]])
plt.savefig(f"Plot_abs_Rp_2_evanicent_1DPC_periods_{P}.pdf", format="pdf", bbox_inches="tight")
plt.show()