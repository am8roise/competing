import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


def delta(epsilon):
	return 0.5 * np.arccos(-2.0 * epsilon / np.pi) - np.pi / 4.0


def velocity(epsilon, a):
	return 8.0 * a * np.sqrt(1.0 - 4.0 * epsilon**2 / np.pi**2)


def initial_couplings(epsilon, a):
	d = delta(epsilon)

	gamma_A = 0.0
	gamma_Cp = -16.0 * (1.0 + np.sin(2.0 * d))
	gamma_Cm = -16.0 * (1.0 - np.sin(2.0 * d))
	gamma_D = 16.0 * (1.0 + np.sin(2.0 * d))
	gamma_E = -32.0 * np.sin(2.0 * d)
	gamma_F = 16.0 * (1.0 - np.sin(2.0 * d))

	g_y = 0.5 * (gamma_A - gamma_D)
	g_x = -0.5 * (gamma_A + gamma_D)
	g_c = -gamma_Cp
	g_m = gamma_Cm
	g_j = -0.5 * (gamma_E + gamma_F)
	g_i = -0.5 * (gamma_E - gamma_F)

	prefactor = epsilon * a / (2.0 * np.pi * velocity(epsilon, a))

	return prefactor * np.array([g_y, g_x, g_c, g_m, g_j, g_i])


def beta_functions(ell, g):
	g_y, g_x, g_c, g_m, g_j, g_i = g

	dg_y = -2.0 * g_c * g_y + 2.0 * (g_j - g_i) * g_x
	dg_x = -2.0 * g_c * g_x + 2.0 * (g_j - g_i) * g_y
	dg_c = -2.0 * (g_y**2 + g_x**2) + g_m**2
	dg_m = 2.0 * (g_c + g_i - g_j) * g_m
	dg_j = 2.0 * g_x * g_y - 0.5 * g_m**2
	dg_i = -2.0 * g_x * g_y + 0.5 * g_m**2

	return [dg_y, dg_x, dg_c, dg_m, dg_j, dg_i]


def main():

	epsilon = 0.01
	a = 1.0
	threshold = 1.0
	ell_max = 10 / epsilon
	max_step = 0.01

	g_initial = initial_couplings(epsilon, a)

	def stop_flow(ell, g):
		return threshold - np.max(np.abs(g))

	stop_flow.terminal = True
	stop_flow.direction = -1

	solution = solve_ivp(
		beta_functions,
		[0.0, ell_max],
		g_initial,
		events=stop_flow,
		max_step=max_step,
		rtol=1e-10,
		atol=1e-12,
	)

	names = ["g_Y", "g_X", "g_C", "g_M", "g_J", "g_I"]

	g_final = solution.y[:, -1]
	g_c_final = g_final[2]

	for i in range(6):
	    print(f"{names[i] + ' / g_C'} = {g_final[i] / g_c_final:+.3f}")


if __name__ == "__main__":
	main()
