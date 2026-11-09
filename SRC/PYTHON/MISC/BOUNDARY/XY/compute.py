import scipy.optimize as sco
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import argparse
import os

from tenet.utils.io import save_graph, format_value, format_sign
from tenet.utils.sty import COLORS

from sum import compute_sum

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

data_filename = 'sum.dat'

if not os.path.exists(data_filename):

	S = []
	Ns = np.arange(20, 30+1, 2)
	Ns = np.append(Ns, np.arange(100, 1000+1, 100))
	Ns = np.unique(Ns)
	print(Ns)

	print("No data found.")
	print(f"Start computation for 2N = [{min(Ns)}, {max(Ns)}]...")

	for n in Ns:
		print(f"\t• 2N = {n}...")
		
		s = compute_sum(n)
		S.append(s)

	engine = {'2N': Ns, 'sum': np.array(S)}

	with open(data_filename, 'wb') as file:
		pickle.dump(engine, file)

else:

	with open(data_filename, 'rb') as file:
		engine = pickle.load(file)

	N = engine['2N']
	S = engine['sum'] * [(-1) ** (n // 2) for n in N]

	def f(N, a, b):
		return a * np.log(N - 1) + b

	fig, ax = plt.subplots(figsize=(4, 2))
		
	ind = np.where((N >= 10) & (N <= 30))[0]

	p, = ax.plot(N[ind], S[ind], linewidth=0, marker='o', markersize=10, color=COLORS[0], zorder=100, label=r"$2N\in[20,30]$")

	ind = np.where((N >= 100) & (N % 100 == 0))[0]

	p, = ax.plot(N[ind], S[ind], linewidth=0, marker='o', markersize=10, color=COLORS[2], zorder=100, label=r"$2N\in[100,1000]$")

	# ----- Small 2N

	ind = np.where((N >= 20) & (N <= 30))[0]

	popt, pcov = sco.curve_fit(f, N[ind], np.log(S[ind]), [-3, 0])
	perr = np.sqrt(np.diag(pcov))

	print(popt, perr)
	
	a = np.exp(popt[1])
	da = perr[1] * np.exp(popt[1])

	o, = ax.plot(N[ind], np.exp(f(N[ind], *popt)), linestyle="-", linewidth=2, color=COLORS[4], markersize=0, zorder=101, label=fr"$({format_value(a, da)}) \times (2N-1)^{{{format_value(popt[0], perr[0])}}}$")

	# ----- Large 2N

	ind = np.where((N >= 100) & (N % 100 == 0))[0]

	popt, pcov = sco.curve_fit(f, N[ind], np.log(S[ind]), [-3, 0])
	perr = np.sqrt(np.diag(pcov))

	print(popt, perr)
	
	a = np.exp(popt[1])
	da = perr[1] * np.exp(popt[1])

	o, = ax.plot(N[ind], np.exp(f(N[ind], *popt)), linestyle=(0, (2, 3)), linewidth=2, color=COLORS[4], markersize=0, zorder=101, label=fr"$({format_value(a, da)}) \times (2N-1)^{{{format_value(popt[0], perr[0])}}}$")
	
	plt.xlabel(r"$2N$")
	plt.ylabel(r"$(-1)^{N}\,S$")

	ax.set_xscale("log")
	ax.set_yscale("log")

	ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([10 ** k for k in range(-10,1)]))
	ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

	ax.legend(
		loc="lower left",
		ncol=1,
		handlelength=1.5,
		handletextpad=0.8,
		labelspacing=0.5,
		bbox_to_anchor=(0.4, 0.7)).set_zorder(1001)

	plt.draw()

	if args.save:
		save_graph(fig, "./", f"sum")
		fig.clf()

	if args.show:
		plt.show()

	plt.close()