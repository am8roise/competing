import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import scipy.optimize as sco
import matplotlib as mpl

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph, format_value
from tenet.utils.sty import COLORS
from tenet.linalg.sun import SUIrrep

su2 = SUIrrep(2, 2)
X, Y, Z = [2 * su2.generators[i] for i in range(3)]

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

start_x = 8

num_eigen = 10

target_chis = [112, 224, 448]

window_eta = 30
step_eta = 5

def running_exponent(sites, corr, window, step):

	sites = np.asarray(sites)
	corr = np.asarray(corr)

	mask = np.isfinite(corr) & (sites > 0)

	sites = sites[mask]
	corr = np.abs(corr[mask])

	j0 = []
	eta = []
	eta_err = []

	for a in range(0, len(sites) - window + 1, step):

		b = a + window

		x = np.log(sites[a:b])
		y = np.log(corr[a:b])

		try:
			popt, pcov = sco.curve_fit(lambda x, m, q: m * x + q, x, y)
			perr = np.sqrt(np.diag(pcov))

			j0.append(a)
			eta.append(-popt[0])
			eta_err.append(perr[0])

		except Exception:
			continue

	return np.array(j0), np.array(eta), np.array(eta_err)

def global_exponent(sites, corr, start_x, end_x):

	sites = np.asarray(sites)
	corr = np.abs(np.asarray(corr))

	_sites = sites[start_x:end_x]
	_corr = corr[start_x:end_x]

	mask = np.isfinite(_corr) & (_sites > 0)

	_sites = _sites[mask]
	_corr = _corr[mask]

	def f(x, a, b):
		return a * x + b

	popt, pcov = sco.curve_fit(
		f,
		np.log(_sites),
		np.log(_corr)
	)

	perr = np.sqrt(np.diag(pcov))

	eta = -popt[0]
	eta_err = perr[0]

	return eta, eta_err

def main():

	model_name = "B"
	simul_name = "PC+PDC/BOTTOM"

	boundary_name = "NONE"
	method_name = "IDMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

	records = {}

	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		if len(data_filenames) == 0:
			continue

		for i in range(len(data_filenames)):

			N = int(vals[i]["N"])
			chi = int(vals[i]["chi"])
			lam = float(vals[i]["lam"])
			theta = float(vals[i]["theta"])

			if target_chis is not None and chi not in target_chis:
				continue

			with open(os.path.join(data_folder, data_filenames[i]), "rb") as file:
				data = pickle.load(file)

			val_L = data['val_L'][:num_eigen]

			sort_L = np.argsort(np.abs(val_L))[::-1]
			val_L = val_L[sort_L]

			xi = - N / np.log(np.abs(val_L[1]))

			print(xi)

			end_x = int(xi / 10)

			corrX, corrY, corrZ = data["corrs"]
			
			corrX = np.abs(corrX)

			sites = np.array(range(1, len(corrX) + 1))

			eta, eta_err = global_exponent(
				sites,
				corrX,
				start_x,
				end_x,
			)

			j0_eta, eta_run, eta_run_err = running_exponent(
				sites,
				corrX,
				window=window_eta,
				step=step_eta,
			)

			key = (lam, theta)

			if key not in records:
				records[key] = []

			records[key].append({
				"N": N,
				"chi": chi,
				"lam": lam,
				"theta": theta,
				"sites": sites,
				"xi": xi,
				"corrX": corrX,
				"j0_eta": j0_eta,
				"eta_run": eta_run,
				"eta_run_err": eta_run_err,
				"eta": eta,
				"eta_err": eta_err,
				"filename": data_filenames[i],
			})

	for key, reps in records.items():

		lam, theta = key

		reps = sorted(reps, key=lambda r: r["chi"])

		if target_chis is None:
			chis_plot = sorted(np.unique([r["chi"] for r in reps]))
		else:
			chis_plot = target_chis

		reps_plot = [r for r in reps if r["chi"] in chis_plot]
		n_curves = len(reps_plot)

		if n_curves == 0:
			continue

		fig, ax = plt.subplots(figsize=(5, 2))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[3],COLORS[2],COLORS[0]], N=n_curves)

		markersizes = [6, 8, 10] * n_curves

		for n, rep in enumerate(reps_plot):

			color = cmap(n / max(n_curves - 1, 1))

			xi = rep["xi"]

			j0_eta = rep["j0_eta"]
			eta_run = rep["eta_run"]
			eta_run_err = rep["eta_run_err"]

			eta = rep["eta"]
			eta_err = rep["eta_err"]
			chi = rep["chi"]

			stop_inds = np.where((j0_eta > 2 * xi))[0]

			if len(stop_inds) > 0:
				stop_plot = stop_inds[0]
			else:
				stop_plot = len(j0_eta)

			j0_eta_plot = j0_eta[:stop_plot]
			eta_run_plot = eta_run[:stop_plot]
			eta_run_err_plot = eta_run_err[:stop_plot]

			ax.plot(
				j0_eta_plot,
				eta_run_plot,
				marker="o",
				markersize=markersizes[n],
				linewidth=0,
				color=color,
				zorder=100 - n,
				label=fr"$\chi={chi}$, $\eta^{{x}}={format_value(eta, eta_err)}$"
			)

			ax.axvline(
				xi,
				color=color,
				linestyle="-",
				linewidth=0.5,
				zorder=70 + n
			)

			ax.text(xi * 0.72, 1.4, fr"$\xi^{{\,}}_{{\chi}}\simeq{xi:.0f}$", color=color, fontsize=10, rotation=90)

		ax.set_xlabel(r"$w^{\,}_{0}$")
		ax.set_ylabel(r"$\eta^{x}_{w^{\,}_{0},w^{\,}_{0}+w}$")

		ax.set_xscale("log")

		ax.set_xlim(3, 3000)
		ax.set_ylim(0.2, 2.2)

		ax.set_yticks([1, 2])

		for spine in ax.spines.items():
			spine[1].set_zorder(1000)

		ax.text(
			0.01, 0.03,
			"(b)",
			transform=ax.transAxes,
			ha="left",
			va="bottom",
			fontsize=12)

		ax.legend(
			loc="upper left",
			handlelength=1,
			bbox_to_anchor=(0.03, 1.1)
		).set_zorder(1001)

		plt.draw()

		if args.save:
			save_graph(
				fig,
				os.path.join(GRAPHS_FOLDER, base_folder),
				f"lam={lam}_theta={theta}"
			)
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()