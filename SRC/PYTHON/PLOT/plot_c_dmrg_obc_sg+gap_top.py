import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import seaborn as sns

from tenet.utils.path import DATA_FOLDER, GRAPHS_FOLDER, COMPUTE_FOLDER
from tenet.utils.io import folder_names, load_file, save_graph
from tenet.utils.sty import COLORS

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save the graph(s).")
parser.add_argument("--show", action="store_true", help="Show the graph(s).")

args = parser.parse_args()

def main():
	"""
	Executes the plotting process.
	"""
	model_name = "C"
	simul_name = "SG+GAP/TOP"

	PBCs = False

	boundary_name = "PBC" if PBCs else "OBC"
	method_name = "DMRG"

	base_folder = os.path.join(model_name, method_name, boundary_name, simul_name)

	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
					
	for folder in folder_names(os.path.join(DATA_FOLDER, base_folder)):
			
		data_folder = os.path.join(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = load_file(data_folder, expression)

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))]).item()

		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))
		
		evs_2 = np.zeros((len(lam), len(theta), 3))
		evs_4 = np.zeros((len(lam), len(theta), 3))
		evs_6 = np.zeros((len(lam), len(theta), 3))
		evs_8 = np.zeros((len(lam), len(theta), 3))
		evs_10 = np.zeros((len(lam), len(theta), 3))

		for i in range(len(data_filenames)):

			with open(os.path.join(data_folder, data_filenames[i]), 'rb') as file:
				data = pickle.load(file)

			evs_2[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.mean(data["evs_2"][0, :])
			evs_2[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.mean(data["evs_2"][1, :])
			evs_2[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.mean(data["evs_2"][2, :])

			evs_4[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 0] = np.mean(data["evs_4"][0, :])
			evs_4[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 1] = np.mean(data["evs_4"][1, :])
			evs_4[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.mean(data["evs_4"][2, :])

			evs_6[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.mean(data["evs_6"][2, :])
			evs_8[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.mean(data["evs_8"][2, :])
			evs_10[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), 2] = np.mean(data["evs_10"][2, :])

		# ------- ev2x

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_2[::-1, :, 0], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{2,x}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"x2")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
		# ------- ev2y

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_2[::-1, :, 1], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{2,y}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"y2")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
		# ------- ev2z

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_2[::-1, :, 2], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{2,z}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"z2")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
		# ------- ev4x

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_4[::-1, :, 0], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{4,x}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"x4")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
		# ------- ev4y

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_4[::-1, :, 1], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{4,y}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"y4")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
		# ------- ev4z

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_4[::-1, :, 2], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{4,z}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"z4")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()
		
		# ------- ev6z

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_6[::-1, :, 2], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{6,z}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"z6")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()	

		# ------- ev8z

		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_8[::-1, :, 2], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{8,z}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"z8")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

		# ------- ev10z
		
		fig, ax = plt.subplots(figsize=(5, 4))

		cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[6],COLORS[8],COLORS[0],COLORS[5],COLORS[3]], N=256)

		data = pd.DataFrame(evs_10[::-1, :, 2], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

		axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, rasterized=True)

		cb = fig.colorbar(ax.collections[0])
		cb.ax.tick_params(labelsize=18)

		plt.xlabel(r"$\theta$")
		plt.ylabel(r"$\lambda$")

		plt.xticks([0*len(theta), 0.5*len(theta), 1*len(theta)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
		plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])			
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		for _, spine in axs.spines.items(): 
		    spine.set_visible(True) 

		plt.title(r"$\overline{G^{10,z}}$", fontsize=18, pad=10)
				
		plt.draw()

		if args.save:
			save_graph(fig, os.path.join(GRAPHS_FOLDER, base_folder, folder), f"z10")
			fig.clf()

		if args.show:
			plt.show()

		plt.close()

if __name__ == "__main__":
	main()

