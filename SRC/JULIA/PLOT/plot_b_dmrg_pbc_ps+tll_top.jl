ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")
include("../utils/math.jl")

model_name = "B"
simul_name = "PS+TLL/TOP"

PBCs = true

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

plot_entropies = false
plot_N = false
plot_all = true

save_plot = true

Ns = []
chis = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	push!(Ns, N)
	push!(chis, chi)
	push!(folders, folder)
end

chis_Ns = numpy.asarray([chis, Ns])
sort = numpy.lexsort(chis_Ns, axis=0) .+ 1

Ns = Ns[sort]
chis = chis[sort]
folders = folders[sort]

Ns_unique = numpy.unique(Ns)

fig, ax = plt.subplots(figsize=(5, 4))
fig_all, ax_all = plt.subplots(figsize=(8.5, 4))

cs_Ns = []

for n in 1:length(Ns_unique)

	ind = numpy.where(Ns .== Ns_unique[n])[1] .+ 1
	println(ind)

	chis_N = chis[ind]
	folders_N = folders[ind]

	fig_chi, ax_chi = plt.subplots(figsize=(3, 3))

	cs_Xs = []
	cs_Xs_err = []

	for x in 1:length(chis_N)
		
		folder = folders_N[x]

		data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

		vals, data_filenames = py"load_file"(data_folder, expression)

		global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
		global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

		global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
		global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

		cs = numpy.zeros((length(lam), length(theta), 2))

		for i in 1:length(data_filenames)

			data = 0
			data = load(joinpath(data_folder, data_filenames[i]))
			
			k = length(data["energies"])

			win = N-6

			println(data_filenames[i])

			if plot_entropies | plot_N | plot_all
				
				Ss = data["entropies"]

				sites = range(1, size(Ss, 2))

				sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
				S = Ss[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

				py"""
				def f(l, c, const):
					import numpy

					N = int(l[len(l)//2]*2)

					return c/3 * numpy.log(N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
				"""

				popt, pcov = scipy.optimize.curve_fit(py"f", collect(sites), S)
				perr = numpy.sqrt(numpy.diag(pcov))

				cs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= popt[1]
				cs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perr[1]

				label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=5)) \pm %$(round(perr[1]; digits=5)))}{6} \ln\left[\frac{2N}{\pi}\sin\frac{\pi \ell}{N} \right] + \mathrm{ const}$")

				ax.scatter(sites, S, marker="x", color=COLORS[2], zorder=100)
				ax.plot(sites, py"f"(collect(sites), popt...), markersize=0, linestyle=":", color=COLORS[5], label=label, zorder=100)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2, alpha=0.33)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0).set_zorder(101)

				plt.xlabel(latexstring(L"$\ell$"))
				plt.ylabel(latexstring(L"$S$"))

				plt.draw()

				if save_plot & plot_entropies
					py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder, "entropies"))
					fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "entropies", "$(chop(data_filenames[i], tail=5)).pdf"))
					ax.cla()
				end

				ax.cla()
			end
		end

		push!(cs_Xs, cs[1, 1, 1])
		push!(cs_Xs_err, cs[1, 1, 2])

	end

	if n == 1
		p, = ax_all.plot([1/Ns_unique[n]^1 for l=1:length(cs_Xs)], cs_Xs, linewidth=0, marker=".", markersize=7.5, markeredgewidth=1.5, color=COLORS[6], zorder=100, label=latexstring(L"$\mathsf{c}^{\,}_{2N,\chi}$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	else
		p, = ax_all.plot([1/Ns_unique[n]^1 for l=1:length(cs_Xs)], cs_Xs, linewidth=0, marker=".", markersize=7.5, markeredgewidth=1.5, color=COLORS[6], zorder=100)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	end


	py"""
	def g(N, a, b):
		import numpy 

		return a*N + b
	"""

	cs_Xs = cs_Xs[end-3:end]
	chis_N = chis_N[end-3:end]

	popt, pcov = scipy.optimize.curve_fit(py"g", 1 ./chis_N, cs_Xs)
	perr = numpy.sqrt(numpy.diag(pcov))

	push!(cs_Ns, cs_Xs[end])

	o, = ax_chi.plot(1 ./chis_N, py"g"(1 ./chis_N, popt...), linestyle="--", marker="none", color=COLORS[5], linewidth=1, zorder=90, markersize=0, label=latexstring(L"$\mathsf{c}(\chi \to \infty) = (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$"))

	ax_chi.scatter(1 ./chis_N, cs_Xs, marker="x", color=COLORS[2], zorder=100)

	ax_chi.minorticks_on()
	ax_chi.grid(which="minor", linewidth=0.2, alpha=0.33)
	ax_chi.grid(which="major", linewidth=0.6)
	ax_chi.legend(loc="lower center", ncol=1, columnspacing=1, labelspacing=0.6, bbox_to_anchor=(0.5, 1.01), borderaxespad=0).set_zorder(110)

	ax_chi.set_xlabel(latexstring(L"$\chi$"))
	ax_chi.set_ylabel(latexstring(L"$\mathsf{c}$"))

	plt.draw()

	if save_plot & plot_N
		py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
		fig_chi.savefig(joinpath(GRAPHS_FOLDER, base_folder, "2N=$(Ns_unique[n]).pdf"))
		fig_chi.clf()
	end

	plt.close()
end

py"""
def g(N, a, b):
	import numpy 

	return a*N + b
"""

Ns_unique = Ns_unique.^1
p, = ax_all.plot(1 ./Ns_unique, cs_Ns, marker="x", markersize=10, markeredgewidth=2, linewidth=0, color=COLORS[4], zorder=100, label=latexstring(L"$\displaystyle{\lim_{\chi\to\infty}}\mathsf{c}^{\,}_{2N,\chi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ind = numpy.where(Ns_unique .% 4 .== 0)[1] .+ 1

cs_Ns = cs_Ns[ind][2:end]
Ns_unique = Ns_unique[ind][2:end]

popt, pcov = scipy.optimize.curve_fit(py"g", 1 ./Ns_unique, cs_Ns)
perr = numpy.sqrt(numpy.diag(pcov))

r2_ = py"r2"(cs_Ns, py"g"(1 ./Ns_unique, popt...))

o, = ax_all.plot(1 ./Ns_unique, py"g"(1 ./Ns_unique, popt...), linestyle="--", marker="none", color=COLORS[1], linewidth=1.5, zorder=90, markersize=0, label=latexstring(L"$\displaystyle{\lim_{2N\to\infty}\lim_{\chi\to\infty}}\mathsf{c}^{\,}_{2N,\chi} = (%$(round(popt[2]; digits=2)) \pm %$(round(perr[2]; digits=2))),\ R^{2} = %$(round(r2_; digits=3))$"))

ax_all.minorticks_on()

ax_all.legend(loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.8, handlelength=1.6).set_zorder(110)

ax_all.set_xlabel(latexstring(L"$1/(2N)$"))
ax_all.set_ylabel(latexstring(L"$\mathsf{c}$"))

plt.draw()

if save_plot & plot_all
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig_all.savefig(joinpath(GRAPHS_FOLDER, base_folder, "all.pdf"))
	fig_all.clf()
end

plt.close()