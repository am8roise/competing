ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "LE+TLT/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

nE = 6

Ns = []
diffs_Ns = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	diffs = numpy.zeros((length(lam), length(theta), nE-1))

	for i=1:length(data_filenames)

		N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))
		
		E = numpy.sort(data["energies"])
		
		diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = [E[e+1] - E[e] for e in 1:nE-1]
	end

	push!(Ns, N)
	push!(diffs_Ns, diffs)
end

ind = numpy.argsort(Ns).+1

Ns = numpy.array(Ns)[ind]
diffs_Ns = numpy.array(diffs_Ns)[ind, :, :, :]

fig, ax = plt.subplots(1, 3, figsize=(5, 3))
plt.subplots_adjust(wspace=0.1)

for (i, l) in enumerate(lam)
	println("lam[$(i)] = ", l)
end
 
for (s, l) in enumerate([7, 2, 1])

	global ps = []
	global os = []

	p1, = ax[s].plot(1 ./ Ns, diffs_Ns[:, l, 1, 1], marker="o", markerfacecolor=COLORS[5], lw=0, markersize=12, label=latexstring(L"$E^{\,}_{1}-E^{\,}_{0}$"), zorder=101)
	push!(ps, p1)

	p2, = ax[s].plot(1 ./ Ns, diffs_Ns[:, l, 1, 2], marker="o", markerfacecolor=COLORS[3], lw=0, markersize=12, label=latexstring(L"$E^{\,}_{2}-E^{\,}_{1}$"), zorder=102)
	push!(ps, p2)

	p3, = ax[s].plot(1 ./ Ns, diffs_Ns[:, l, 1, 3], marker="o", markerfacecolor=COLORS[1], lw=0, markersize=8, label=latexstring(L"$E^{\,}_{3}-E^{\,}_{2}$"), zorder=103)
	push!(ps, p3)

	p4, = ax[s].plot(1 ./ Ns, diffs_Ns[:, l, 1, 4], marker="o", markerfacecolor=COLORS[6], lw=0, markersize=4, label=latexstring(L"$E^{\,}_{4}-E^{\,}_{3}$"), zorder=104)
	push!(ps, p4)

	p5, = ax[s].plot(1 ./ Ns, diffs_Ns[:, l, 1, 5], marker="o", markerfacecolor=COLORS[4], lw=0, markersize=4, label=latexstring(L"$E^{\,}_{5}-E^{\,}_{4}$"), zorder=105)
	push!(ps, p5)

	py"""
	def f(N, a, b):
		import numpy 

		return a/N + b
	"""

	x = numpy.linspace(0, 1 / numpy.min(Ns), 100)

	popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffs_Ns[:, l, 1, 1])
	perr = numpy.sqrt(numpy.diag(pcov))

	ax[s].plot(x, py"f"(1 ./ x, popt...), linestyle="-", marker="none", color=p1.get_markerfacecolor(), linewidth=2, zorder=91)

	popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffs_Ns[:, l, 1, 2])
	perr = numpy.sqrt(numpy.diag(pcov))

	ax[s].plot(x, py"f"(1 ./ x, popt...), linestyle="-", marker="none", color=p2.get_markerfacecolor(), linewidth=2, zorder=92)

	popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffs_Ns[:, l, 1, 3])
	perr = numpy.sqrt(numpy.diag(pcov))

	ax[s].plot(x, py"f"(1 ./ x, popt...), linestyle=(0.75, (2, 2)), marker="none", color=p3.get_markerfacecolor(), linewidth=2, zorder=93)

	popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffs_Ns[:, l, 1, 4])
	perr = numpy.sqrt(numpy.diag(pcov))

	ax[s].plot(x, py"f"(1 ./ x, popt...), linestyle=(0, (0, 2)), marker="none", color=p4.get_markerfacecolor(), linewidth=2, zorder=94)

	popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffs_Ns[:, l, 1, 5])
	perr = numpy.sqrt(numpy.diag(pcov))

	ax[s].plot(x, py"f"(1 ./ x, popt...), linestyle=(0, (0, 2)), marker="none", color=p5.get_markerfacecolor(), linewidth=2, zorder=95)

	ax[s].set_xlabel(latexstring(L"$(2N)^{-1}$"))

	ax[s].dataLim.y0 = 0
	ax[s].dataLim.y1 = numpy.max(diffs_Ns[:, :, 1, :])
	ax[s].autoscale_view()

	ax[s].set_xlim(0, 1 / numpy.min(Ns) + 0.005)

	ax[s].set_xticks([0, 1 / Ns[5], 1 / Ns[3], 1 / Ns[1]])
	ax[s].set_xticklabels([L"0", L"\frac{1}{%$(Ns[5])}", L"\frac{1}{%$(Ns[3])}", L"\frac{1}{%$(Ns[1])}"])

	if s == 1
		ax[s].set_title(latexstring(L"$\lambda=\pi$"), pad=5)
	else
		ax[s].set_title(latexstring(L"$\lambda=\pi-%$(round(pi-lam[l]; digits=2))$"), pad=5)
	end

	for spine in ax[s].spines.values()
		spine.set_zorder(1000)
	end

	if s > 1
		ax[s].set_yticks([])
	end
end

ax[1].set_yticks([0, 0.1, 0.2])

ax[1].text(
	0.03, 0.97,
	"(a)",
	transform=ax[1].transAxes,
	ha="left",
	va="top",
	fontsize=12)
ax[2].text(
	0.03, 0.97,
	"(b)",
	transform=ax[2].transAxes,
	ha="left",
	va="top",
	fontsize=12)
ax[3].text(
	0.03, 0.97,
	"(c)",
	transform=ax[3].transAxes,
	ha="left",
	va="top",
	fontsize=12)

ax[3].legend(
	handles=ps,
	loc="center left",
	ncol=1,
	borderpad=0.5,
	labelspacing=0.8,
	handletextpad=0.5,
	bbox_to_anchor=(0.95, 0.5)).set_zorder(1000)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "gaps.pdf"))
	fig.clf()
end

plt.close()
