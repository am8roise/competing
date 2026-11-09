ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "A"
simul_name = "LM+DBL/RIGHT"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
folders = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	push!(Ns, N)
	push!(folders, folder)
end

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[2]], N=2*length(Ns))
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["-", "--"], marker=[".", "."], markersize=[6, 6], color=cmap(numpy.linspace(0, 1, 2*length(Ns)))))

fig, ax = plt.subplots(figsize=(8, 3))

sort = numpy.argsort(Ns) .+ 1
folders = folders[sort]

for folder in folders

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	mags = numpy.ones((length(lam), length(theta), 6))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		mX = data["mags"][1, 1, :]
		mY = data["mags"][1, 2, :]
		mZ = data["mags"][1, 3, :]

		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l in 1:length(mX)])
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l in 1:length(mY)])
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
		mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l in 1:length(mZ)])
	end

	for l in 1:length(lam)

		p1, = ax.plot(theta .- pi/2, numpy.abs(mags[l, :, 4]), markeredgewidth=1, linewidth=1, zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{uni}}|,\ 2N=%$(N)$"))
		p1.set_marker(matplotlib.markers.MarkerStyle(p1.get_marker(), capstyle="round"))

		p2, = ax.plot(theta .- pi/2, numpy.abs(mags[l, :, 5]), markeredgewidth=1, linewidth=1, zorder=105, label=latexstring(L"$|m^{z}_{\mathrm{sta}}|,\ 2N=%$(N)$"))
		p2.set_marker(matplotlib.markers.MarkerStyle(p2.get_marker(), capstyle="round"))
	end
end

ax.minorticks_on()

ax.legend(loc="center left", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.02, 0.48)).set_zorder(110)

plt.xlabel(latexstring(L"$\theta$"))

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "op.pdf"))
	ax.cla()
end

plt.close()