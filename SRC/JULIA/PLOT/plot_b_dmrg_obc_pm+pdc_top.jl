ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

model_name = "B"
simul_name = "PM+PDC/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

fig, ax = plt.subplots(figsize=(2.8, 1.5))

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		mX = data["mags"][1, 1, :]
		mY = data["mags"][1, 2, :]
		mZ = data["mags"][1, 3, :]

		mZ_SPT = data["opSPT"][1, 3, :]

		println("mz sta : ", numpy.abs(numpy.mean(mZ .* [(-1)^l for l in range(1, length(mZ))])))
		println("mz sta SPT : ", numpy.abs(numpy.mean(mZ_SPT .* [(-1)^l for l in range(1, length(mZ_SPT))])))

		p, = ax.plot(range(1, length(mX)), mX, marker="o", color=COLORS[7], linewidth=0.3, markersize=3, label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle^{\,}_{\Psi}$"), zorder=103)

		p, = ax.plot(range(1, length(mY)), mY, marker="o", color=COLORS[6], linewidth=0.3, markersize=7, label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle^{\,}_{\Psi}$"), zorder=102)
		
		p, = ax.plot(range(1, length(mZ)), mZ, marker="o", color=COLORS[4], linewidth=0.3, markersize=5, label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle^{\,}_{\Psi}$"), zorder=101)

		ax.minorticks_on()

		ax.set_xlabel(L"$j$")

		ax.set_yticks([-1, 0, 1])

		ax.set_xlim(0, 65)
		ax.set_ylim(-1, 1)
		
		ax.legend(
			loc="lower center",
			ncol=3,
			bbox_to_anchor=(0.5, 0.93)).set_zorder(1000)	

		plt.draw()

		if save_plot
			py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, folder))
			fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, folder, "spins.pdf"))
		end

		plt.close()
	end
end

