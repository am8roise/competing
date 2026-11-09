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
simul_name = "LM+TLL/TOP"

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

save_plot = true

Ns = []
binder_Ns = []

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))

	data_folder = joinpath(DATA_FOLDER, base_folder, folder, COMPUTE_FOLDER)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])
	global chi = Int(numpy.unique([vals[i]["chi"] for i in 1:length(vals)])[])

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i in 1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i in 1:length(vals)]))
	
	binder = numpy.ones((length(lam), length(theta), 2, 3, 2))

	for i in 1:length(data_filenames)

		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		binder[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :, :, :] = data["binder"][1, :, :, :]
	end

	push!(Ns, N)
	push!(binder_Ns, binder)
end

ind = numpy.argsort(Ns) .+ 1

Ns = numpy.array(Ns)[ind]
binder_Ns = numpy.array(binder_Ns)[ind, :, :, :, :, :]

len = length(Ns)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[COLORS[1],COLORS[3],COLORS[4],COLORS[6],COLORS[7],COLORS[8]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", "--", ":"], outer=len)[1:len], marker=repeat(["o"], outer=len)[1:len], markersize=repeat([3], outer=len)[1:len], color=cmap(numpy.linspace(0, 1, len)), markerfacecolor=cmap(numpy.linspace(0, 1, len))))

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)

for n in 1:length(Ns)

	p, = ax[1].plot(lam, binder_Ns[n, :, 1, 2, 1, 2], linewidth=0.6, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

end

for n in 1:length(Ns)

	p, = ax[2].plot(lam, binder_Ns[n, :, 1, 1, 1, 1], linewidth=0.6, label=latexstring(L"$2N=%$(Ns[n])$"), zorder=101)
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
end

ax[1].minorticks_on()
ax[2].minorticks_on()

ax[1].legend(loc="lower center", ncol=3, markerscale=2, labelspacing=0.4, columnspacing=2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.25, 1.02)).set_zorder(110)

ax[1].set_xlabel(latexstring(L"$\lambda$"))
ax[2].set_xlabel(latexstring(L"$\lambda$"))
ax[1].set_ylabel(latexstring(L"$U\left[m^{x}_{\mathrm{sta}}\right]$"))
ax[2].set_ylabel(latexstring(L"$U\left[m^{\mathrm{SPT},x}_{\mathrm{sta}}\right]$"))

ax[1].axvline(3*pi/4, color="k", linestyle="--", linewidth=0.8)
ax[2].axvline(3*pi/4, color="k", linestyle="--", linewidth=0.8)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "op.pdf"))
end

plt.close()

# ind_0 = numpy.where(Ns .% 4 .== 0)[1] .+ 1
# ind_2 = numpy.where(Ns .% 4 .== 2)[1] .+ 1

# binder_Ns_0 = binder_Ns[ind_0,:, 1, 1, 1, 2]
# binder_Ns_2 = binder_Ns[ind_2,:, 1, 1, 1, 2]

# cross_0 = []
# for n in zip(1:length(Ns[ind_0])-1, 2:length(Ns[ind_0]))

# 	ind = numpy.where((lam .> 0.5) .& (lam .< 1))[1] .+ 1

# 	d = binder_Ns_0[n[2], ind] .- binder_Ns_0[n[1], ind]
# 	λ = lam[ind]

# 	for i in 1:length(λ)-1
# 		d1, d2 = d[i], d[i+1]

# 		if d1 * d2 < 0 || d2 == 0
# 			x = d1 / (d1 - d2)
# 			global cross_ = λ[i] + x * (λ[i+1] - λ[i])
# 			break
# 		end
# 	end

# 	# try
# 	# 	global cross_ = lam[ind][py"find_nearest"(binder_Ns_0[n[1], ind], binder_Ns_0[n[2], ind])]
# 	# catch
# 	# 	global cross_ = 0
# 	# end
# 	push!(cross_0, cross_)
# end

# cross_2 = []
# for n in zip(1:length(Ns[ind_2])-1, 2:length(Ns[ind_2]))

# 	ind = numpy.where((lam .> 0.5) .& (lam .< 1))[1] .+ 1

# 	try
# 		global cross_ = lam[ind][py"find_nearest"(binder_Ns_2[n[1], ind], binder_Ns_2[n[2], ind])]
# 	catch
# 		global cross_ = 0
# 	end
# 	push!(cross_2, cross_) 
# end

# print(cross_0, cross_2)

# fig, ax = plt.subplots(figsize=(4, 3))

# p, = ax.plot(1 ./ Ns[ind_0][1:end-2] .^ 2, cross_0[1:end-1], linewidth=0, markeredgewidth=0.2, markeredgecolor="0.")
# p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

# py"""
# def f(N, a, b):
# 	return a * 1 / N + b
# """

# popt, pcov = scipy.optimize.curve_fit(py"f", collect(Ns[ind_0][1:end-2] .^ 2), cross_0[1:end-1])
# perr = numpy.sqrt(numpy.diag(pcov))

# println(popt, perr)

# p, = ax.plot(1 ./ Ns[ind_0][1:end-2] .^ 2, py"f"(Ns[ind_0][1:end-2] .^ 2, popt...), linewidth=0.6, marker="none")
# p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

# plt.draw()

# if save_plot
# 	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder))
# 	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, " .pdf"))
# end

# plt.close()