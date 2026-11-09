ENV["MPLBACKEND"] = "Agg"

using PyCall
using PyPlot
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/sty.jl")
include("../utils/py.jl")

PBCs = false

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

save_plot = true

model_name = "A"
simul_name = "PM+DBL/BOTTOM"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

#= θ = 0, λ = 0.1, 2N = 30 =#

fig, ax = plt.subplots(2, 2, figsize=(3.5, 3))
plt.subplots_adjust(wspace=0.2)
plt.subplots_adjust(hspace=0.2)

ps = []
data = load(joinpath(DATA_FOLDER, base_folder, "N=30_chi=128/ANALYZED/N=30_chi=128_lam=0.1_theta=0.0.jld2"))

println(data["energies"])

mX = data["mags"][4, 1, :]
mY = data["mags"][4, 2, :]
mZ = data["mags"][4, 3, :]

p, = ax[1].plot(range(1, length(mX)), mX, markersize=3, lw=0.8, marker="x", color=COLORS[1], zorder=100, label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[2].plot(range(1, length(mY)), mY,  markersize=3, lw=0.8, marker="x", color=COLORS[4], zorder=100, label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[2].plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, marker="x", color=COLORS[7], zorder=100, label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

mX = data["opSPT"][4, 1, :]
mY = data["opSPT"][4, 2, :]
mZ = data["opSPT"][4, 3, :]

println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))

p, = ax[3].plot(range(2, length(mX)+1), mX, markersize=3, lw=0.8, marker=".", color=COLORS[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{X}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[4].plot(range(2, length(mY)+1), mY, markersize=3, lw=0.8, marker=".", color=COLORS[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Y}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[4].plot(range(2, length(mZ)+1), mZ, markersize=3, lw=0.8, marker=".", color=COLORS[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Z}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

ax[1].set(xticklabels=[])
ax[3].set(xticklabels=[])

ax[3].set(yticklabels=[])
ax[4].set(yticklabels=[])

ax[1].minorticks_on()
ax[2].minorticks_on()
ax[3].minorticks_on()
ax[4].minorticks_on()

ax[2].set_xlabel(L"$j$")
ax[4].set_xlabel(L"$j$")

ax[1].set_xticks([0, 10, 20, 30])
ax[2].set_xticks([0, 10, 20, 30])
ax[3].set_xticks([0, 10, 20, 30])
ax[4].set_xticks([0, 10, 20, 30])

ax[1].set_xlim(0, 33)
ax[2].set_xlim(0, 33)
ax[3].set_xlim(0, 33)
ax[4].set_xlim(0, 33)

ax[1].set_ylim(-1.15, 1.15)
ax[2].set_ylim(-0.2, 0.2)
ax[3].set_ylim(-1.15, 1.15)
ax[4].set_ylim(-0.2, 0.2)

ax[1].legend(handles=ps[[1, 2, 3, 4, 5, 6]], loc="lower center", markerscale=1.5, bbox_to_anchor=(1.1, 1.1), borderaxespad=0, ncol=2, labelspacing=0.3, columnspacing=0.5, handletextpad=0.3, handlelength=1.4).set_zorder(200)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, "N=30_chi=128"))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "N=30_chi=128", "lam=0.1_theta=0.pdf"))
	fig.clf()
end

plt.close()

#= θ = 0, λ = 0.1, 2N = 32 =#

fig, ax = plt.subplots(2, 2, figsize=(3.5, 3))
plt.subplots_adjust(wspace=0.2)
plt.subplots_adjust(hspace=0.2)

ps = []
data = load(joinpath(DATA_FOLDER, base_folder, "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.1_theta=0.0.jld2"))

println(data["energies"])

mX = data["mags"][4, 1, :]
mY = data["mags"][4, 2, :]
mZ = data["mags"][4, 3, :]

p, = ax[1].plot(range(1, length(mX)), mX, markersize=3, lw=0.8, marker="x", color=COLORS[1], zorder=100, label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[2].plot(range(1, length(mY)), mY,  markersize=3, lw=0.8, marker="x", color=COLORS[4], zorder=100, label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[2].plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, marker="x", color=COLORS[7], zorder=100, label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

mX = data["opSPT"][4, 1, :]
mY = data["opSPT"][4, 2, :]
mZ = data["opSPT"][4, 3, :]

println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))

p, = ax[3].plot(range(2, length(mX)+1), mX, markersize=3, lw=0.8, marker=".", color=COLORS[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{X}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[4].plot(range(2, length(mY)+1), mY, markersize=3, lw=0.8, marker=".", color=COLORS[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Y}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax[4].plot(range(2, length(mZ)+1), mZ, markersize=3, lw=0.8, marker=".", color=COLORS[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Z}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

ax[1].set(xticklabels=[])
ax[3].set(xticklabels=[])

ax[3].set(yticklabels=[])
ax[4].set(yticklabels=[])

ax[1].minorticks_on()
ax[2].minorticks_on()
ax[3].minorticks_on()
ax[4].minorticks_on()

ax[2].set_xlabel(L"$j$")
ax[4].set_xlabel(L"$j$")

ax[1].set_xticks([0, 10, 20, 30])
ax[2].set_xticks([0, 10, 20, 30])
ax[3].set_xticks([0, 10, 20, 30])
ax[4].set_xticks([0, 10, 20, 30])

ax[1].set_xlim(0, 33)
ax[2].set_xlim(0, 33)
ax[3].set_xlim(0, 33)
ax[4].set_xlim(0, 33)

ax[1].set_ylim(-1.15, 1.15)
ax[2].set_ylim(-0.2, 0.2)
ax[3].set_ylim(-1.15, 1.15)
ax[4].set_ylim(-0.2, 0.2)

ax[1].legend(handles=ps[[1, 2, 3, 4, 5, 6]], loc="lower center", markerscale=1.5, bbox_to_anchor=(1.1, 1.1), borderaxespad=0, ncol=2, labelspacing=0.3, columnspacing=0.5, handletextpad=0.3, handlelength=1.4).set_zorder(200)

plt.draw()

if save_plot
	py"create_folder_if_missing"(joinpath(GRAPHS_FOLDER, base_folder, "N=32_chi=128"))
	fig.savefig(joinpath(GRAPHS_FOLDER, base_folder, "N=32_chi=128", "lam=0.1_theta=0.pdf"))
	fig.clf()
end

plt.close()