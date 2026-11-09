using ITensors, ITensorMPS
using OrderedCollections
using PyCall
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../dmrg/build_hamiltonian.jl")
include("../dmrg/end_sweep.jl")

function runDMRG(model, params, tweaks)

	os, sites = build_hamiltonian(model, params)
	
	H = MPO(os, sites)
	psi = randomMPS(ComplexF64, sites; linkdims=tweaks["link_dim_init"])

	if tweaks["num_excited"] > 1
		println("----- get state\n \t | k : 1\n")
		tweaks["save_every"] = false
	end

	energies = []
	obs = Observer(H, tweaks["data_filename"], tweaks["save_every"], tweaks["energy_tol"])

	energy, psi = dmrg(H, psi; nsweeps=tweaks["max_sweeps"], cutoff=tweaks["cut_off"], maxdim=params["chi"], observer=obs, outputlevel=tweaks["output_level"], eigsolve_krylovdim=tweaks["krylov_dim"], eigsolve_maxiter=tweaks["max_iter_krylov"], noise=tweaks["noise"])

	push!(energies, energy)
	psis = [psi]

	for i=1:tweaks["num_excited"]-1

		println("\n----- get state\n\t | k : ", i+1, "\n")

		psi = randomMPS(ComplexF64, sites; linkdims=tweaks["link_dim_init"])
		obs = Observer(H, tweaks["data_filename"], tweaks["save_every"], tweaks["energy_tol"])

		energy, psi = dmrg(H, psis, psi; nsweeps=tweaks["max_sweeps"], cutoff=tweaks["cut_off"], maxdim=params["chi"], observer=obs, weight=10, outputlevel=tweaks["output_level"], eigsolve_krylovdim=tweaks["krylov_dim"], eigsolve_maxiter=tweaks["max_iter_krylov"], noise=tweaks["noise"])

		push!(energies, energy)
		push!(psis, psi)

	end
		 
	return energies, psis
end

model_name = "A"
simul_name = "PA+PDC/BOTTOM"

nLam = 1
λ = range(pi/4-pi/40, pi/4-pi/40, nLam)
nTheta = 1
θ = range(pi/4-pi/40, pi/4-pi/40, nTheta)

nN = 1
N = range(1024, 1024, nN)
nChi = 1
χ = range(512, 512, nChi)

PBCs = false

k = 1
save_every = true

max_sweeps = 30
cut_off = [1E-9]
noise = [0]
output_level = 1
energy_tol = 1E-9
krylov_dim = 10
max_iter_krylov = 2
link_dim_init = 16

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

model = Dict("name" => model_name, "PBCs" => PBCs)
tweaks = Dict("num_excited" => k, "max_sweeps" => max_sweeps, "cut_off" => cut_off, "output_level" => output_level, "energy_tol" => energy_tol, "krylov_dim" => krylov_dim, "max_iter_krylov" => max_iter_krylov, "link_dim_init" => link_dim_init, "noise" => noise, "save_every" => save_every)

for i in 1:nLam
	for j in 1:nTheta
		for n in 1:nN
			for x in 1:nChi
				
				params = OrderedDict("N" => trunc(Int, N[n]), "chi" => trunc(Int, χ[x]), "lam" => round(λ[i]; digits=15), "theta" => round(θ[j]; digits=15))

				println("\n\t\t start DMRG • $(model_name) / $(boundary_name) \n")
				println("\t | N : ", params["N"])
				println("\t | χ : ", params["chi"])
				println("\t | λ : ", params["lam"])
				println("\t | θ : ", params["theta"], "\n")

				folder = "N=$(params["N"])_chi=$(params["chi"])"
				data_folder = joinpath(DATA_FOLDER, base_folder, folder)
				data_filename = joinpath(data_folder, "$(py"dict_to_str"(params)).jld2")

				tweaks["data_filename"] = data_filename
				
				if !isfile(data_filename)

					energies, psis = runDMRG(model, params, tweaks)

					dict = Dict("psis" => psis, "energies" => energies)
					save(data_filename, dict)

					println("\n---------------------------------------------------------- DONE")
				else
					println("---------------------------------------------------------- ALREADY DONE")
				end

			end
		end
	end
end
