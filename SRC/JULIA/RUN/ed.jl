using MPSKit, MPSKitModels, TensorKit, LinearAlgebra, KrylovKit
using OrderedCollections
using JLD2

include("../utils/io.jl")
include("../utils/path.jl")
include("../ed/build_hamiltonian.jl")

model_name = "B"
simul_name = "PE+TEL/TOP"

nLam = 1
λ = range(3*pi/4, 3*pi/4, nLam)
nTheta = 1
θ = range(0, 0, nTheta)

nN = 9
N = range(6, 22, nN)

PBCs = false

k = 10

max_iter_krylov = 100
krylov_dim = k*10

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "ED"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

model = Dict("name" => model_name, "PBCs" => PBCs)

errors = []

for i in 1:nLam
	for j in 1:nTheta
		for n in 1:nN

			params = OrderedDict("N" => trunc(Int, N[n]), "lam" => round(λ[i]; digits=15), "theta" => round(θ[j]; digits=15))

			folder = "N=$(params["N"])"
			data_folder = joinpath(DATA_FOLDER, base_folder, folder)
			data_filename = joinpath(data_folder, "$(py"dict_to_str"(params)).jld2")

			println("\n\t\t start ED • $(model_name) / $(boundary_name) \n")
			println("\t | N : ", params["N"])
			println("\t | λ : ", params["lam"])
			println("\t | θ : ", params["theta"], "\n")
			
			if !isfile(data_filename)
				
				H = build(model, params)

				try
					energies, ψ = exact_diagonalization(H; num=k, which=:SR, alg=Lanczos(; maxiter=max_iter_krylov, krylovdim=krylov_dim));

					dict = Dict("energies" => energies)
					save(data_filename, dict)
				catch
					push!(errors, params)
				end

				println("---------------------------------------------------------- DONE")

			else
				println("---------------------------------------------------------- ALREADY DONE")
			end

		end
	end
end