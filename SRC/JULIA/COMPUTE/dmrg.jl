using ITensors, ITensorMPS
using ITensorCorrelators
using PyCall
using FileIO

include("../utils/io.jl")
include("../utils/path.jl")
include("../utils/py.jl")
include("../utils/helpers.jl")

PBCs = true
force = true

compute_entropies_entanglement = false
compute_mags = true
compute_corrs = false
compute_corrsSPT = false
compute_corrsCOS = false
compute_evs_2 = false
compute_evs_4 = false
compute_evs_6 = false
compute_evs_8 = false
compute_evs_10 = false
compute_opSPT = false
compute_boundary = false
compute_binder = false

model_name = "B"
simul_name = "PS+TLL/BOTTOM"

boundary_name = PBCs ? "PBC" : "OBC"
method_name = "DMRG"

base_folder = joinpath(model_name, method_name, boundary_name, simul_name)

expression = raw"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

for folder in py"folder_names"(joinpath(DATA_FOLDER, base_folder))
	
	data_folder = joinpath(DATA_FOLDER, base_folder, folder)

	vals, data_filenames = py"load_file"(data_folder, expression)

	global N = Int(numpy.unique([vals[i]["N"] for i in 1:length(vals)])[])

	r = Int(N / 2)

	for i in 1:length(data_filenames)
		
		data = 0
		data = load(joinpath(data_folder, data_filenames[i]))

		println(data_filenames[i])

		k = length(data["energies"])

		quantities = Dict{Any, Any}("energies" => data["energies"])

		if isfile(joinpath(data_folder, COMPUTE_FOLDER, data_filenames[i]))

			quantities = load(joinpath(data_folder, COMPUTE_FOLDER, data_filenames[i]))
			println("\t • ALREADY COMPUTED : ", keys(quantities))
		end

		if force | !isfile(joinpath(data_folder, COMPUTE_FOLDER, data_filenames[i]))

			if compute_binder
				println("\t | binder...")

				try
					global binder = quantities["binder"]
				catch
					global binder = zeros((k, 2, 3, 2))
				end

				for j in 1:k
					# binder[j, 1, 1, 1] = binder_cumulant(data["psis"][j], ["Sx"]; SPT=false, staggered=false)
					binder[j, 1, 1, 2] = binder_cumulant(data["psis"][j], ["Sx"]; SPT=false, staggered=true)

					# binder[j, 1, 2, 1] = binder_cumulant(data["psis"][j], ["Sy"]; SPT=false, staggered=false)
					# binder[j, 1, 2, 2] = binder_cumulant(data["psis"][j], ["Sy"]; SPT=false, staggered=true)

					# binder[j, 1, 3, 1] = binder_cumulant(data["psis"][j], ["Sz"]; SPT=false, staggered=false)
					# binder[j, 1, 3, 2] = binder_cumulant(data["psis"][j], ["Sz"]; SPT=false, staggered=true)

					# binder[j, 2, 1, 1] = binder_cumulant(data["psis"][j], ["Sz", "Sx", "Sz"]; SPT=true, staggered=false)
					binder[j, 2, 1, 2] = binder_cumulant(data["psis"][j], ["Sz", "Sx", "Sz"]; SPT=true, staggered=true)

					# binder[j, 2, 2, 1] = binder_cumulant(data["psis"][j], ["Sz", "Sy", "Sz"]; SPT=true, staggered=false)
					# binder[j, 2, 2, 2] = binder_cumulant(data["psis"][j], ["Sz", "Sy", "Sz"]; SPT=true, staggered=true)

					# binder[j, 2, 3, 1] = binder_cumulant(data["psis"][j], ["Sz", "Sz", "Sz"]; SPT=true, staggered=false)
					# binder[j, 2, 3, 2] = binder_cumulant(data["psis"][j], ["Sz", "Sz", "Sz"]; SPT=true, staggered=true)
				end
				quantities["binder"] = binder

			end

			if compute_entropies_entanglement
				println("\t | entropies/entanglement...")

				entropies = zeros((k, N-1))
				entanglement = Array{Vector{Float64}}(undef, k, N-1)
				for j in 1:k
					psi = data["psis"][j]
					for b in 1:N-1
						psi = ITensorMPS.orthogonalize!(psi, b)
						U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
						SvN = 0.0
						diag = []
						for l in 1:dim(S, 1)
							push!(diag, S[l,l])
							p = S[l,l]^2
							SvN -= p * log(p)
						end
						entropies[j, b] = SvN
						entanglement[j, b] = diag
					end
				end
				quantities["entropies"] = entropies
				quantities["entanglement"] = entanglement
			end

			if compute_mags
				println("\t | mags...")

				mags = zeros((k, 3, N))
				for j in 1:k
					mags[j, 1, :] = 2 * expect(data["psis"][j], "Sx")
					mags[j, 2, :] = 2 * expect(data["psis"][j], "Sy")
					mags[j, 3, :] = 2 * expect(data["psis"][j], "Sz")
				end
				quantities["mags"] = mags
			end

			if compute_corrs
				println("\t | corrs...")

				corrs = zeros((k, 3, N, N))
				for j in 1:k
					corrs[j, 1, :, :] .= real(4 * correlation_matrix(data["psis"][j], "Sx", "Sx"))
					corrs[j, 2, :, :] .= real(4 * correlation_matrix(data["psis"][j], "Sy", "Sy"))
					corrs[j, 3, :, :] .= real(4 * correlation_matrix(data["psis"][j], "Sz", "Sz"))
				end
				quantities["corrs"] = corrs
			end

			if compute_corrsSPT
				println("\t | corrsSPT...")

				corrsSPT = zeros((k, 3, r))	
				for j in 1:k
					X = correlator(data["psis"][j], ("Sz", "Sx", "Sz", "Sz", "Sx", "Sz"), [(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1)) for l in 1:r])
					Y = correlator(data["psis"][j], ("Sz", "Sy", "Sz", "Sz", "Sy", "Sz"), [(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1)) for l in 1:r])
					Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1)) for l in 1:r])

					corrsSPT[j, 1, :] = [64 * real(X[(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))]) for l in 1:r]
					corrsSPT[j, 2, :] = [64 * real(Y[(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))]) for l in 1:r]
					corrsSPT[j, 3, :] = [64 * real(Z[(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))]) for l in 1:r]
				end
				quantities["corrsSPT"] = corrsSPT
			end

			if compute_corrsCOS
				println("\t | corrsCOS...")

				corrsCOS22 = zeros((k, 4, r))
				corrsCOS24 = zeros((k, 4, r))
				corrsCOS42 = zeros((k, 4, r))
				corrsCOS44 = zeros((k, 4, r))

				for j in 1:k

					j0 = Int(N/2-r/2)

					sites22 = [
						(
							j0,
							j0+1,
							j0+l,
							j0+l+1
						)
						for l in 1:r
					]

					sites24 = [
						(
							j0,
							j0+1,
							j0+l-1,
							j0+l,
							j0+l+1,
							j0+l+2
						)
						for l in 1:r
					]

					sites42 = [
						(
							j0-1,
							j0,
							j0+1,
							j0+2,
							j0+l,
							j0+l+1
						)
						for l in 1:r
					]

					sites44 = [
						(
							j0-1,
							j0,
							j0+1,
							j0+2,
							j0+l-1,
							j0+l,
							j0+l+1,
							j0+l+2
						)
						for l in 1:r
					]

					XX22 = correlator(
						data["psis"][j],
						("Sx", "Sx", "Sx", "Sx"),
						sites22
					)

					XY22 = correlator(
						data["psis"][j],
						("Sx", "Sx", "Sy", "Sy"),
						sites22
					)

					YX22 = correlator(
						data["psis"][j],
						("Sy", "Sy", "Sx", "Sx"),
						sites22
					)

					YY22 = correlator(
						data["psis"][j],
						("Sy", "Sy", "Sy", "Sy"),
						sites22
					)

					XX24 = correlator(
						data["psis"][j],
						(
							"Sx", "Sx",
							"Sz", "Sx", "Sx", "Sz"
						),
						sites24
					)

					XY24 = correlator(
						data["psis"][j],
						(
							"Sx", "Sx",
							"Sz", "Sy", "Sy", "Sz"
						),
						sites24
					)

					YX24 = correlator(
						data["psis"][j],
						(
							"Sy", "Sy",
							"Sz", "Sx", "Sx", "Sz"
						),
						sites24
					)

					YY24 = correlator(
						data["psis"][j],
						(
							"Sy", "Sy",
							"Sz", "Sy", "Sy", "Sz"
						),
						sites24
					)

					XX42 = correlator(
						data["psis"][j],
						(
							"Sz", "Sx", "Sx", "Sz",
							"Sx", "Sx"
						),
						sites42
					)

					XY42 = correlator(
						data["psis"][j],
						(
							"Sz", "Sx", "Sx", "Sz",
							"Sy", "Sy"
						),
						sites42
					)

					YX42 = correlator(
						data["psis"][j],
						(
							"Sz", "Sy", "Sy", "Sz",
							"Sx", "Sx"
						),
						sites42
					)

					YY42 = correlator(
						data["psis"][j],
						(
							"Sz", "Sy", "Sy", "Sz",
							"Sy", "Sy"
						),
						sites42
					)

					XX44 = correlator(
						data["psis"][j],
						(
							"Sz", "Sx", "Sx", "Sz",
							"Sz", "Sx", "Sx", "Sz"
						),
						sites44
					)

					XY44 = correlator(
						data["psis"][j],
						(
							"Sz", "Sx", "Sx", "Sz",
							"Sz", "Sy", "Sy", "Sz"
						),
						sites44
					)

					YX44 = correlator(
						data["psis"][j],
						(
							"Sz", "Sy", "Sy", "Sz",
							"Sz", "Sx", "Sx", "Sz"
						),
						sites44
					)

					YY44 = correlator(
						data["psis"][j],
						(
							"Sz", "Sy", "Sy", "Sz",
							"Sz", "Sy", "Sy", "Sz"
						),
						sites44
					)

					corrsCOS22[j, 1, :] = [
						16 * real(XX22[sites22[l]])
						for l in 1:r
					]

					corrsCOS22[j, 2, :] = [
						16 * real(XY22[sites22[l]])
						for l in 1:r
					]

					corrsCOS22[j, 3, :] = [
						16 * real(YX22[sites22[l]])
						for l in 1:r
					]

					corrsCOS22[j, 4, :] = [
						16 * real(YY22[sites22[l]])
						for l in 1:r
					]

					corrsCOS24[j, 1, :] = [
						64 * real(XX24[sites24[l]])
						for l in 1:r
					]

					corrsCOS24[j, 2, :] = [
						64 * real(XY24[sites24[l]])
						for l in 1:r
					]

					corrsCOS24[j, 3, :] = [
						64 * real(YX24[sites24[l]])
						for l in 1:r
					]

					corrsCOS24[j, 4, :] = [
						64 * real(YY24[sites24[l]])
						for l in 1:r
					]

					corrsCOS42[j, 1, :] = [
						64 * real(XX42[sites42[l]])
						for l in 1:r
					]

					corrsCOS42[j, 2, :] = [
						64 * real(XY42[sites42[l]])
						for l in 1:r
					]

					corrsCOS42[j, 3, :] = [
						64 * real(YX42[sites42[l]])
						for l in 1:r
					]

					corrsCOS42[j, 4, :] = [
						64 * real(YY42[sites42[l]])
						for l in 1:r
					]

					corrsCOS44[j, 1, :] = [
						256 * real(XX44[sites44[l]])
						for l in 1:r
					]

					corrsCOS44[j, 2, :] = [
						256 * real(XY44[sites44[l]])
						for l in 1:r
					]

					corrsCOS44[j, 3, :] = [
						256 * real(YX44[sites44[l]])
						for l in 1:r
					]

					corrsCOS44[j, 4, :] = [
						256 * real(YY44[sites44[l]])
						for l in 1:r
					]
				end

				quantities["corrsCOS22"] = corrsCOS22
				quantities["corrsCOS24"] = corrsCOS24
				quantities["corrsCOS42"] = corrsCOS42
				quantities["corrsCOS44"] = corrsCOS44
			end

			if compute_evs_2
				println("\t | evs_2...")

				evs_2 = zeros((k, 3, N-1))
				for j in 1:k
					X = correlator(data["psis"][j], ("Sx", "Sx"), [(l, l+1) for l in Int(N/2-r/2):Int(N/2+r/2)])
					Y = correlator(data["psis"][j], ("Sy", "Sy"), [(l, l+1) for l in Int(N/2-r/2):Int(N/2+r/2)])
					Z = correlator(data["psis"][j], ("Sz", "Sz"), [(l, l+1) for l in Int(N/2-r/2):Int(N/2+r/2)])

					evs_2[j, 1, Int(N/2-r/2):Int(N/2+r/2)] = [4 * real(X[(l, l+1)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
					evs_2[j, 2, Int(N/2-r/2):Int(N/2+r/2)] = [4 * real(Y[(l, l+1)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
					evs_2[j, 3, Int(N/2-r/2):Int(N/2+r/2)] = [4 * real(Z[(l, l+1)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
				end
				quantities["evs_2"] = evs_2
			end

			if compute_evs_4
				println("\t | evs_4...")

				evs_4 = zeros((k, 3, N-3))
				for j in 1:k
					X = correlator(data["psis"][j], ("Sz", "Sx", "Sx", "Sz"), [(l-1, l, l+1, l+2) for l in Int(N/2-r/2):Int(N/2+r/2)])
					Y = correlator(data["psis"][j], ("Sz", "Sy", "Sy", "Sz"), [(l-1, l, l+1, l+2) for l in Int(N/2-r/2):Int(N/2+r/2)])
					Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz"), [(l-1, l, l+1, l+2) for l in Int(N/2-r/2):Int(N/2+r/2)])

					evs_4[j, 1, Int(N/2-r/2):Int(N/2+r/2)] = [16 * real(X[(l-1, l, l+1, l+2)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
					evs_4[j, 2, Int(N/2-r/2):Int(N/2+r/2)] = [16 * real(Y[(l-1, l, l+1, l+2)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
					evs_4[j, 3, Int(N/2-r/2):Int(N/2+r/2)] = [16 * real(Z[(l-1, l, l+1, l+2)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
				end
				quantities["evs_4"] = evs_4
			end

			if compute_evs_6
				println("\t | evs_6...")

				evs_6 = zeros((k, 3, N-5))
				for j in 1:k					
					Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5) for l in Int(N/2-r/2):Int(N/2+r/2)])

					
					evs_6[j, 3, Int(N/2-r/2):Int(N/2+r/2)] = [16 * 4 * real(Z[(l, l+1, l+2, l+3, l+4, l+5)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
				end
				quantities["evs_6"] = evs_6
			end

			if compute_evs_8
				println("\t | evs_8...")

				evs_8 = zeros((k, 3, N-7))
				for j in 1:k
					Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7) for l in Int(N/2-r/2):Int(N/2+r/2)])

					evs_8[j, 3, Int(N/2-r/2):Int(N/2+r/2)] = [16 * 16 * real(Z[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
				end
				quantities["evs_8"] = evs_8
			end

			if compute_evs_10
				println("\t | evs_10...")

				evs_10 = zeros((k, 3, N-9))
				for j in 1:k
					Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9) for l in Int(N/2-r/2):Int(N/2+r/2)])

					evs_10[j, 3, Int(N/2-r/2):Int(N/2+r/2)] = [16 * 16 * 4 *real(Z[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)]) for l in Int(N/2-r/2):Int(N/2+r/2)]
				end
				quantities["evs_10"] = evs_10
			end

			if compute_opSPT
				println("\t | opSPT...")

				if PBCs == false

					opSPT = zeros((k, 3, N-2))
					for j in 1:k
						X = correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(l, l+1, l+2) for l in 1:N-2])
						Y = correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(l, l+1, l+2) for l in 1:N-2])
						Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(l, l+1, l+2) for l in 1:N-2])

						opSPT[j, 1, :] = [8 * real(X[(l, l+1, l+2)]) for l in 1:N-2]
						opSPT[j, 2, :] = [8 * real(Y[(l, l+1, l+2)]) for l in 1:N-2]
						opSPT[j, 3, :] = [8 * real(Z[(l, l+1, l+2)]) for l in 1:N-2]
					end
					quantities["opSPT"] = opSPT

				elseif PBCs == true

					opSPT = zeros((k, 3, N))
					for j in 1:k
						X = correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(l, l+1, l+2) for l in 1:N-2])
						Y = correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(l, l+1, l+2) for l in 1:N-2])
						Z = correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(l, l+1, l+2) for l in 1:N-2])

						opSPT[j, 1, 2:N-1] = [8 * real(X[(l, l+1, l+2)]) for l in 1:N-2]
						opSPT[j, 2, 2:N-1] = [8 * real(Y[(l, l+1, l+2)]) for l in 1:N-2]
						opSPT[j, 3, 2:N-1] = [8 * real(Z[(l, l+1, l+2)]) for l in 1:N-2]

						opSPT[j, 1, 1] = 8 * real(correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(N, 1, 2)])[(N, 1, 2)])
						opSPT[j, 2, 1] = 8 * real(correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(N, 1, 2)])[(N, 1, 2)])
						opSPT[j, 3, 1] = 8 * real(correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(N, 1, 2)])[(N, 1, 2)])

						opSPT[j, 1, N] = 8 * real(correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(N-1, N, 1)])[(N-1, N, 1)])
						opSPT[j, 2, N] = 8 * real(correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(N-1, N, 1)])[(N-1, N, 1)])
						opSPT[j, 3, N] = 8 * real(correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(N-1, N, 1)])[(N-1, N, 1)])
					end
					quantities["opSPT"] = opSPT
				end
			end	

			if compute_boundary
				println("\t | boundary...")

				boundary = zeros((k, 2))
				for j in 1:k
					boundary[j, 1] = 16 * real(correlator(data["psis"][j], ("Sx", "Sz", "Sz", "Sx"), [(1, 2, N-1, N)])[(1, 2, N-1, N)])
					boundary[j, 2] = 16 * real(correlator(data["psis"][j], ("Sy", "Sz", "Sz", "Sy"), [(1, 2, N-1, N)])[(1, 2, N-1, N)])
				end
				quantities["boundary"] = boundary
			end

			save(joinpath(data_folder, COMPUTE_FOLDER, data_filenames[i]), quantities)

		end
	end
end