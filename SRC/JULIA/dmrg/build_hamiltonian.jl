using ITensors, ITensorMPS

function build_hamiltonian(model, params)
	
	if (model["name"] == "A") | (model["name"] == "B") | (model["name"] == "C")
		
		N = params["N"]
		λ = params["lam"]
		θ = params["theta"]

		sites = siteinds("S=1/2", N)
		os = OpSum()

		if model["name"] == "A"

			for j=1:N-1
				os += cos(λ)*cos(θ) * 4, "Sx", j, "Sx", j+1
				os += cos(λ)*sin(θ) * 4, "Sy", j, "Sy", j+1
			end

			for j=1:N-3
				os += sin(λ)*cos(θ) * 16, "Sz", j, "Sx", j+1, "Sx", j+2, "Sz", j+3
				os += sin(λ)*sin(θ) * 16, "Sz", j, "Sy", j+1, "Sy", j+2, "Sz", j+3
			end

			if model["PBCs"]

				os += cos(λ)*cos(θ) * 4, "Sx", N, "Sx", 1
				os += cos(λ)*sin(θ) * 4, "Sy", N, "Sy", 1

				os += sin(λ)*cos(θ) * 16, "Sz", N-2, "Sx", N-1, "Sx", N, "Sz", 1
				os += sin(λ)*sin(θ) * 16, "Sz", N-2, "Sy", N-1, "Sy", N, "Sz", 1
				os += sin(λ)*cos(θ) * 16, "Sz", N-1, "Sx", N, "Sx", 1, "Sz", 2
				os += sin(λ)*sin(θ) * 16, "Sz", N-1, "Sy", N, "Sy", 1, "Sz", 2
				os += sin(λ)*cos(θ) * 16, "Sz", N, "Sx", 1, "Sx", 2, "Sz", 3
				os += sin(λ)*sin(θ) * 16, "Sz", N, "Sy", 1, "Sy", 2, "Sz", 3
			end

		elseif model["name"] == "B"

			for j=1:N-1
				os += cos(λ)*cos(θ) * 4, "Sx", j, "Sx", j+1
				os += cos(λ)*sin(θ) * 4, "Sy", j, "Sy", j+1
			end
			
			for j=1:N-3
				os += sin(λ)*sin(θ) * 16, "Sz", j, "Sx", j+1, "Sx", j+2, "Sz", j+3
				os += sin(λ)*cos(θ) * 16, "Sz", j, "Sy", j+1, "Sy", j+2, "Sz", j+3
			end

			if model["PBCs"]

				os += cos(λ)*cos(θ) * 4, "Sx", N, "Sx", 1
				os += cos(λ)*sin(θ) * 4, "Sy", N, "Sy", 1

				os += sin(λ)*sin(θ) * 16, "Sz", N-2, "Sx", N-1, "Sx", N, "Sz", 1
				os += sin(λ)*cos(θ) * 16, "Sz", N-2, "Sy", N-1, "Sy", N, "Sz", 1
				os += sin(λ)*sin(θ) * 16, "Sz", N-1, "Sx", N, "Sx", 1, "Sz", 2
				os += sin(λ)*cos(θ) * 16, "Sz", N-1, "Sy", N, "Sy", 1, "Sz", 2
				os += sin(λ)*sin(θ) * 16, "Sz", N, "Sx", 1, "Sx", 2, "Sz", 3
				os += sin(λ)*cos(θ) * 16, "Sz", N, "Sy", 1, "Sy", 2, "Sz", 3
			end
		
		elseif model["name"] == "C"

			for j=1:N-1
				os += cos(λ)*cos(θ) * 4, "Sx", j, "Sx", j+1
				os += sin(λ)*sin(θ) * 4, "Sy", j, "Sy", j+1
			end
			
			for j=1:N-3
				os += cos(λ)*sin(θ) * 16, "Sz", j, "Sx", j+1, "Sx", j+2, "Sz", j+3
				os += sin(λ)*cos(θ) * 16, "Sz", j, "Sy", j+1, "Sy", j+2, "Sz", j+3
			end

			if model["PBCs"]

				os += cos(λ)*cos(θ) * 4, "Sx", N, "Sx", 1
				os += sin(λ)*sin(θ) * 4, "Sy", N, "Sy", 1

				os += cos(λ)*sin(θ) * 16, "Sz", N-2, "Sx", N-1, "Sx", N, "Sz", 1
				os += sin(λ)*cos(θ) * 16, "Sz", N-2, "Sy", N-1, "Sy", N, "Sz", 1
				os += cos(λ)*sin(θ) * 16, "Sz", N-1, "Sx", N, "Sx", 1, "Sz", 2
				os += sin(λ)*cos(θ) * 16, "Sz", N-1, "Sy", N, "Sy", 1, "Sz", 2
				os += cos(λ)*sin(θ) * 16, "Sz", N, "Sx", 1, "Sx", 2, "Sz", 3
				os += sin(λ)*cos(θ) * 16, "Sz", N, "Sy", 1, "Sy", 2, "Sz", 3
			end
		end

	else
		println("***** Model $(model["name"]) is not defined *****")
	end

	return os, sites
end