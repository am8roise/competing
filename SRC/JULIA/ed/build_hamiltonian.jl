using MPSKit, MPSKitModels, TensorKit, LinearAlgebra, KrylovKit

id = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
σx = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
σy = TensorMap(ComplexF64[0 -1im; 1im 0], ℂ^2 ← ℂ^2)
σz = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

function build(model, params)

	N = params["N"]
	lam = params["lam"]
	theta = params["theta"]
	
	if model["PBCs"]

		if model["name"] == "A"

			infinite_chain = PeriodicVector([ℂ^2])
			H = InfiniteMPOHamiltonian(infinite_chain, (1, 2) => cos(lam)*cos(theta) * σx⊗σx + cos(lam)*sin(theta) * σy⊗σy, (1, 2, 3, 4) => sin(lam)*cos(theta) * σz⊗σx⊗σx⊗σz + sin(lam)*sin(theta) * σz⊗σy⊗σy⊗σz)

			H = periodic_boundary_conditions(H, Int(N))

		elseif model["name"] == "B"

			infinite_chain = PeriodicVector([ℂ^2])
			H = InfiniteMPOHamiltonian(infinite_chain, (1, 2) => cos(lam)*cos(theta) * σx⊗σx + cos(lam)*sin(theta) * σy⊗σy, (1, 2, 3, 4) => sin(lam)*sin(theta) * σz⊗σx⊗σx⊗σz + sin(lam)*cos(theta) * σz⊗σy⊗σy⊗σz)

			H = periodic_boundary_conditions(H, Int(N))

		elseif model["name"] == "C"

			infinite_chain = PeriodicVector([ℂ^2])
			H = InfiniteMPOHamiltonian(infinite_chain, (1, 2) => cos(lam)*cos(theta) * σx⊗σx + sin(lam)*sin(theta) * σy⊗σy, (1, 2, 3, 4) => cos(lam)*sin(theta) * σz⊗σx⊗σx⊗σz + sin(lam)*cos(theta) * σz⊗σy⊗σy⊗σz)

			H = periodic_boundary_conditions(H, Int(N))

		end

	else

		if model["name"] == "A"

			chain = fill(ℂ^2, Int(N))
			H  = cos(lam)*cos(theta) * FiniteMPOHamiltonian(chain, (x, x+1) => σx⊗σx for x in 1:(length(chain)-1))
			H += cos(lam)*sin(theta) * FiniteMPOHamiltonian(chain, (x, x+1) => σy⊗σy for x in 1:(length(chain)-1))
			H += sin(lam)*cos(theta) * FiniteMPOHamiltonian(chain, (x, x+1, x+2, x+3) => σz⊗σx⊗σx⊗σz for x in 1:(length(chain)-3))
			H += sin(lam)*sin(theta) * FiniteMPOHamiltonian(chain, (x, x+1, x+2, x+3) => σz⊗σy⊗σy⊗σz for x in 1:(length(chain)-3))
		
		elseif model["name"] == "B"

			chain = fill(ℂ^2, Int(N))
			H  = cos(lam)*cos(theta) * FiniteMPOHamiltonian(chain, (x, x+1) => σx⊗σx for x in 1:(length(chain)-1))
			H += cos(lam)*sin(theta) * FiniteMPOHamiltonian(chain, (x, x+1) => σy⊗σy for x in 1:(length(chain)-1))
			H += sin(lam)*sin(theta) * FiniteMPOHamiltonian(chain, (x, x+1, x+2, x+3) => σz⊗σx⊗σx⊗σz for x in 1:(length(chain)-3))
			H += sin(lam)*cos(theta) * FiniteMPOHamiltonian(chain, (x, x+1, x+2, x+3) => σz⊗σy⊗σy⊗σz for x in 1:(length(chain)-3))

		elseif model["name"] == "C"

			chain = fill(ℂ^2, Int(N))
			H  = cos(lam)*cos(theta) * FiniteMPOHamiltonian(chain, (x, x+1) => σx⊗σx for x in 1:(length(chain)-1))
			H += sin(lam)*sin(theta) * FiniteMPOHamiltonian(chain, (x, x+1) => σy⊗σy for x in 1:(length(chain)-1))
			H += cos(lam)*sin(theta) * FiniteMPOHamiltonian(chain, (x, x+1, x+2, x+3) => σz⊗σx⊗σx⊗σz for x in 1:(length(chain)-3))
			H += sin(lam)*cos(theta) * FiniteMPOHamiltonian(chain, (x, x+1, x+2, x+3) => σz⊗σy⊗σy⊗σz for x in 1:(length(chain)-3))

		end
	end

	return H
end