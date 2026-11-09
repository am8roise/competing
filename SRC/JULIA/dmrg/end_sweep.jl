using ITensors, ITensorMPS
using FileIO

mutable struct Observer <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   H::MPO
   data_filename::String
   save_every::Bool

   Observer(H, data_filename, save_every, energy_tol) = new(energy_tol, 0.0, H, data_filename, save_every)
end

function ITensorMPS.checkdone!(o::Observer; kwargs...)
	psi = kwargs[:psi]
	energy = kwargs[:energy]

	if o.save_every
		dict = Dict("psis" => [psi], "energies" => [energy])
		save(o.data_filename, dict)
	end

	if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
		println("Energy difference less than $(o.energy_tol), stopping DMRG")
		return true
	end

	o.last_energy = energy
	return false
end