using ITensors, ITensorMPS

function binder_cumulant(ψ::MPS, opname::AbstractVector{<:AbstractString}; SPT=false, staggered=false)
	
	s = siteinds(ψ)
	n = SPT ? length(s) - 2 : length(s)

	weights = staggered ? (-1) .^ (1:n) : ones(Float64, n)

	norm = 2. / n

	os = OpSum()
	for j in 1:n
		if SPT
			os += norm * weights[j], opname[1], j, opname[2], j+1, opname[3], j+2
		else
			os += norm * weights[j], opname[1], j
		end
	end

	M = MPO(os, s)
	
	ϕ = apply(M, ψ)
	mm = real(inner(ϕ', ϕ))

	χ = apply(M, ϕ)
	mmmm  = real(inner(χ', χ))

	return  1 - mmmm / (3 * mm ^ 2)
end
