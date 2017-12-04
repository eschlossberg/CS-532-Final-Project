using StatsBase

function pegasos(supports::Array{Float64},y::Array{Float64} ,lamb::Float64, max_iter::Int64, kernel)
	alpha = zeros(size(supports,2))
	for K = 1:max_iter
		#support vector to increment by
		idx = sample(collect(1:size(supports,2)))
		new_alpha = zeros(size(suppors,2))
		filter_idxs = filter( x-> x != idx,collect(1:size(alpha,1)))
		new_alpha[filter_idxs] = alpha[filter_idxs]
		#handle the idx case
		test_val = y[idx]*(lamb*K)^(-1)
		sum_part = 0
		for i = 1:size(supports,2)
			sum_part += alpha[i]*y[idx]*kernel(supports[:,idx], supports[:,i])
		end
		test_val *= sum_part
		if test_val < 1
			new_alpha[idx] = alpha[idx] + 1
		else
			new_alpha[idx] = alpha[idx]
		end
		alpha = new_alpha
	end
	return alpha
end

#phi is just the transformation into a higher dimensional space
function alpha_to_w(alpha::Array{Float64}, lamb::Float64, t::Int64, supports::Array{float64},y::Array{Float64},phi )
	sum_part = 0
	for i =1:size(supports,2)
		sum_part +=alpha[i]*y[i]*phi(supports[:,i])
	end
	return (1/(lamb*t))*sum_part
end
