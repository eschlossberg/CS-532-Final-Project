using StatsBase

function pegasos(supports::Array{Float64},y::Array{Float64} ,lamb::Float64, max_iter::Int64, kernel)
	alpha = SharedArray{Float64}(size(supports,1))
	alpha[:] = 0
	@parallel for K = 1:max_iter
		#support vector to increment by
		idx = sample(collect(1:size(supports,1)))
		new_alpha = zeros(size(supports,1))
		filter_idxs = filter( x-> x != idx,collect(1:size(alpha,1)))
		new_alpha[filter_idxs] = alpha[filter_idxs]
		#handle the idx case
		test_val = y[idx]*((lamb*K)^(-1))
		sum_part = 0
		for i = 1:size(supports,1)
			sum_part += alpha[i]*y[idx]*kernel(supports[idx,:], supports[i,:])
		end
		test_val *= sum_part
		if test_val < 1
			new_alpha[idx] = alpha[idx] + 1
		else
			new_alpha[idx] = alpha[idx]
		end
		alpha = new_alpha
	end
	return Array(alpha)
end

#non kernel version of pegasos
function pegasos(supports::Array{Float64}, y::Array{Float64}, lamb::Float64, T::Int64)
	w = zeros(size(supports,2))
	for t = 1:T
		idx = sample(collect(1:size(supports,1)))
		nt = 1/(lamb*t)
		if y[idx]*dot(w, supports[idx,:]) < 1
			w = (1 - nt *lamb)*w + nt*y[idx]* supports[idx,:]
		else
			w = (1 - nt *lamb)*w
		end
	end
	return w
end

function predict(alpha::Array{Float64}, kernel, x::Array{Float64}, supports::Array{Float64}, y::Array{Float64})
	pred_sub = SharedArray{Float64}(size(supports,1))
	pred_sub[:] = 0
	to_check = filter(x-> alpha[x] != 0, 1:size(alpha,1))
	for i in to_check
		pred_sub[i] =  alpha[i]*kernel(x, supports[i,:])*y[i]
	end
	return sign(sum(pred_sub))
end
