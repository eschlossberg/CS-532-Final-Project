using DataFrames
import Base.sign
include("pegasos.jl")
function sign(x::Bool)
    if x
        return 1.0
    else
        return -1.0
    end
end

function error_rate(test::Array{Float64}, real::Array{Float64})
    trials = size(test,1)
    ninc = 0
    for i in 1:trials
        if test[i] != 1.0 && test[i] != -1.0
            throw(ArgumentError("not a vector of prediction labels"))
        end
        if test[i] != real[i]
            ninc +=1
        end
    end
    return ninc/trials
end
@everywhere function RBF(x::Array{Float64}, y::Array{Float64})
    return exp(-norm(x - y)*.0002)
end

@everywhere function polynomial(x::Array{Float64}, y::Array{Float64})
	return (dot(x,y) + 1)^2
end

@everywhere function histogram_intersection(x::Array{Float64}, y::Array{Float64})
    return sum(min.(x,y))
end

@everywhere function hyper_tangent(x::Array{Float64}, y::Array{Float64})
    return tanh(3*dot(x,y) +1)
end
function main()
    kernel = RBF
    println("############## BEGIN TEST ##############")
    data = readtable("dow_jones_index.data")

    train_indices = filter(x -> data[x,:quarter] == 1, 1:750)
    test_indices = filter(x-> data[x, :quarter] == 2, 1:750)

    feature_vecs = [:open,:high,:low,:close, :volume, :percent_change_price, :percent_change_volume_over_last_wk, :previous_weeks_volume, :next_weeks_open]
    for symb in feature_vecs
        data[isna.(data[symb]), symb] = 0
    end
    data[:volume] = log.(data[:volume]+1)
    data[:previous_weeks_volume] = log.(data[:previous_weeks_volume]+1)
    test_label = :percent_change_next_weeks_price
    X = Array{Float64}(data[train_indices,feature_vecs])
    train_data = data[train_indices,:]
    y  = Array{Float64}(sign.(train_data[:percent_return_next_dividend] .> 1))
    #maybe add in secondary model for next dividend payout
    X_test = Array{Float64}(data[test_indices, feature_vecs])
    test_data = data[test_indices,:]
    y_test = Array{Float64}(sign.(test_data[:percent_return_next_dividend] .> 1))

    λ_1 = 0.05
    #build the model
    w_tik = (X'*X + λ_1*eye(size(X,2)))^(-1)*X'*y
    preds = sign.(X_test*w_tik)
    @printf "Tikhonov error rate: %f \n" error_rate(preds, y_test)
    println("training an SVM")

    λs = [1/2^i for i in 1:10 ]
    append!(λs, [2^i for i in 1:5])
    α_best = Array{Float64}(size(X,1))
    min_err = Inf
    for λ in λs
        alpha = pegasos(X,y, λ, 4000,kernel)
        preds_self = Array{Float64}(size(X,1))
        for i in 1:size(X,1)
            preds_self[i] = predict(alpha, kernel, X[i,:], X,y)
        end
        err_rate = error_rate(preds_self,y)
        if err_rate < min_err
            min_err = err_rate
            α_best = alpha
        end
    end
    preds_svm = Array{Float64}(size(X_test, 1))
    for i in 1:size(X_test,1)
        preds_svm[i] = predict(α_best, polynomial, X_test[i,:], X, y)
    end
    println(filter(i -> preds_svm[i] != -1, 1:size(preds_svm,1)))
    @printf "Error on kernel SVM: %f \n" error_rate(preds_svm, y_test)
    what = pegasos(X, y, λ_1, 1500)
    @printf "Error on SVM: %f \n" error_rate(sign.(X_test*what), y_test)
end


main()
