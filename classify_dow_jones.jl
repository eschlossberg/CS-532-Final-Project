using DataFrames
include("pegasos.jl")

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
    ret = exp(-norm(x - y)*.02)
    if(ret == 0)
        throw(ErrorException("kernel is zero"))
    end
    return ret
end

@everywhere function polynomial(x::Array{Float64}, y::Array{Float64})
	return (dot(x,y) + 1)^3
end

function main()
    data = readtable("dow_jones_index.data")

    train_indices = filter(x -> data[x,:quarter] == 1, 1:750)
    test_indices = filter(x-> data[x, :quarter] == 2, 1:750)

    feature_vecs = [:open,:high,:low,:close, :volume, :percent_change_price, :percent_change_volume_over_last_wk, :previous_weeks_volume]
    for symb in feature_vecs
        data[isna.(data[symb]), symb] = 0
    end
    data[:volume] = log.(data[:volume]+1)
    data[:previous_weeks_volume] = log.(data[:previous_weeks_volume]+1)
    test_label = :percent_change_next_weeks_price
    X = Array{Float64}(data[train_indices,feature_vecs])
    y  = sign.(Array{Float64}(data[train_indices, test_label ]))
    #maybe add in secondary model for next dividend payout
    X_test = Array{Float64}(data[test_indices, feature_vecs])
    y_test = sign.(Array(data[test_indices, test_label]))
    λ_1 = 0.05
    #build the model
    w_tik = (X'*X + λ_1*eye(size(X,2)))^(-1)*X'*y
    preds = sign.(X_test*w_tik)
    @printf "Tikhonov error rate: %f \n" error_rate(preds, y_test)
    println("training an SVM")

    λs = [1/2^i for i in 1:5 ]
    append!(λs, [2^i for i in 1:5])
    α_best = Array{Float64}(size(X,1))
    min_err = Inf
    for λ in λs
        alpha = pegasos(X,y, λ, 1200,RBF)
        preds_self = Array{Float64}(size(X,1))
        for i in 1:size(X,1)
            preds_self[i] = predict(alpha, RBF, X[i,:], X,y)
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
    @printf "Error on kernel SVM: %f \n" error_rate(preds_svm, y_test)
    what = pegasos(X, y, λ_1, 1500)
    @printf "Error on SVM: %f \n" error_rate(sign.(X_test*what), y_test)
end


main()
