using DataFrames
using StatsBase
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
    println("########### BEGIN TEST ###########")
    data = readtable("phishing_attacks.csv")
    #consider suspicious to be phishing attack
    sus_indexes = filter(x->data[x, :Result] == 0, 1:size(data,1))
    data[sus_indexes, :Result] = -1
    features = [:SFH,:popUpWidnow,:SSLfinal_State,:Request_URL,:URL_of_Anchor,:web_traffic,:URL_Length,:age_of_domain,:having_IP_Address]
    label = :Result
    #now we're ready to randomly split our dataset
    train_indexes = sample(collect(1:size(data,1)),Int64(round(size(data,1)/4)), replace=false )
    test_indexes = filter(x-> x ∉ train_indexes, 1:size(data,1))
    X = Array{Float64}(data[train_indexes, features])
    Γ = Array{Float64}(data[test_indexes, features])
    y = Array{Float64}(data[train_indexes, label])
    ω = Array{Float64}(data[test_indexes, label])
    #build some models
    λ = .005
    w_tik = (X'*X + λ*eye(size(X,2)))^(-1)*X'*y
    preds_tik = sign.(Γ*w_tik)
    @printf "Thikonov: %f \n" error_rate(preds_tik, ω)
    #train some SVMS
    α = pegasos(X,y, λ, 500,RBF)
    preds_svm = Array{Float64}(size(Γ, 1))
    for i in 1:size(Γ,1)
        preds_svm[i] = predict(α, RBF, Γ[i,:], X, y)
    end
    @printf "Kernel SVM: %f \n" error_rate(preds_svm, ω)
    #finally the regular SVM
    w_hat = pegasos(X,y , λ, 900)
    @printf "SVM: %f \n" error_rate(sign.(Γ*w_hat), ω)



end

main()
