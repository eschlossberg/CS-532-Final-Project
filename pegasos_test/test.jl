include("../pegasos.jl")
using MLDatasets
train_data, train_labels = MNIST.traindata()
#build training vectors
println("building training vectors ...")
train_data = reshape(train_data[:,:,1:10000], 10000, 784)
#build the label vector
#create indices for recoginizing 0
train_labels = train_labels[1:10000]
zero_indices = filter(x -> train_labels[x] == 0, 1:size(train_labels, 1))
y_filtered = zeros(10000, 1)
y_filtered[:] = -1.0
y_filtered[zero_indices] = 1.0

function error_rate(test, real)
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

X = train_data
y = y_filtered
lamb = 0.5

w = (X'*X + lamb *eye(784))^(-1)*X'*y
#now use w to make preds
println("building test set ...")
pred_vecs, test_labels = MNIST.testdata()
pred_vecs = reshape(pred_vecs[:,:,1:5000],5000 ,784 )

real_ans = test_labels
zero_indices = filter(x->real_ans[x] == 0, 1:size(test_labels,1))
real_ans[:] = -1.0
real_ans[zero_indices] = 1.0

preds = sign.(pred_vecs*w)
@printf "Thikonov Error Rate on MNIST: %f\n" error_rate(preds, real_ans)

#define a kernel function
@everywhere function RBF(x::Array{Float64}, y::Array{Float64})
    return exp(-norm(x - y)*.02)
end

@everywhere function polynomial(x::Array{Float64}, y::Array{Float64})
	return (dot(x,y) + 1)^2
end

alpha = pegasos(X,y, 0.005, 2000, RBF)
println("first training round complete")
println(alpha != zeros(size(X,1)))
println("making predictions")
preds = SharedArray{Float64}(size(pred_vecs,1))
@sync @parallel for idx in 1:size(pred_vecs, 1)
	preds[idx] = predict(alpha, RBF, pred_vecs[idx,:], X,y)
end
@printf "Pegasos Error Rate with RBF on MNIST: %f \n" error_rate(Array(preds), real_ans)
@printf "polynomial kernel \n"
alpha_poly = pegasos(X, y, 0.05, 2000, polynomial)



preds_poly = SharedArray{Float64}(size(pred_vecs,1))
@sync @parallel for i in 1:size(pred_vecs,1)
	preds_poly[i] = predict(alpha_poly, polynomial, pred_vecs[i,:], X,y)
end
@printf "Pegasos Polynomial Kernel: %f \n" error_rate(Array(preds_poly), real_ans)
w_2 = pegasos(X,y, 0.1,2000 )
@printf "Pegasos No Kernel: %f \n" error_rate(sign.(pred_vecs*w_2), real_ans)
@printf "Vector of all ones %f\n" error_rate(ones(size(real_ans,1)), real_ans)
