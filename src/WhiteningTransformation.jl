# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

function no_whitening(dataset::DataSet)::WhiteningResult
    datamean = zeros(dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    return transform_data(dataset, eye(dataset.P), datamean)
end


function cholesky_whitening(dataset::DataSet)::WhiteningResult
    datamean = zeros(dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    for n in 1:dataset.N
        buffer = view(dataset.data, :, n) - datamean[:]
        setindex!(dataset.data, buffer, :, n)
    end

    covmatrix = cov(transpose(dataset.data), FrequencyWeights(dataset.weights), corrected=true)
    covmatrix_inv = inv(Symmetric(covmatrix))
    w = chol(covmatrix_inv)
    wres = convert(Array{Float64, 2}, w)

    return transform_data(dataset, wres, datamean)
end

function statistical_whitening(dataset::DataSet)::WhiteningResult
    datamean = zeros(dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    for n in 1:dataset.N
        dataset.data[:, n] -= datamean
    end

    covmatrix = cov(transpose(dataset.data), FrequencyWeights(dataset.weights), corrected=true)

    E = eigfact(covmatrix).vectors
    w_d = transpose(E)
    D = Diagonal(inv(E) * covmatrix * E)
    w = inv(full(sqrt.(D))) * w_d
    wres = convert(Array{Float64, 2}, w)

    return transform_data(dataset, wres, datamean)
end

function transform_data{T<:AbstractFloat}(dataset::DataSet{T}, W::Matrix{T}, datamean::Vector{T})::WhiteningResult
    local determinant::Float64

    if W == eye(dataset.P)
        determinant = 1.0
    else
        buffer = Vector{T}(dataset.P)

        dataset.data = W * dataset.data

        determinant = abs(det(W))
    end

    sortedLogProb = sortperm(dataset.logprob, rev = true)

    box = Matrix{Float64}(dataset.P, 2)
    box[:, 1] = dataset.data[:, sortedLogProb[1]]
    box[:, 2] = dataset.data[:, sortedLogProb[1]]

    for n in sortedLogProb[2:floor(Int64, dataset.N * 0.50)]
        for p in 1:dataset.P
            if dataset.data[p,n] > box[p, 2]
                box[p, 2] = dataset.data[p,n]
            elseif dataset.data[p,n] < box[p, 1]
                box[p, 1] = dataset.data[p,n]
            end
        end
    end

    LogMedium("Calculate Optimal Target Probability")
    maxP = dataset.logprob[sortedLogProb[1]]
    minP = dataset.logprob[sortedLogProb[dataset.N]]

    suggTargetProb = dataset.logprob[sortedLogProb[floor(Int64, dataset.N * 0.8)]]
    suggTargetProb = exp(maxP - suggTargetProb)
    #suggTargetProb = min(100, exp(maxP - suggTargetProb))

    LogMedium("Determinant:\t" * string(determinant))
    LogMedium("Suggested Target Probability Factor:\t" * string(suggTargetProb))
    LogMedium("Maximum Probability Factor:\t" * string(exp(maxP - minP)))

    return WhiteningResult(determinant, box, maxP-minP, suggTargetProb, W, datamean)
end
