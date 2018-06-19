# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


function no_whitening{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I})::WhiteningResult{T}
    datamean = Vector{T}(0)

    return transform_data(dataset, eye(T, dataset.P), datamean)
end


function cholesky_whitening{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I})::WhiteningResult{T}
    datamean = zeros(T, dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    for n in 1:dataset.N
        buffer = view(dataset.data, :, n) - datamean
        setindex!(dataset.data, buffer, :, n)
    end

    covmatrix = cov(transpose(dataset.data), FrequencyWeights(dataset.weights), corrected=true)
    covmatrix_inv = inv(Symmetric(covmatrix))
    w = chol(covmatrix_inv)
    wres = convert(Matrix{T}, w)

    return transform_data(dataset, wres, datamean)
end

function statistical_whitening{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I})::WhiteningResult{T}
    datamean = zeros(T, dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    dataset.data .-= datamean

    covmatrix = cov(transpose(dataset.data), FrequencyWeights(dataset.weights), corrected=true)

    E = eigfact(covmatrix).vectors
    w_d = transpose(E)
    D = Diagonal(inv(E) * covmatrix * E)
    w = inv(full(sqrt.(D))) * w_d
    wres = convert(Matrix{T}, w)

    return transform_data(dataset, wres, datamean)
end

function transform_data{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, W::Matrix{T}, datamean::Vector{T})::WhiteningResult{T}
    local determinant::T

    if W == eye(T, dataset.P)
        determinant = 1.0
    else
        dataset.data = W * dataset.data

        determinant = abs(det(W))
    end

    maxP::T = select(dataset.logprob, dataset.N)
    suggTargetProb::T = exp(maxP - select(dataset.logprob, floor(Int64, dataset.N * 0.2)))

    dataset.iswhitened = true
    
    @log_msg LOG_DEBUG "Determinant:\t" * string(determinant)
    @log_msg LOG_DEBUG "Suggested Target Probability Factor:\t" * string(suggTargetProb)

    return WhiteningResult(determinant, suggTargetProb, W, datamean)
end
