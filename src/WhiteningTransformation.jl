# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


function data_whitening{T<:AbstractFloat, I<:Integer}(method::Symbol, dataset::DataSet{T, I})::WhiteningResult{T}
    local whiteningresult::WhiteningResult
    if method == :CholeskyWhitening
        whiteningresult = cholesky_whitening(dataset)
    elseif method == :StatisticalWhitening
        whiteningresult = statistical_whitening(dataset)
    elseif method == :NoWhitening
        whiteningresult = no_whitening(dataset)
    else
        error("Unknown whitening method. Use :CholeskyWhitening, :StatisticalWhitening or :NoWhitening")
    end
    return whiteningresult
end

function no_whitening{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I})::WhiteningResult{T}
    datamean = Array{Float64}(0)

    return transform_data(dataset, eye(dataset.P), datamean)
end


function cholesky_whitening{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I})::WhiteningResult{T}
    datamean = zeros(dataset.P)

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
    wres = convert(Array{Float64, 2}, w)

    return transform_data(dataset, wres, datamean)
end

function statistical_whitening{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I})::WhiteningResult{T}
    datamean = zeros(dataset.P)

    for p in eachindex(datamean)
        datamean[p] = mean(view(dataset.data, p, :))
    end

    dataset.data .-= datamean

    covmatrix = cov(transpose(dataset.data), FrequencyWeights(dataset.weights), corrected=true)

    E = eigfact(covmatrix).vectors
    w_d = transpose(E)
    D = Diagonal(inv(E) * covmatrix * E)
    w = inv(full(sqrt.(D))) * w_d
    wres = convert(Array{Float64, 2}, w)

    return transform_data(dataset, wres, datamean)
end

function transform_data{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, W::Matrix{T}, datamean::Vector{T})::WhiteningResult{T}
    local determinant::Float64

    if W == eye(dataset.P)
        determinant = 1.0
    else
        buffer = deepcopy(dataset.data)

        Base.A_mul_B!(dataset.data, W, buffer)

        determinant = abs(det(W))
    end

    LogMedium("Calculate Optimal Target Probability")
    sortids = sortperm(dataset.logprob, rev=true)
    maxP = dataset.logprob[sortids[1]]
    minP = dataset.logprob[sortids[end]]

    suggTargetProb = dataset.logprob[sortids[floor(Int64, dataset.N * 0.9)]]
    suggTargetProb = exp(maxP - suggTargetProb)
    #suggTargetProb = min(100, exp(maxP - suggTargetProb))

    LogMedium("Determinant:\t" * string(determinant))
    LogMedium("Suggested Target Probability Factor:\t" * string(suggTargetProb))
    LogMedium("Maximum Probability Factor:\t" * string(exp(maxP - minP)))

    return WhiteningResult(determinant, maxP-minP, suggTargetProb, W, datamean)
end
