

export CholeskyWhitening, StatisticalWhitening



function CholeskyWhitening(Data::DataSet)::WhiteningResult
    data_mean = zeros(Data.P)

    for p in eachindex(data_mean)
        data_mean[p] = mean(view(Data.Data, p, :))
    end

    for n in 1:Data.N
        buffer = view(Data.Data, :, n) - data_mean[:]
        setindex!(Data.Data, buffer, :, n)
    end

    covmatrix = cov(transpose(Data.Data), FrequencyWeights(Data.Weights), corrected=true)
    covmatrix_inv = inv(Symmetric(covmatrix))
    w = chol(covmatrix_inv)
    wres = convert(Array{Float64, 2}, w)

    return TransformData(Data, wres, data_mean)
end

function StatisticalWhitening(Data::DataSet)::WhiteningResult
    data_mean = zeros(Data.P)

    for p in eachindex(data_mean)
        data_mean[p] = mean(view(Data.Data, p, :))
    end

    for n in 1:Data.N
        Data.Data[:, n] -= data_mean
    end

    covmatrix = cov(transpose(Data.Data), FrequencyWeights(Data.Weights), corrected=true)

    E = eigfact(covmatrix).vectors
    w_d = transpose(E)
    D = Diagonal(inv(E) * covmatrix * E)
    w = inv(full(sqrt.(D))) * w_d
    wres = convert(Array{Float64, 2}, w)

    return TransformData(Data, wres, data_mean)
end

function TransformData{T<:AbstractFloat}(Data::DataSet{T}, W::Matrix{T}, Data_Mean::Vector{T})::WhiteningResult
    buffer = Vector{T}(Data.P)

    Data.Data = W * Data.Data

    sortedLogProb = sortperm(Data.LogProb, rev = true)
    determinant = abs(det(W))

    box = Matrix{Float64}(Data.P, 2)
    box[:, 1] = Data.Data[:, sortedLogProb[1]]
    box[:, 2] = Data.Data[:, sortedLogProb[1]]

    for n in sortedLogProb[2:floor(Int64, Data.N * 0.50)]
        for p in 1:Data.P
            if Data.Data[p,n] > box[p, 2]
                box[p, 2] = Data.Data[p,n]
            elseif Data.Data[p,n] < box[p, 1]
                box[p, 1] = Data.Data[p,n]
            end
        end
    end

    LogMedium("Calculate Optimal Target Probability")
    maxP = Data.LogProb[sortedLogProb[1]]
    minP = Data.LogProb[sortedLogProb[Data.N]]

    suggTargetProb = Data.LogProb[sortedLogProb[floor(Int64, Data.N * 0.8)]]
    suggTargetProb = exp(maxP - suggTargetProb)
    #suggTargetProb = min(100, exp(maxP - suggTargetProb))

    LogMedium("Determinant:\t" * string(determinant))
    LogMedium("Suggested Target Probability Factor:\t" * string(suggTargetProb))
    LogMedium("Maximum Probability Factor:\t" * string(exp(maxP - minP)))

    return WhiteningResult(determinant, box, maxP-minP, suggTargetProb, W, Data_Mean)
end
