# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).
using BAT.Logging
using ProgressMeter
using HDF5
using MCMCHarmonicMean

function loadhdf5(
    T::DataType,
    path::String
)::DataSet

    file = h5open(path, "r")

    N = read(file, "N")#length(data_1st)
    P = read(file, "P")#length(params)

    params = read(file, "parameter_names")

    @log_msg LOG_INFO "File $path \thas $N data points with $P parameters"
    @log_msg LOG_INFO "File Parameter Names: $params"
    data = Matrix{T}(P, N)

    @showprogress for i = 1:P
        data[i, :] = read(file, params[i])
    end
    LogProbability = convert(Vector{T}, read(file, "LogProbability"))
    #LogLikelihood = file["LogLikelihood"]
    #LogPrior = file["LogPrior"]
    Chain = convert(Array{Int64, 1}, read(file, "Chain"))

    @log_msg LOG_DEBUG "Finding Duplicates to generate weighted samples"
    chains = 1
    for i in Chain
        if i > chains
            chains = round(Int64, i)
        end
    end
    chains += 1
    weights = zeros(T, N)
    lastIndex = zeros(Int64, chains)
    removedPoints = 0

    uniqueID = Vector{Int64}(0)

    @showprogress for i = 1:N
        c = Chain[i] + 1
        if lastIndex[c] > 0
            if data[:, i] == data[:, lastIndex[c]]
                weights[lastIndex[c]] += 1.0
                removedPoints += 1
            else
                lastIndex[c] = i
                weights[i] = 1.0
                push!(uniqueID, i)
            end
        else
            lastIndex[c] = i
            weights[i] = 1.0
            push!(uniqueID, i)
        end
    end

    @log_msg LOG_INFO "$removedPoints Duplicates found."
    newData = data[:, uniqueID]
    newLogProb = LogProbability[uniqueID]
    newWeights = weights[uniqueID]

    result = DataSet(newData, newLogProb, newWeights)

    return result
end
