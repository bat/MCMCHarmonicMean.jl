# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

export load_mcmc_data

function load_mcmc_data(path::String, params::Array{String}, range = Colon(), treename::String = "", dataFormat::DataType = Float64)::DataSet
    ending = split(path, ".")[end]
    local res::DataSet

    if ending == "h5"
        res = loadhdf5(dataFormat, path, params, range)
    elseif ending == "root"
        println("ROOT support disabled. use root_interface.jl for conversion to HDF5")
    else
        println("File ending not supported. Use *.root or *.h5")
    end
    return res
end

function loadhdf5(T::DataType, path::String, params::Array{String}, range)::DataSet
    file = h5open(path, "r")

    data_1st = file[params[1]][range]
    N = length(data_1st)
    P = length(params)
    @log_msg LOG_INFO "File $path \thas $N data points with $P parameters"
    data = Matrix{T}(P, N)
    data[1, :] = data_1st
    @showprogress for i in 2:length(params)
        data[i, :] = file[params[i]][range]
    end
    LogProbability = file["LogProbability"][range]
    #LogLikelihood = file["LogLikelihood"][range]
    Chain = convert(Array{Int64, 1}, file["Chain"][range])

    @log_msg LOG_DEBUG "Finding Duplicates to generate weighted samples"
    chains = 1
    for i in Chain
        if i > chains
            chains = round(Int64, i)
        end
    end
    chains += 1
    weights = zeros(N)
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

    result = DataSet(newData, newLogProb, newWeights, N - removedPoints, P)

    return result
end
