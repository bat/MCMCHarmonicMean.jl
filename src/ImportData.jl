
export LoadMCMCData

function LoadMCMCData(path::String, params::Array{String}, range = Colon(), modelname::String = "", dataFormat::DataType = Float64)::DataSet
    ending = split(path, ".")[end]
    local res::DataSet

    if ending == "h5"
        @time res = loadHDF5(dataFormat, path, params, range)
    else
        error("File Ending not known. Use *.h5 files")
    end
    return res
end

function loadHDF5(T::DataType, path::String, params::Array{String}, range)::DataSet
    file = h5open(path, "r")

    data_1st = file[params[1]][range]
    N = length(data_1st)
    P = length(params)
    LogHigh("File $path \thas $N data points with $P parameters")
    data = Matrix{T}(P, N)
    data[1, :] = data_1st
    @showprogress for i in 2:length(params)
        data[i, :] = file[params[i]][range]
    end
    LogProbability = file["LogProbability"][range]
    #LogLikelihood = file["LogLikelihood"][range]
    Chain = convert(Array{Int64, 1}, file["Chain"][range])

    LogMedium("Finding Duplicates")
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

    LogHigh("Deleted $removedPoints Duplicates")
    newData = data[:, uniqueID]
    newLogProb = LogProbability[uniqueID]
    newWeights = weights[uniqueID]

    result = DataSet{T}(newData, newLogProb, newWeights, N - removedPoints, P)

    return result
end


function root2hdf5(T::DataType, path::String, params::Array{String}, range, modelname::String)
    error("ROOT support not enabled")
end
