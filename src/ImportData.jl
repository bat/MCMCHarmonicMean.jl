
export LoadMCMCData

function LoadMCMCData(path::String, params::Array{String}, range = Colon(), modelname::String = "", dataFormat::DataType = Float64)::DataSet
    ending = split(path, ".")[end]
    local res::DataSet

    if ending == "h5"
        @time res = loadHDF5(dataFormat, path, params, range)
    elseif ending == "root"
        @time res = loadRoot(dataFormat, path, params, range, modelname)
    else
        error("File Ending not known. Use *.root or *.h5 files")
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

function loadROOT(T::DataType, path::String, params::Array{String}, range, modelname::String)
    if modelname == ""
        modelname = split(split(path, ".")[1], "/")[end]
    end
    filename_hdf5 = path[1:end-5] * ".h5"

    valnames = Array{Symbol, 1}(length(params) + 4)
    valnames[1:length(params)] = params
    valnames[end - 3] = :LogProbability
    valnames[end - 2] = :LogLikelihood
    valnames[end - 1] = :LogPrior
    valnames[end] = :Chain

    bindings = TTreeBindings()
    root_values = [bindings[bname] = Ref(zero(Float64)) for bname in valnames[1:dim+3]]::Array{Base.RefValue{Float64},1}
    root_integers = [bindings[bname] = Ref(zero(UInt32)) for bname  in valnames[dim+4:dim+4]]::Array{Base.RefValue{UInt32}, 1}

    open(TChainInput, bindings, modelname, filename_root) do input
        n = length(input)
        info("$input has $n entries.")
        hdf5_arrays = [zeros(Float32, n) for _ in valnames]::Array{Array{Float32,1},1}
        i = 0
        @showprogress for _ in input
            i += 1
            @inbounds for j in eachindex(root_values, hdf5_arrays)
                hdf5_arrays[j][i] =
                try
                    root_values[j].x
                catch
                    root_integers[1].x
                end
            end
        end
        if convertToHDF5
            h5open(filename_hdf5, "w") do file
                chunk_size = max(n, 16*1024)
                dsargs = ("chunk", (chunk_size,), "compress", 6)

                @showprogress for j in eachindex(valnames, hdf5_arrays)
                    file[string(valnames[j]), dsargs...] = hdf5_arrays[j]
                end
            end
        end
    end

    return loadHDF5(T, path, params, range)
end
