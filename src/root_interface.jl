using ROOTFramework, Cxx
using HDF5
using ProgressMeter


function MCMCHarmonicMean.root2hdf5(T::DataType, path::String, params::Array{String}, range = Colon(), treename::String = "")
    if treename == ""
        treename = convert(String, split(split(path, ".")[1], "/")[end])
    end
    filename_hdf5 = path[1:end-5] * ".h5"

    dim = length(params)
    filename_root = path

    valnames = Array{Symbol, 1}(length(params) + 4)
    valnames[1:length(params)] = params
    valnames[end - 3] = :LogProbability
    valnames[end - 2] = :LogLikelihood
    valnames[end - 1] = :LogPrior
    valnames[end] = :Chain

    bindings = TTreeBindings()
    root_values = [bindings[bname] = Ref(zero(Float64)) for bname in valnames[1:dim+3]]::Array{Base.RefValue{Float64},1}
    root_integers = [bindings[bname] = Ref(zero(UInt32)) for bname  in valnames[dim+4:dim+4]]::Array{Base.RefValue{UInt32}, 1}

    open(TChainInput, bindings, treename, filename_root) do input
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

        h5open(filename_hdf5, "w") do file
            chunk_size = max(n, 16*1024)
            dsargs = ("chunk", (chunk_size,), "compress", 6)

            @showprogress for j in eachindex(valnames, hdf5_arrays)
                file[string(valnames[j]), dsargs...] = hdf5_arrays[j]
            end
        end
    end
end
