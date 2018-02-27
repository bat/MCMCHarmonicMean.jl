include("path_to_package/MCMCHarmonicMean.jl/src/root_interface.jl")

#also set treename/parametertreename if filename has changed
root2hdf5(Float64, "filename.root")
dataset = loadhdf5(Float64, "filename.h5")
data = HMIData(dataset)
hm_integrate(data)
