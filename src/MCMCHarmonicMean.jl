# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

__precompile__(true)

module MCMCHarmonicMean


using ProgressMeter
using Distributions
using HDF5
using StatsBase


include("./DataTypes.jl")
include("./ImportData.jl")

# include("./DataTree.jl")
include("./DataTreeList.jl")

include("./Hyperrectangle.jl")
include("./WhiteningTransformation.jl")
include("./HarmonicMeanIntegration.jl")

include("./Log.jl")


end # module
