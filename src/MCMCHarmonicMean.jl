# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

__precompile__(true)

module MCMCHarmonicMean


using ProgressMeter
using Distributions
using HDF5
using StatsBase
using BAT
using Base.Threads
using MultiThreadingTools

include("./DataTypes.jl")

# include("./DataTree.jl")
include("./DataTreeList.jl")

include("./PointCloud.jl")
include("./IntegrationVolume.jl")
include("./ImportData.jl")


include("./Hyperrectangle.jl")
include("./WhiteningTransformation.jl")
include("./HarmonicMeanIntegration.jl")

include("./Log.jl")



export hm_integrate

export DataSet

export HMIntegrationPrecisionSettings
export HMIntegrationFastSettings
export HMIntegrationStandardSettings
export HMIntegrationMultiThreadingSettings
export HMIntegrationSettings

export data_whitening

export IntegrationResult
export IntegrationVolume

end # module
