# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

__precompile__(true)

module MCMCHarmonicMean


using ProgressMeter
using Distributions
#using HDF5
using StatsBase

using BAT
using BAT.Logging
@enable_logging
set_log_level!(MCMCHarmonicMean, LOG_INFO)


using Base.Threads
using MultiThreadingTools

include("./util.jl")

include("./DataTypes.jl")
include("./DataTree.jl")

include("./PointCloud.jl")
include("./IntegrationVolume.jl")
include("./ImportData.jl")


include("./Hyperrectangle.jl")
include("./WhiteningTransformation.jl")
include("./HarmonicMeanIntegration.jl")


export hm_integrate
export hm_swapdata, hm_reset_tolerance, hm_reset_hyperrectangles

export DataSet
export WhiteningResult
export SearchTree, DataTree
export IntermediateResult

export HMIntegrationPrecisionSettings
export HMIntegrationFastSettings
export HMIntegrationStandardSettings
export HMIntegrationMultiThreadingSettings
export HMIntegrationSettings

export data_whitening

export HMIData
export PointCloud
export IntegrationVolume

end # module
