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
using StatsBase
using ElasticArrays
using StatsFuns
using DataFrames
using GLM

include("./DataTypes.jl")
include("./DataTree.jl")

include("./util.jl")

include("./PointCloud.jl")
include("./IntegrationVolume.jl")
#include("./ImportData.jl")


include("./Hyperrectangle.jl")
include("./WhiteningTransformation.jl")
include("./HarmonicMeanIntegration.jl")

include("./uncertainty.jl")


export hm_integrate
export hm_swapdata, hm_reset_tolerance, hm_reset_hyperrectangles

export DataSet
export WhiteningResult
export SearchTree, DataTree
export IntermediateResult

export HMIPrecisionSettings
export HMIFastSettings
export HMIStandardSettings
export HMIMultiThreadingSettings
export HMISettings

export data_whitening
export isinitialized

export HMIData
export HMIResult
export PointCloud
export IntegrationVolume

end # module
