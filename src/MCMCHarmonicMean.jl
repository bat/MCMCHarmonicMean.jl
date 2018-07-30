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
using DataStructures

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


export hm_init
export hm_whiteningtransformation
export hm_createpartitioningtree
export hm_findstartingsamples
export hm_determinetolerance
export hm_hyperrectanglecreation
export hm_integratehyperrectangles

export hm_integrate

export split_samples
export split_dataset

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
