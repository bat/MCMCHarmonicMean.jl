# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

__precompile__(true)

module MCMCHarmonicMean


using ProgressMeter
using Distributions
using StatsBase
using LinearAlgebra
using ParallelProcessingTools
using Base.Threads
using ElasticArrays
using DataStructures
using RecipesBase
using LaTeXStrings
using Base.CoreLogging

include("spatial_volume.jl")


include("data_types.jl")
include("data_tree.jl")

include("util.jl")

include("point_cloud.jl")
include("integration_volume.jl")


include("hyper_rectangle.jl")
include("whitening_transformation.jl")
include("harmonic_mean_integration.jl")
include("hm_integration_rectangle.jl")

include("uncertainty.jl")
include("plot_recipes.jl")

export hm_init
export hm_whiteningtransformation!
export hm_createpartitioningtree!
export hm_findstartingsamples!
export hm_determinetolerance!
export hm_hyperrectanglecreation!
export hm_integratehyperrectangles!

export hm_integrate!

export split_samples
export split_dataset

export DataSet
export WhiteningResult
export SpacePartitioningTree
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
