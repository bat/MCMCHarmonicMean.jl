# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

abstract type SearchTree end

"""
    DataSet{T<:AbstractFloat, I<:Integer}

Holds the MCMC output. For construction use constructor: function DataSet{T<:Real}(data::Matrix{T}, logprob::Vector{T}, weights::Vector{T})
# Variables
- 'data::Matrix{T}' : An P x N array with N data points with P parameters.
- 'logprob::Vector{T}' : The logarithmic probability for each data point
- 'weights::Vector{T}' : How often each sample occurred. Set to an array of ones if working directly on MCMC output
- 'N::I' : number of data points.
- 'P::I' : number of parameters.
"""
mutable struct DataSet{T<:AbstractFloat, I<:Integer}
    data::Array{T, 2}
    logprob::Array{T, 1}
    weights::Array{T, 1}
    ids::Array{I, 1}    #used to divide the dataset into sub-sets
    N::I
    P::I
    nsubsets::I    #number of sub-sets
    subsetsize::T
    iswhitened::Bool
    isnew::Bool
    partitioningtree::SearchTree
    startingIDs::Array{I, 1}
    tolerance::T
end

function DataSet(
    data::Tuple{DensitySampleVector{T, T, I, ElasticArray{T, 2,1}, Array{T, 1},Array{I ,1}}, MCMCSampleIDVector, MCMCBasicStats},
    nsubsets::Int64 = 0,
    subsetsize::T = zero(T)
    )::DataSet{T, I} where {T<:AbstractFloat, I<:Integer}

    DataSet(data..., nsubsets, subsetsize)
end

function DataSet(
    samples::DensitySampleVector{T, T, I, ElasticArray{T, 2,1}, Array{T, 1},Array{I ,1}},
    sampleIDs::MCMCSampleIDVector,
    mcmcstats::MCMCBasicStats,
    nsubsets::Int64 = 0,
    subsetsize::T = zero(T)
    )::DataSet{T, I} where {T<:AbstractFloat, I<:Integer}

    DataSet(convert(Array{T, 2}, samples.params), samples.log_value, samples.weight, nsubsets, subsetsize)
end

function DataSet(
    data:: Array{T, 2},
    logprob::Array{T, 1},
    weights::Array{I, 1},
    nsubsets::Int64 = 0,
    subsetsize::T = zero(T)
    )::DataSet{T, Int64} where {T<:AbstractFloat, I<:Integer}

    DataSet(data, logprob, convert(Array{T, 1}, weights), nsubsets, subsetsize)
end

function DataSet(
    data:: Array{T, 2},
    logprob::Array{T, 1},
    weights::Array{T, 1},
    nsubsets::Int64 = 0,
    subsetsize::T = zero(T)
    )::DataSet{T, Int64} where {T<:AbstractFloat}

    P, N = size(data)

    if iszero(nsubsets)
        nsubsets = 10
    end

    maxbatchsize = sum(weights) / 10 / nsubsets
    if iszero(subsetsize)
        subsetsize = 100.0
    end
    subsetsize = min(maxbatchsize, subsetsize)

    ids = zeros(Int64, N)
    cnt = 1

    batch_currentsize = 0.0

    for i = 1:N
        ids[i] = cnt
        batch_currentsize += weights[i]

        if batch_currentsize >= subsetsize
            cnt += 1
            if cnt > nsubsets
                cnt = 1
            end
            batch_currentsize = 0.0
        end
    end

    DataSet(data, logprob, weights, ids, N, P, nsubsets, subsetsize, false, true, DataTree(T, Int64), Array{Int64, 1}(0), T(0))
end

Base.show(io::IO, data::DataSet) = print(io, "DataSet: $(data.N) samples, $(data.P) parameters")
Base.eltype(data::DataSet{T, I}) where {T<:AbstractFloat, I<:Integer} = (T, I)

"""
    HMISettings

holds the settings for the hm_integrate function. There are several default constructors available:
HMIFastSettings()
HMIStandardSettings()
HMIPrecisionSettings()

#Variables
- 'whitening_method::Symbol' : which whitening method to use
- 'max_startingIDs::Integer' : influences how many starting ids are allowed to be generated
- 'max_startingIDs_fraction::AbstractFloat' : how many points are considered as possible starting points as a fraction of total points available
- 'rect_increase::AbstractFloat' : describes the procentual rectangle volume increase/decrease during hyperrectangle creation. Low values can increase the precision if enough points are available but can cause systematically wrong results if not enough points are available.
- 'use_all_rects::Bool' : All rectangles are used for the integration process no matter how big their overlap is. If enabled the rectangles are weighted by their overlap.
- 'useMultiThreading' : activate multithreading support.
end
"""
mutable struct HMISettings
    whitening_function::Function
    max_startingIDs::Integer
    max_startingIDs_fraction::AbstractFloat
    rect_increase::AbstractFloat
    useMultiThreading::Bool
    warning_minstartingids::Integer
end
HMIFastSettings() =      return HMISettings(cholesky_whitening, 100,   0.1, 0.1, true, 16)
HMIStandardSettings() =  return HMISettings(cholesky_whitening, 1000,  0.5, 0.1, true, 16)
HMIPrecisionSettings() = return HMISettings(cholesky_whitening, 10000, 2.5, 0.1, true, 16)

"""
    WhiteningResult{T<:AbstractFloat}

Stores the information obtained during the Whitening Process
# Variables
- 'determinant::T' : The determinant of the whitening matrix
- 'targetprobfactor::T' : The suggested target probability factor
- 'whiteningmatrix::Matrix{T}' : The whitening matrix
- 'meanvalue::Vector{T}' : the mean vector of the input data
"""
struct WhiteningResult{T<:AbstractFloat}
    determinant::T
    targetprobfactor::T
    whiteningmatrix::Matrix{T}
    meanvalue::Vector{T}
end
WhiteningResult(T::DataType) = WhiteningResult(zero(T), zero(T), Matrix{T}(0, 0), Vector{T}(0))
Base.show(io::IO, wres::WhiteningResult) = print(io, "Whitening Result: Determinant: $(wres.determinant), Target Prob. Factor: $(wres.targetprobfactor)")
isinitialized(x::WhiteningResult) = !(iszero(x.determinant) && iszero(x.targetprobfactor) && isempty(x.whiteningmatrix) && isempty(x.meanvalue))

"""
    SearchResult{T<:AbstractFloat, I<:Integer}

Stores the results of the search tree's search function
# Variables
- 'pointIDs::Vector{I}' : the IDs of the points found, might be empty because it is optional
- 'points::I' : The number of points found.
- 'maxLogProb::T' : the maximum log. probability of the points found.
- 'minLogProb::T' : the minimum log. probability of the points found.
- 'maxWeightProb::T' : the weighted minimum log. probability found.
- 'minWeightProb::T' : the weighted maximum log. probfactor found.
"""
mutable struct SearchResult{T<:AbstractFloat, I<:Integer}
    pointIDs::Vector{I}
    points::I
    maxLogProb::T
    minLogProb::T
    maxWeightProb::T
    minWeightProb::T
end

function SearchResult(T::DataType, I::DataType)
    @assert T<:AbstractFloat
    @assert I<:Integer
    return SearchResult{T, I}(Vector{I}(0), I(0), T(0), T(0), T(0), T(0))
end
Base.show(io::IO, sres::SearchResult) = print(io, "Search Result: Points: $(sres.points), Max. Log. Prob.: $(sres.maxLogProb), Min. Log. Prob.: $(sres.minLogProb)")


"""
    PointCloud{T<:AbstractFloat, I<:Integer}

Stores the information of the points of an e.g. HyperRectVolume
# Variables
- 'maxLogProb::T' : The maximum log. probability of one of the points inside the hyper-rectangle
- 'minLogProb::T' : The minimum log. probability of one of the points inside the hyper-rectangle
- 'maxWeightProb::T' : the weighted max. log. probability
- 'minWeightProb::T' : the weighted min. log. probability
- 'probfactor::T' : The probability factor of the hyper-rectangle
- 'probweightfactor::T' : The weighted probability factor
- 'points::I' : The number of points inside the hyper-rectangle
- 'pointIDs::Vector{I}' : the IDs of the points inside the hyper-rectangle, might be empty because it is optional and costs performance
"""
mutable struct PointCloud{T<:AbstractFloat, I<:Integer}
    maxLogProb::T
    minLogProb::T
    maxWeightProb::T
    minWeightProb::T
    probfactor::T
    probweightfactor::T
    points::I
    pointIDs::Vector{I}
    searchres::SearchResult{T, I}
end
PointCloud(T::DataType, I::DataType) = PointCloud(T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), I(0), Vector{I}(0), SearchResult(T, I))
Base.show(io::IO, cloud::PointCloud) = print(io, "Point Cloud with $(cloud.points) points, probability factor: $(cloud.probfactor)")


"""
    IntegrationVolume{T<:AbstractFloat, I<:Integer}

# Variables
- 'pointcloud::PointCloud{T, I}' : holds the point cloud of the integration volume
- 'spatialvolume::SpatialVolume{T}' : the boundaries of the integration volume
- 'volume::T' : the volume

Hold the point cloud and the spatial volume for integration.
"""
mutable struct IntegrationVolume{T<:AbstractFloat, I<:Integer}
    pointcloud::PointCloud{T, I}
    spatialvolume::HyperRectVolume{T}
    volume::T
end
Base.show(io::IO, vol::IntegrationVolume) = print(io, "Hyperrectangle: $(vol.pointcloud.points) points, $(vol.volume) Volume")



mutable struct IntermediateResults{T<:AbstractFloat}
    integrals::Array{T, 1}
    weights_overlap::Array{T, 1}
    weights_cov::Array{T, 1}
    volumeID::Array{Int64, 1}
    overlap::Array{T, 2}
    Y::Array{T, 2}
    Σ::Array{T, 2}
end
IntermediateResults(T::DataType, n::Int64) = IntermediateResults(zeros(T, n), zeros(T, n), zeros(T, n), [Int64(i) for i=1:n], Array{T, 2}(0,0), Array{T, 2}(0,0), Array{T,2}(0,0))
Base.length(x::IntermediateResults) = length(x.integrals)

mutable struct HMIEstimate{T<:AbstractFloat}
    estimate::T
    uncertainty::T
    weights::Array{T, 1}
end
HMIEstimate(T::DataType) = HMIEstimate(zero(T), zero(T), Array{T, 1}(0))
function HMIEstimate(a::HMIEstimate{T}, b::HMIEstimate{T})::HMIEstimate{T} where {T<:AbstractFloat}
    val = mean([a.estimate, b.estimate], AnalyticWeights([1 / a.uncertainty^2, 1 / b.uncertainty^2]))
    unc = 1 / sqrt(1 / a.uncertainty^2 + 1 / b.uncertainty^2)
    HMIEstimate(val, unc, [a.weights..., b.weights...])
end
Base.show(io::IO, ires::HMIEstimate) = print(io, "$(signif(ires.estimate, 6))\t+-\t$(signif(ires.uncertainty, 6))")

"""
    HMIData{T<:AbstractFloat, I<:Integer}

Includes all the informations of the integration process, including a list of hyper-rectangles, the results of the whitening transformation,
the starting ids, and the average number of points and volume of the created hyper-rectangles.

# Variables
- 'integral::T' : The Harmonic Mean integration result
- 'error::T' : an error estimation. the quality of the error estimation depends on the number of hyper-rectangles created (the more the better)
- 'nvols::I' : the number of hyper-rectangles used for the integration
- 'points::T' : average number of points
- 'volume::T' : average volume
- 'volumelist::Vector{IntegrationVolume{T, I}}' : a list of the hyper-rectangles
- 'startingIDs::Vector{I}' : a list of possible starting points for the hyper-rectangle creation process
- 'whiteningresult::WhiteningResult{T}' : the results of the whitening transformation
- 'integrals::Vector{T}' : The integral estimates of the different hyper-rectangles
"""
mutable struct HMIData{T<:AbstractFloat, I<:Integer}
    dataset1::DataSet{T, I}
    dataset2::DataSet{T, I}
    whiteningresult::WhiteningResult{T}
    volumelist1::Vector{IntegrationVolume{T, I}}
    volumelist2::Vector{IntegrationVolume{T, I}}
    cubelist1::Vector{HyperRectVolume{T}}
    cubelist2::Vector{HyperRectVolume{T}}
    rejectedrects1::Vector{I}
    rejectedrects2::Vector{I}
    integrals1::IntermediateResults{T}
    integrals2::IntermediateResults{T}
    integral_standard::HMIEstimate{T}
    integral_covweighted::HMIEstimate{T}
end

function HMIData(
    dataset1::DataSet{T, I},
    dataset2::DataSet{T, I})::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

    HMIData(
        dataset1,
        dataset2,
        WhiteningResult(T),
        Vector{IntegrationVolume{T, I}}(0),
        Vector{IntegrationVolume{T, I}}(0),
        Vector{HyperRectVolume{T}}(0),
        Vector{HyperRectVolume{T}}(0),
        Vector{I}(0),
        Vector{I}(0),
        IntermediateResults(T, 0),
        IntermediateResults(T, 0),
        HMIEstimate(T),
        HMIEstimate(T)
    )
end

function HMIData(dataset::DataSet{T, I})::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}
    HMIData(split_dataset(dataset)...)
end

function HMIData(
    data::Tuple{DensitySampleVector{T, T, I, ElasticArray{T, 2,1}, Array{T, 1},Array{I ,1}}, MCMCSampleIDVector, MCMCBasicStats})::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

    HMIData(data...)
end

function HMIData(
    samples::DensitySampleVector{T, T, I, ElasticArray{T, 2,1}, Array{T, 1},Array{I ,1}},
    sampleIDs::MCMCSampleIDVector,
    mcmcstats::MCMCBasicStats)::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

    HMIData(split_samples(samples, sampleIDs, mcmcstats)...)
end

Base.show(io::IO, ires::HMIData) = print(io, "Data Set 1: $(length(ires.volumelist1)) Volumes, Data Set 2: $(length(ires.volumelist2))
\tStandard integration result:\t $(ires.integral_standard)
\tCov. weighted result:\t\t $(ires.integral_covweighted)")
