# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).



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
    data::Matrix{T}
    logprob::Vector{T}
    weights::Vector{T}
    N::I
    P::I
end

function DataSet(data::Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats})
    return DataSet(data[1])
end
function DataSet(data::BAT.DensitySampleVector)
    T = typeof(data.params[1,1])
    return DataSet(
        convert(Matrix{T}, data.params),
        convert(Vector{T}, data.log_value),
        convert(Vector{T}, data.weight))
end
function DataSet{T<:AbstractFloat}(data::Matrix{T}, logprob::Vector{T}, weights::Vector{T})
    return DataSet{T, Int64}(data, logprob, weights, size(data)[2], size(data)[1])
end

function Base.show(io::IO, data::DataSet)
    print(io, "DataSet: $(data.N) samples, $(data.P) parameters")
end


"""
    HMIntegrationSettings

holds the settings for the hm_integrate function. There are several default constructors available:
HMIntegrationFastSettings()
HMIntegrationStandardSettings()
HMIntegrationPrecisionSettings()

#Variables
- 'whitening_method::Symbol' : which whitening method to use
- 'max_startingIDs::Integer' : influences how many starting ids are allowed to be generated
- 'max_startingIDs_fraction::AbstractFloat' : how many points are considered as possible starting points as a fraction of total points available
- 'rect_increase::AbstractFloat' : describes the procentual rectangle volume increase/decrease during hyperrectangle creation. Low values can increase the precision if enough points are available but can cause systematically wrong results if not enough points are available.
- 'use_all_rects::Bool' : All rectangles are used for the integration process no matter how big their overlap is. If enabled the rectangles are weighted by their overlap.
- 'stop_ifenoughpoints::Bool' : if the hyper-rectangles created already contain enough points than no further rectangles are created. Increases the performance for the integration of easy target densities. Might decrease the reliability of the error estimation.
- 'useMultiThreading' : activate experimental multithreading support. need use_all_rects enabled and stop_ifenoughpoints disabled.
end

"""
mutable struct HMIntegrationSettings
    whitening_method::Symbol
    max_startingIDs::Integer
    max_startingIDs_fraction::AbstractFloat
    rect_increase::AbstractFloat
    stop_ifenoughpoints::Bool
    skip_centerIDsinsideHyperRects::Bool
    useMultiThreading::Bool
end
HMIntegrationFastSettings() =      return HMIntegrationSettings(:StatisticalWhitening, 100,   0.1, 0.1, true, true, true)
HMIntegrationStandardSettings() =  return HMIntegrationSettings(:StatisticalWhitening, 1000,  0.5, 0.1, false, false, true)
HMIntegrationPrecisionSettings() = return HMIntegrationSettings(:StatisticalWhitening, 10000, 2.5, 0.1, false, false, true)

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

function Base.show(io::IO, wres::WhiteningResult)
    print(io, "Whitening Result: Determinant: $(wres.determinant), Target Prob. Factor: $(wres.targetprobfactor)")
end


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

function Base.show(io::IO, sres::SearchResult)
    print(io, "Search Result: Points: $(sres.points), Max. Log. Prob.: $(sres.maxLogProb), Min. Log. Prob.: $(sres.minLogProb)")
end


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
end

function Base.show(io::IO, cloud::PointCloud)
    print(io, "Point Cloud with $(cloud.points) points, probability factor: $(cloud.probfactor)")
end


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
function Base.show(io::IO, vol::IntegrationVolume)
    print(io, "Hyperrectangle: $(vol.pointcloud.points) points, $(vol.volume) Volume")
end




"""
    IntegrationResult{T<:AbstractFloat, I<:Integer}

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
struct IntegrationResult{T<:AbstractFloat, I<:Integer}
    integral::T
    error::T
    nvols::I
    points::T
    volume::T
    volumelist::Vector{IntegrationVolume{T, I}}
    startingIDs::Vector{I}
    tolerance::T
    whiteningresult::WhiteningResult{T}
    integrals::Vector{T}
end

function Base.show(io::IO, ires::IntegrationResult)
    print(io, "Integration Result: $(ires.integral) +- $(ires.error), Rectangles: $(ires.nvols), average Points: $(ires.points), average Volume: $(ires.volume)")
end

struct IntermediateResult{T<:AbstractFloat}
    integral::T
    error::T
    points::T
    volume::T
end

function Base.show(io::IO, ires::IntermediateResult)
    print(io, "Rectangle Integration Result: $(ires.integral) +- $(ires.error), average Points: $(ires.points), average Volume: $(ires.volume)")
end
