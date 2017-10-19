# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).



"""
    DataSet{T<:Real}

Holds the MCMC output. For construction use constructor: function DataSet{T<:Real}(data::Matrix{T}, logprob::Vector{T}, weights::Vector{T})
# Variables
- 'data::Matrix{T}' : An P x N array with N data points with P parameters.
- 'logprob::Vector{T}' : The logarithmic probability for each data point
- 'weights::Vector{T}' : How often each sample occurred. Set to an array of ones if working directly on MCMC output
- 'N::Integer' : number of data points.
- 'P::Integer' : number of parameters.
"""
mutable struct DataSet{T<:Real}
    data::Matrix{T}
    logprob::Vector{T}
    weights::Vector{T}
    N::Integer
    P::Integer
end
function DataSet{T<:Real}(data::Matrix{T}, logprob::Vector{T}, weights::Vector{T})
    return DataSet(data, logprob, weights, size(data)[2], size(data)[1])
end

function Base.show(io::IO, data::DataSet)
    println("DataSet: $(data.N) samples, $(data.P) parameters")
end


struct HMIntegrationSettings
    whitening_method::Symbol
    max_startingIDs::Integer
    max_startingIDs_fraction::AbstractFloat
    rect_increase::AbstractFloat
    use_all_rects::Bool
    stop_ifenoughpoints::Bool
end
HMIntegrationFastSettings() =      return HMIntegrationSettings(:StatisticalWhitening, 100,   0.001, 0.1, false, true )
HMIntegrationStandardSettings() =  return HMIntegrationSettings(:StatisticalWhitening, 1000,  0.005, 0.1, false, false)
HMIntegrationPrecisionSettings() = return HMIntegrationSettings(:StatisticalWhitening, 10000, 0.025, 0.1, true,  false)

"""
    WhiteningResult{T<:Real}

Stores the information obtained during the Whitening Process
# Variables
- 'determinant::Float64' : The determinant of the whitening matrix
- 'boundingbox::Matrix{T}' : The bounding box for possible hyper-rectangle starting points
- 'logprobdiff::Float64' : The log. probability difference between the most probable and least probable sample
- 'targetprobfactor::Float64' : The suggested target probability factor
- 'whiteningmatrix::Matrix{T}' : The whitening matrix
- 'meanvalue::Vector{T}' : the mean vector of the input data
"""

struct WhiteningResult{T<:Real}
    determinant::Float64
    boundingbox::Matrix{T}
    logprobdiff::Float64
    targetprobfactor::Float64
    whiteningmatrix::Matrix{T}
    meanvalue::Vector{T}
end

function Base.show(io::IO, wres::WhiteningResult)
    println("Whitening Result: Determinant: $(wres.determinant), Target Prob. Factor: $(wres.targetprobfactor)")
end


"""
    SearchResult

Stores the results of the search tree's search function
# Variables
- 'pointIDs::Array{Int64, 1}' : the IDs of the points found, might be empty because it is optional
- 'points::Int64' : The number of points found.
- 'maxLogProb::Float64' : the maximum log. probability of the points found.
- 'minLogProb::Float64' : the minimum log. probability of the points found.
"""

mutable struct SearchResult
    pointIDs::Array{Int64, 1}
    points::Int64
    maxLogProb::Float64
    minLogProb::Float64
end

function Base.show(io::IO, sres::SearchResult)
    println("Search Result: Points: $(sres.points), Max. Log. Prob.: $(sres.maxLogProb), Min. Log. Prob.: $(sres.minLogProb)")
end


"""
    PointCloud

Stores the information of the points of an e.g. HyperRectVolume
# Variables
- 'maxLogProb::Float64' : The maximum log. probability of one of the points inside the hyper-rectangle
- 'minLogProb::Float64' : The minimum log. probability of one of the points inside the hyper-rectangle
- 'probfactor::Float64' : The probability factor of the hyper-rectangle
- 'points::Int64' : The number of points inside the hyper-rectangle
- 'pointIDs::Vector{Int64}' : the IDs of the points inside the hyper-rectangle, might be empty because it is optional and costs performance
"""

mutable struct PointCloud
    maxLogProb::Float64
    minLogProb::Float64
    probfactor::Float64
    points::Int64
    pointIDs::Vector{Int64}
end

function Base.show(io::IO, cloud::PointCloud)
    println("Point Cloud with $(cloud.points) points, probability factor: $(cloud.probfactor)")
end


"""
    IntegrationVolume

# Variables
- 'pointcloud::PointCloud' : holds the point cloud of the integration volume
- 'spatialvolume::SpatialVolume' : the boundaries of the integration volume
- 'volume::Float64' : the volume

Hold the point cloud and the spatial volume for integration.
"""

mutable struct IntegrationVolume
    pointcloud::PointCloud
    spatialvolume::SpatialVolume
    volume::Float64
end
function Base.show(io::IO, vol::IntegrationVolume)
    println("Hyperrectangle: $(vol.pointcloud.points) points, $(vol.volume) Volume")
end




"""
    IntegrationResult

Includes all the informations of the integration process, including a list of hyper-rectangles, the results of the whitening transformation,
the starting ids, and the average number of points and volume of the created hyper-rectangles.

# Variables
- 'integral::Float64' : The Harmonic Mean integration result
- 'error::Float64' : an error estimation. the quality of the error estimation depends on the number of hyper-rectangles created (the more the better)
- 'nvols::Int64' : the number of hyper-rectangles used for the integration
- 'points::Float64' : average number of points
- 'volume::Float64' : average volume
- 'volumelist::Vector{IntegrationVolume}' : a list of the hyper-rectangles
- 'startingIDs::Array{Int64, 1}' : a list of possible starting points for the hyper-rectangle creation process
- 'whiteningresult::WhiteningResult' : the results of the whitening transformation
"""
struct IntegrationResult
    integral::Float64
    error::Float64
    nvols::Int64
    points::Float64
    volume::Float64
    volumelist::Vector{IntegrationVolume}
    startingIDs::Vector{Int64}
    whiteningresult::WhiteningResult
end

function Base.show(io::IO, ires::IntegrationResult)
    println("Integration Result: $(ires.integral) +- $(ires.error), Rectangles: $(ires.nvols), average Points: $(ires.points), average Volume: $(ires.volume)")
end

struct IntermediateResult
    integral::Float64
    error::Float64
    points::Float64
    volume::Float64
end

function Base.show(io::IO, ires::IntermediateResult)
    println("Rectangle Integration Result: $(ires.integral) +- $(ires.error), average Points: $(ires.points), average Volume: $(ires.volume)")
end
