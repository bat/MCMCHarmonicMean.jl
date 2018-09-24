abstract type SpatialVolume{T<:Real} end


struct HyperRectVolume{T<:Real} <: SpatialVolume{T}
    lo::Vector{T}
    hi::Vector{T}

    function HyperRectVolume{T}(lo::Vector{T}, hi::Vector{T}) where {T<:Real}
        (axes(lo) != axes(hi)) && throw(ArgumentError("lo and hi must have the same indices"))
        new{T}(lo, hi)
    end
end

HyperRectVolume(lo::Vector{T}, hi::Vector{T}) where {T<:Real} = HyperRectVolume{T}(lo, hi)
Base.ndims(vol::HyperRectVolume) = size(vol.lo, 1)

struct HyperSphereVolume{T<:Real} <: SpatialVolume{T}
    origin::Vector{T}
    radius::T

    function HyperRectVolume{T}(origin::Vector{T}, radius::T) where {T<:Real}
        new{T}(origin, radius)
    end
end

HyperSphereVolume(origin::Vector{T}, radius::T) where {T<:Real} = HyperRectVolume{T}(origin, radius)
Base.ndims(vol::HyperSphereVolume) = length(origin)



#=

function split_samples(
    samples::DensitySampleVector{T, T, I, ElasticArray{T, 2,1}, Array{T, 1},Array{I ,1}},
    sampleIDs::MCMCSampleIDVector,
    mcmcstats::MCMCBasicStats
)::Tuple{DataSet{T, I}, DataSet{T, I}} where {T<:Real, I<:Integer}

    N = length(sampleIDs)
    n = floor(Int64, N / 2)

    ds1 = DataSet(samples.params[:, 1:n], samples.log_value[1:n], samples.weight[1:n])
    ds2 = DataSet(samples.params[:, n+1:N], samples.log_value[n+1:N], samples.weight[n+1:N])

    return ds1, ds2
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
=#
