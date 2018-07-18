# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


"""
    PointCloud{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, hyperrect::HyperRectVolume{T}, searchpts::Bool = false)::PointCloud

creates a point cloud by searching the data tree for points which are inside the hyper-rectangle
The parameter searchpts determines if an array of the point IDs is created as well
"""
function PointCloud{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, hyperrect::HyperRectVolume{T}, searchpts::Bool)::PointCloud
    result = PointCloud(T, I)

    PointCloud!(result, dataset, hyperrect, searchpts)
    return result
end


function PointCloud!{T<:AbstractFloat, I<:Integer}(cloud::PointCloud{T, I}, dataset::DataSet{T, I}, hyperrect::HyperRectVolume{T}, searchpts::Bool)

    search!(cloud.searchres, dataset, hyperrect, searchpts)

    cloud.points = cloud.searchres.points

    resize!(cloud.pointIDs, cloud.points)
    copy!(cloud.pointIDs, cloud.searchres.pointIDs)

    cloud.maxWeightProb = cloud.searchres.maxWeightProb
    cloud.minWeightProb = cloud.searchres.minWeightProb
    cloud.maxLogProb = cloud.searchres.maxLogProb
    cloud.minLogProb = cloud.searchres.minLogProb

    cloud.probfactor = exp(cloud.maxLogProb - cloud.minLogProb)
    cloud.probweightfactor = exp(cloud.maxWeightProb - cloud.minWeightProb)
end

function create_pointweights{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, volumes::Vector{IntegrationVolume{T, I}})::Vector{T}
    pweights = zeros(T, dataset.N)

    for i in eachindex(volumes)
        for p in eachindex(volumes[i].pointcloud.pointIDs)
            pweights[volumes[i].pointcloud.pointIDs[p]] += 1
        end
    end

    return pweights
end

function Base.copy!{T<:AbstractFloat, I<:Integer}(target::PointCloud{T, I}, src::PointCloud{T, I})
    target.points = src.points

    resize!(target.pointIDs, length(src.pointIDs))
    copy!(target.pointIDs, src.pointIDs)

    target.maxLogProb = src.maxLogProb
    target.minLogProb = src.minLogProb

    target.maxWeightProb = src.maxWeightProb
    target.minWeightProb = src.minWeightProb

    target.probfactor = src.probfactor
    target.probweightfactor = src.probweightfactor
end
