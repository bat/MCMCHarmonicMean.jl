# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


"""
    PointCloud(datatree::Tree, hyperrect::HyperRectVolume, searchpts::Bool = false)::PointCloud

creates a point cloud by searching the data tree for points which are inside the hyper-rectangle
The parameter searchpts determines if an array of the point IDs is created as well
"""
function PointCloud(datatree::Tree, hyperrect::HyperRectVolume, searchpts::Bool)::PointCloud
    result = PointCloud()

    PointCloud!(result, datatree, hyperrect, searchpts)
    return result
end

function PointCloud()::PointCloud
    return PointCloud(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, Array{Int64}(0))
end

function PointCloud!{T<:AbstractFloat, I<:Integer}(cloud::PointCloud{T, I}, datatree::Tree{T}, hyperrect::HyperRectVolume{T}, searchpts::Bool)

    res = search(datatree, hyperrect, searchpts)

    cloud.points = res.points

    resize!(cloud.pointIDs, cloud.points)
    copy!(cloud.pointIDs, res.pointIDs)

    cloud.maxWeightProb = res.maxWeightProb
    cloud.minWeightProb = res.minWeightProb
    cloud.maxLogProb = res.maxLogProb
    cloud.minLogProb = res.minLogProb

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
