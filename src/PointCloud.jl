# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


"""
    PointCloud(datatree::Tree, hyperrect::HyperRectVolume, searchpts::Bool = false)::PointCloud

creates a point cloud by searching the data tree for points which are inside the hyper-rectangle
The parameter searchpts determines if an array of the point IDs is created as well 
"""
function PointCloud(datatree::Tree, hyperrect::HyperRectVolume, searchpts::Bool = false)::PointCloud
    res = search(datatree, hyperrect, searchpts)

    points = res.points
    pointIDs = res.pointIDs

    maxProb = res.maxLogProb
    minProb = res.minLogProb

    maxwp = res.maxWeightProb
    minwp = res.minWeightProb

    probFactor = exp(maxProb - minProb)
    probwf = exp(maxwp - minwp)

    return PointCloud(maxProb, minProb, maxwp, minwp, probFactor, probwf, points, pointIDs)
end


function create_pointweights{T<:Real}(dataset::DataSet{T}, pointclouds::Vector{PointCloud})::Vector{T}
    pweights = zeros(dataset.N)

    for cloud in pointclouds
        for point in cloud.pointIDs
            pweights[point] += 1
        end
    end

    return pweights
end
