# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).



function PointCloud(datatree::Tree, hyperrect::HyperRectVolume, searchpts::Bool = false)::PointCloud
    res = search(datatree, hyperrect, searchpts)

    points = res.points
    pointIDs = res.pointIDs

    maxProb = res.maxLogProb
    minProb = res.minLogProb

    probFactor = exp(maxProb - minProb)

    return PointCloud(maxProb, minProb, probFactor, points, pointIDs)
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
