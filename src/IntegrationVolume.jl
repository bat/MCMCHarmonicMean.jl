# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).



"""
    IntegrationVolume(datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)::IntegrationVolume

creates an integration region by calculating the point cloud an the volume of the spatial volume.
"""
function IntegrationVolume(datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)::IntegrationVolume
    cloud = PointCloud(datatree, spvol, searchpts)
    vol = _getvolume(spvol)

    return IntegrationVolume(cloud, spvol, vol)
end


"""
    IntegrationVolume!(intvol::IntegrationVolume, datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)

updates an integration volume with new boundaries. Recalculates the pointcloud and volume.
"""
function IntegrationVolume!(intvol::IntegrationVolume, datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)
    cloud = PointCloud(datatree, spvol, searchpts)
    vol = _getvolume(spvol)

    intvol.spatialvolume = spvol
    intvol.pointcloud = cloud
    intvol.volume = vol
end

@inline function _getvolume(spvol::SpatialVolume)
    vol = 1.0
    for i = 1:ndims(spvol)
        vol *= spvol.hi[i] - spvol.lo[i]
    end
    return vol
end


"""
     resize_integrationvol(datatree::Tree, changed_dim::Int64, intvol::IntegrationVolume, newrect::HyperRectVolume, searchpts::Bool = false)::IntegrationVolume

resizes an integration volume along only one(!!) dimension, faster than recalcuating the integration volume with IntegrationVolume!
"""
#TODO provide targetIntegrationVolume to reduce memory consumption
function resize_integrationvol(datatree::Tree, changed_dim::Int64, intvol::IntegrationVolume, newrect::HyperRectVolume, searchpts::Bool = false)::IntegrationVolume
    P = datatree.P
    N = datatree.N

    searchVol = deepcopy(newrect)
    increase = true
    oldrect = intvol.spatialvolume

    #increase
    if oldrect.lo[changed_dim] > newrect.lo[changed_dim]
        searchVol.hi[changed_dim] = oldrect.lo[changed_dim]
        searchVol.lo[changed_dim] = newrect.lo[changed_dim]
    elseif oldrect.hi[changed_dim] < newrect.hi[changed_dim]
        searchVol.lo[changed_dim] = oldrect.hi[changed_dim]
        searchVol.hi[changed_dim] = newrect.hi[changed_dim]
    else
        increase = false
        if oldrect.lo[changed_dim] < newrect.lo[changed_dim]
            searchVol.lo[changed_dim] = oldrect.lo[changed_dim]
            searchVol.hi[changed_dim] = newrect.lo[changed_dim]
        elseif oldrect.hi[changed_dim] > newrect.hi[changed_dim]
            searchVol.hi[changed_dim] = oldrect.hi[changed_dim]
            searchVol.lo[changed_dim] = newrect.hi[changed_dim]
        else
            error("resize_integrationvol(): Volume didn't change.")
        end
    end

    minProb = Inf
    maxProb = -Inf
    maxwp = Inf
    minwp = Inf
    local pointIDs::Vector{Int64}
    points = 0

    res = search(datatree, searchVol, searchpts)

    if increase
        maxProb = intvol.pointcloud.maxLogProb
        minProb = intvol.pointcloud.minLogProb
        maxwp = intvol.pointcloud.maxWeightProb
        minwp = intvol.pointcloud.minWeightProb

        points = intvol.pointcloud.points + res.points

        if searchpts
            pointIDs = deepcopy(intvol.pointcloud.pointIDs)
            append!(pointIDs, res.pointIDs)
        else
            pointIDs = Vector{Int64}(0)
        end

        maxProb = res.maxLogProb > maxProb ? res.maxLogProb : maxProb
        minProb = res.minLogProb < minProb ? res.minLogProb : minProb
        maxwp = res.maxWeightProb > maxwp ? res.maxWeightProb : maxwp
        minwp = res.minWeightProb < minwp ? res.minWeightProb : minwp
    else
        points = intvol.pointcloud.points - res.points

        if searchpts
            res = search(datatree, newrect, searchpts)
            maxProb = res.maxLogProb
            minProb = res.minLogProb
            maxwp = res.maxWeightProb
            minwp = res.minWeightProb
            pointIDs = res.pointIDs
        else
            maxProb = intvol.pointcloud.maxLogProb
            minProb = intvol.pointcloud.minLogProb
            maxwp = intvol.pointcloud.maxWeightProb
            minwp = intvol.pointcloud.minWeightProb
            pointIDs = Array{Int64, 1}(0)
        end
    end

    volume = prod(newrect.hi - newrect.lo)

    probFactor = exp(maxProb - minProb)
    probwf = exp(maxwp - minwp)
    pointcloud = PointCloud(maxProb, minProb, maxwp, minwp, probFactor, probwf, points, pointIDs)
    return IntegrationVolume(pointcloud, deepcopy(newrect), volume)
end
