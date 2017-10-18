# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

function IntegrationVolume(datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)::IntegrationVolume
    cloud = PointCloud(datatree, spvol, searchpts)
    vol = _getvolume(spvol)

    return IntegrationVolume(cloud, spvol, vol)
end

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


#only works correct if the resized dim gets changed and nothing else
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

    minProb = 0.0
    maxProb = 0.0
    local pointIDs::Vector{Int64}
    points = 0

    res = search(datatree, searchVol, searchpts)

    if increase
        maxProb = intvol.pointcloud.maxLogProb
        minProb = intvol.pointcloud.minLogProb

        points = intvol.pointcloud.points + res.points

        if searchpts
            pointIDs = deepcopy(intvol.pointcloud.pointIDs)
            append!(pointIDs, res.pointIDs)
        else
            pointIDs = Vector{Int64}(0)
        end

        if res.maxLogProb > maxProb
            maxProb = res.maxLogProb
        end
        if res.minLogProb < minProb
            minProb = res.minLogProb
        end
    else
        points = intvol.pointcloud.points - res.points

        if searchpts
            res = search(datatree, newrect, searchpts)
            maxProb = res.maxLogProb
            minProb = res.minLogProb
            pointIDs = res.pointIDs
        else
            maxProb = intvol.pointcloud.maxLogProb
            minProb = intvol.pointcloud.minLogProb
            pointIDs = Array{Int64, 1}(0)
        end
    end

    volume = prod(newrect.hi - newrect.lo)

    probFactor = exp(maxProb - minProb)
    pointcloud = PointCloud(maxProb, minProb, probFactor, points, pointIDs)
    return IntegrationVolume(pointcloud, deepcopy(newrect), volume)
end
