# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).



"""
    IntegrationVolume(datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)::IntegrationVolume

creates an integration region by calculating the point cloud an the volume of the spatial volume.
"""
function IntegrationVolume{T<:AbstractFloat, I<:Integer}(datatree::Tree{T, I}, spvol::HyperRectVolume{T}, searchpts::Bool = true)::IntegrationVolume{T, I}
    cloud = PointCloud(datatree, spvol, searchpts)
    vol = prod(spvol.hi - spvol.lo)

    return IntegrationVolume(cloud, spvol, vol)
end


"""
    IntegrationVolume!(intvol::IntegrationVolume, datatree::Tree, spvol::SpatialVolume, searchpts::Bool = true)

updates an integration volume with new boundaries. Recalculates the pointcloud and volume.
"""
function IntegrationVolume!{T<:AbstractFloat, I<:Integer}(intvol::IntegrationVolume{T, I}, datatree::Tree{T, I}, spvol::HyperRectVolume{T}, searchpts::Bool = true)
    if ndims(intvol.spatialvolume) != ndims(spvol)
        intvol.spatialvolume = deepcopy(spvol)
    else
        copy!(intvol.spatialvolume, spvol)
    end

    PointCloud!(intvol.pointcloud, datatree, spvol, searchpts)

    intvol.volume = prod(spvol.hi - spvol.lo)
end




function resize_integrationvol{T<:AbstractFloat, I<:Integer}(original::IntegrationVolume{T, I}, datatree::Tree{T, I},
        changed_dim::I, newrect::HyperRectVolume{T}, searchpts::Bool = false)::IntegrationVolume{T, I}
    result = deepcopy(original)
    return resize_integrationvol!(result, original, datatre, changed_dim, newrect, searchpts)
end
function resize_integrationvol!{T<:AbstractFloat, I<:Integer}(result::IntegrationVolume{T, I}, original::IntegrationVolume{T, I}, datatree::Tree{T, I},
        changed_dim::I, newrect::HyperRectVolume{T}, searchpts::Bool, searchVol::HyperRectVolume{T})
    P = datatree.P
    N = datatree.N

    copy!(searchVol, newrect)
    increase = true

    #increase
    if original.spatialvolume.lo[changed_dim] > newrect.lo[changed_dim]
        searchVol.hi[changed_dim] = original.spatialvolume.lo[changed_dim]
        searchVol.lo[changed_dim] = newrect.lo[changed_dim]
    elseif original.spatialvolume.hi[changed_dim] < newrect.hi[changed_dim]
        searchVol.lo[changed_dim] = original.spatialvolume.hi[changed_dim]
        searchVol.hi[changed_dim] = newrect.hi[changed_dim]
    else
        increase = false
        if original.spatialvolume.lo[changed_dim] < newrect.lo[changed_dim]
            searchVol.lo[changed_dim] = original.spatialvolume.lo[changed_dim]
            searchVol.hi[changed_dim] = newrect.lo[changed_dim]
        elseif original.spatialvolume.hi[changed_dim] > newrect.hi[changed_dim]
            searchVol.hi[changed_dim] = original.spatialvolume.hi[changed_dim]
            searchVol.lo[changed_dim] = newrect.hi[changed_dim]
        else
            error("resize_integrationvol(): Volume didn't change.")
        end
    end


    res = search(datatree, searchVol, searchpts)
    if increase

        result.pointcloud.points = original.pointcloud.points + res.points

        if searchpts
            resize!(result.pointcloud.pointIDs, result.pointcloud.points)
            copy!(result.pointcloud.pointIDs, original.pointcloud.pointIDs)
            copy!(result.pointcloud.pointIDs, original.pointcloud.points + 1, res.pointIDs, 1)
        end

        result.pointcloud.maxLogProb = max(original.pointcloud.maxLogProb, res.maxLogProb)
        result.pointcloud.minLogProb = max(original.pointcloud.minLogProb, res.minLogProb)
        result.pointcloud.maxWeightProb = max(original.pointcloud.maxWeightProb, res.maxWeightProb)
        result.pointcloud.minWeightProb = max(original.pointcloud.minWeightProb, res.minWeightProb)
    else
        result.pointcloud.points = original.pointcloud.points - res.points

        if searchpts
            newids = search(datatree, newrect, searchpts).pointIDs
            resize!(result.pointcloud.pointIDs, result.pointcloud.points)
            copy!(result.pointcloud.pointIDs, newids)
        end
    end

    result.volume = prod(newrect.hi - newrect.lo)
    result.pointcloud.probfactor = exp(result.pointcloud.maxLogProb - result.pointcloud.minLogProb)
    result.pointcloud.probweightfactor = exp(result.pointcloud.maxWeightProb - result.pointcloud.minWeightProb)
    copy!(result.spatialvolume, newrect)
end

function Base.copy!{T<:AbstractFloat, I<:Integer}(target::IntegrationVolume{T, I}, src::IntegrationVolume{T, I})
    target.volume = src.volume

    copy!(target.spatialvolume, src.spatialvolume)
    target.pointcloud.points = src.pointcloud.points

    copy!(target.pointcloud, src.pointcloud)
end

#remove as soon as BAT2 has a copy! function
function Base.copy!{T<:AbstractFloat}(target::HyperRectVolume{T}, src::HyperRectVolume{T})
    p = ndims(src)
    resize!(target.lo, p)
    copy!(target.lo, src.lo)
    resize!(src.hi, p)
    copy!(target.hi, src.hi)
end
