# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


"""
    HyperCubeVolume{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume

creates a hypercube shaped spatial volume
"""
function HyperCubeVolume{T<:AbstractFloat}(origin::Vector{T}, edgelength::T)::HyperRectVolume{T}
    dim = length(origin)
    lo = Vector{T}(dim)
    hi = Vector{T}(dim)

    _setcubeboundaries!(lo, hi, origin, edgelength)

    return HyperRectVolume(lo, hi)
end

"""
    HyperCubeVolume{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume

resizes a hypercube shaped spatial volume
"""
function HyperCubeVolume!{T<:AbstractFloat}(rect::HyperRectVolume{T}, neworigin::Vector{T}, newedgelength::T)
    _setcubeboundaries!(rect.lo, rect.hi, neworigin, newedgelength)
end

@inline function _setcubeboundaries!{T<:AbstractFloat}(lo::Vector{T}, hi::Vector{T}, origin::Vector{T}, edgelength::T)
    for i = 1:length(lo)
        lo[i] = origin[i] - edgelength * 0.5
        hi[i] = origin[i] + edgelength * 0.5
    end
end

"""
    find_hypercube_centers(dataset::DataSet, datatree::Tree, whiteningresult::WhiteningResult, settings::HMIntegrationSettings)::Vector{I}

finds possible starting points for the hyperrectangle creation
"""
function find_hypercube_centers{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, datatree::Tree{T, I},
        whiteningresult::WhiteningResult{T}, settings::HMIntegrationSettings)::Vector{I}
    weights = [T(-Inf) for i=1:dataset.N]

    sortLogProb = sortperm(dataset.logprob, rev = true)

    NMax = ceil(I, min(min(dataset.N, settings.max_startingIDs * 10), dataset.N * min(1.0, 10 * settings.max_startingIDs_fraction)))

    ignorePoint = falses(dataset.N)

    testlength = find_density_test_cube_edgelength(dataset.data[:, sortLogProb[1]], datatree, max(round(I, dataset.N * 0.001), dataset.P * 10))

    @showprogress for n::I in sortLogProb[1:NMax]
        if ignorePoint[n]
            continue
        end

        mode = view(dataset.data, :, n)

        weights[n] = dataset.logprob[n]

        cubevol =  HyperCubeVolume(dataset.data[:, n], testlength)
        cube = IntegrationVolume(datatree, cubevol, true)
        for id::I in cube.pointcloud.pointIDs
            ignorePoint[id] = true
        end
    end

    sortIdx = sortperm(weights, rev = true)

    stop::T = 1.0
    for i::I = 1:dataset.N
        if weights[sortIdx[i]] == -Inf
            stop = i
            break
        end
    end
    NMax::I = stop - 1

    max_startingIDs::I = min(settings.max_startingIDs, round(I, dataset.N * settings.max_startingIDs_fraction))
    if stop > max_startingIDs
        NMax = max_startingIDs
    end
    stop = dataset.logprob[sortIdx[1]] - log(whiteningresult.targetprobfactor)
    for i = 1:NMax
        if dataset.logprob[sortIdx[i]] < stop
            NMax = i
            break
        end
    end

    #return at least 10 possible hyper-rect centers
    if NMax < 10 && stop >= 10
        NMax = 10
    elseif NMax < 10 && stop < 10
        LogMedium("Returned minimum number of starting points: 10")
        return sortLogProb[1:10]
    end

    LogMedium("Possible Hypersphere Centers: $NMax out of $(dataset.N) points")

    return sortIdx[1:NMax]
end

function find_density_test_cube_edgelength{T<:AbstractFloat, I<:Integer}(mode::Vector{T}, datatree::Tree{T, I}, points::I = 100)
    return find_density_test_cube(mode, datatree, points)[1]
end

function find_density_test_cube{T<:AbstractFloat, I<:Integer}(mode::Vector{T}, datatree::Tree{T, I}, points::I)
    P = datatree.P

    l::T = 1.0
    tol::T = 1.0
    mult::T = 1.2^(1.0 / P)

    rect = HyperCubeVolume(mode, l)
    intvol = IntegrationVolume(datatree, rect, false)
    pt = intvol.pointcloud.points

    while pt < points / tol || pt > points * tol
        tol += 0.001
        if pt > points
            l /= mult
        else
            l *= mult
        end
        HyperCubeVolume!(rect, mode, l)
        IntegrationVolume!(intvol, datatree, rect, false)
        pt = intvol.pointcloud.points
    end

    return l, intvol
end
