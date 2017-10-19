# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


function HyperCubeVolume{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume
    dim = length(origin)
    lo = Vector{T}(dim)
    hi = Vector{T}(dim)

    _setcubeboundaries!(lo, hi, origin, edgelength)

    return HyperRectVolume(lo, hi)
end

function HyperCubeVolume!{T<:Real}(rect::HyperRectVolume{T}, neworigin::Vector{T}, newedgelength::T)
    _setcubeboundaries!(rect.lo, rect.hi, neworigin, newedgelength)
end

@inline function _setcubeboundaries!{T<:Real}(lo::Vector{T}, hi::Vector{T}, origin::Vector{T}, edgelength::T)
    for i = 1:length(lo)
        lo[i] = origin[i] - edgelength * 0.5
        hi[i] = origin[i] + edgelength * 0.5
    end
end


function find_hypercube_centers(dataset::DataSet, datatree::Tree, whiteningresult::WhiteningResult, settings::HMIntegrationSettings)::Vector{Int64}
    weight_Prob = 1.0
    weight_Dens = 1.0
    weight_Loca = 10.0
    weights = [-Inf for i=1:dataset.N]

    sortLogProb = sortperm(dataset.logprob, rev = true)

    NMax = ceil(Int64, min(min(dataset.N, settings.max_startingIDs * 10), dataset.N * min(1.0, 10 * settings.max_startingIDs_fraction)))

    ignorePoint = falses(dataset.N)

    testlength = find_density_test_cube_edgelength(dataset.data[:, sortLogProb[1]], datatree, max(round(Int64, dataset.N * 0.001), dataset.P * 10))

    @showprogress for n in sortLogProb[1:NMax]
        if ignorePoint[n]
            continue
        end

        mode = view(dataset.data, :, n)

        weights[n] = weight_Prob * dataset.logprob[n]

        cubevol =  HyperCubeVolume(dataset.data[:, n], testlength)
        cube = IntegrationVolume(datatree, cubevol, true)
        for id in cube.pointcloud.pointIDs
            ignorePoint[id] = true
        end
    end

    sortIdx = sortperm(weights, rev = true)

    stop = 1
    for i = 1:dataset.N
        if weights[sortIdx[i]] == -Inf
            stop = i
            break
        end
    end
    NMax = stop - 1

    max_startingIDs = min(settings.max_startingIDs, dataset.N * settings.max_startingIDs_fraction)
    if stop > max_startingIDs
        NMax = round(Int64, max_startingIDs)
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

function find_density_test_cube_edgelength{T<:Real}(mode::Vector{T}, datatree::Tree{T}, points::Int64 = 100)
    return find_density_test_cube(mode, datatree, points)[1]
end

function find_density_test_cube{T<:Real}(mode::Vector{T}, datatree::Tree{T}, points::Int64)
    P = datatree.P

    l = 1.0
    tol = 1.0
    mult = 1.2^(1.0 / P)

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
