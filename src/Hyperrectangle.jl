# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


"""
    HyperCubeVolume{T<:Real}(origin::Vector{T}, edgelength::T)::HyperRectVolume

creates a hypercube shaped spatial volume
"""
function HyperCubeVolume(
    origin::Vector{T},
    edgelength::T
)::HyperRectVolume{T} where {T<:AbstractFloat}

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
function HyperCubeVolume!(
    rect::HyperRectVolume{T},
    neworigin::Vector{T},
    newedgelength::T
) where {T<:AbstractFloat}

    _setcubeboundaries!(rect.lo, rect.hi, neworigin, newedgelength)
end

@inline function _setcubeboundaries!(
    lo::Vector{T},
    hi::Vector{T},
    origin::Vector{T},
    edgelength::T
) where {T<:AbstractFloat}

    for i = 1:length(lo)
        lo[i] = origin[i] - edgelength * 0.5
        hi[i] = origin[i] + edgelength * 0.5
    end
end

"""
    find_hypercube_centers(dataset::DataSet{T, I}, whiteningresult::WhiteningResult, settings::HMISettings)::Vector{I}

finds possible starting points for the hyperrectangle creation
"""
function find_hypercube_centers(
    dataset::DataSet{T, I},
    whiteningresult::WhiteningResult{T},
    settings::HMISettings
)::Bool where {T<:AbstractFloat, I<:Integer}


weights = [T(-Inf) for i=1:dataset.N]

   sortLogProb = sortperm(dataset.logprob, rev = true)

   NMax = round(I, sqrt(dataset.N) * settings.max_startingIDs_fraction)
   if NMax < 2 * settings.warning_minstartingids
       NMax = min(2 * settings.warning_minstartingids, dataset.N)
   end
   @log_msg LOG_DEBUG "Considered starting Points $NMax"

   ignorePoint = falses(dataset.N)

   testlength = find_density_test_cube_edgelength(dataset.data[:, sortLogProb[1]], dataset, round(I, sqrt(dataset.N)))
   @log_msg LOG_DEBUG "Test length of starting cubes: $testlength"

   @showprogress for n::I in sortLogProb[1:NMax]
       if ignorePoint[n]
           continue
       end

       weights[n] = dataset.logprob[n]

       cubevol = HyperCubeVolume(dataset.data[:, n], testlength)
       cube = IntegrationVolume(dataset, cubevol, true)

       ignorePoint[cube.pointcloud.pointIDs] = true
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

   #return at least settings.warning_minstartingids possible hyper-rect centers
   if NMax < settings.warning_minstartingids && stop >= settings.warning_minstartingids
       NMax = settings.warning_minstartingids
   elseif NMax < settings.warning_minstartingids && stop < settings.warning_minstartingids
       @log_msg LOG_WARNING "Returned minimum number of starting points: $(settings.warning_minstartingids)"
       returnids = round.(I, [i for i=0:(settings.warning_minstartingids-1)] * dataset.N * 0.01 .+ 1.0)
       dataset.startingIDs = sortLogProb[returnids]
       return false
   end

   @log_msg LOG_DEBUG "Possible Hypersphere Centers: $NMax out of $(dataset.N) points"

   dataset.startingIDs = sortIdx[1:NMax]

return true

#TODO FIND BUG
#=
    sortLogProb = sortperm(dataset.logprob, rev = true)

    NMax = min(settings.max_startingIDs, round(I, sqrt(dataset.N * settings.max_startingIDs_fraction)))
    NConsidered = round(I, sqrt(dataset.N) * settings.max_startingIDs_fraction)
    @log_msg LOG_DEBUG "Considered starting samples $NConsidered"

    discardedsamples = falses(dataset.N)

    testlength = find_density_test_cube_edgelength(dataset.data[:, sortLogProb[1]], dataset, round(I, sqrt(dataset.N)))
    @log_msg LOG_DEBUG "Test length of starting cubes: $testlength"

    maxprob = dataset.logprob[sortLogProb[1]]
    startingsamples = Array{I, 1}(NMax)
    cntr = 0
    @showprogress for n::I in sortLogProb[1:NConsidered]
        if discardedsamples[n]
            continue
        end
        if cntr == NMax || dataset.logprob[n] < maxprob - log(whiteningresult.targetprobfactor)
            break
        end

        cntr += 1
        startingsamples[cntr] = n

        cubevol = HyperCubeVolume(dataset.data[:, n], testlength)
        cube = IntegrationVolume(dataset, cubevol, true)

        discardedsamples[cube.pointcloud.pointIDs] = true
    end
    resize!(startingsamples, cntr)

    success = true
    if cntr < settings.warning_minstartingids
        startingsamples = sortperm(dataset.logprob)[1:settings.warning_minstartingids]
        success = false
        @log_msg LOG_WARNING "Returned minimum number of starting points: $(settings.warning_minstartingids)"
    end


    @log_msg LOG_DEBUG "Selected Starting Samples: $cntr out of $(dataset.N) points"
    dataset.startingIDs = startingsamples

    success
    =#
end

function find_density_test_cube_edgelength(
    mode::Vector{T},
    dataset::DataSet{T, I},
    points::I = 100
)::T where {T<:AbstractFloat, I<:Integer}

    return find_density_test_cube(mode, dataset, points)[1]
end

function find_density_test_cube(
    mode::Vector{T},
    dataset::DataSet{T, I},
    points::I
)::Tuple{T, IntegrationVolume{T, I}} where {T<:AbstractFloat, I<:Integer}

    P = dataset.P

    l::T = 1.0
    tol::T = 1.0
    mult::T = 2.0^(1.0 / P)
    last_change = 0

    rect = HyperCubeVolume(mode, l)
    intvol = IntegrationVolume(dataset, rect, false)
    pt = intvol.pointcloud.points

    iterations = 0
    while pt < points / tol || pt > points * tol
        iterations += 1
        tol += 0.001 * iterations
        if pt > points
            l /= mult
            mult = last_change == -1 ? mult^2.0 : mult^0.5
            last_change = -1
        else
            l *= mult
            mult = last_change == 1 ? mult^2.0 : mult^0.5
            last_change = 1
        end

        HyperCubeVolume!(rect, mode, l)
        IntegrationVolume!(intvol, dataset, rect, false)
        pt = intvol.pointcloud.points
    end

    @log_msg LOG_TRACE "Tolerance Test Cube: Iterations $iterations\tPoints: $(intvol.pointcloud.points)\ttarget Points: $points"
    return l, intvol
end
