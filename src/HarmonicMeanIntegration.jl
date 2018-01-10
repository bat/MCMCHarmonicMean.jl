# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

"""
    hm_integrate(Data::DataSet, settings::HMIntegrationSettings = HMIntegrationSettings())::IntegrationResult

This function starts the Harmonic Mean Integration. It needs a data set as input (see DataSet documentation). When loading data from a
file (either HDF5 or ROOT) the function
function LoadMCMCData(path::String, params::Array{String}, range = Colon(), modelname::String = "", dataFormat::DataType = Float64)::DataSet()
can be used.
"""
function hm_integrate(bat_input::DensitySampleVector; range = Colon(), settings::HMIntegrationSettings = HMIntegrationStandardSettings())
    hm_integrate(DataSet(bat_input.weight), range = range, settings = settings)
end
function hm_integrate(bat_input::Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats}; range = Colon(), settings::HMIntegrationSettings = HMIntegrationStandardSettings())
    hm_integrate(DataSet(bat_input), range = range, settings = settings)
end
function hm_integrate{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}; range = Colon(), settings::HMIntegrationSettings = HMIntegrationStandardSettings())::IntegrationResult
    if dataset.N < dataset.P * 50
        @log_msg LOG_ERROR "Not enough points for integration"
    end

    if range != Colon()
        dataset.N = length(range)
        dataset.data = dataset.data[:, 1:dataset.N]
        dataset.weights = dataset.weights[1:dataset.N]
        dataset.logprob = dataset.logprob[1:dataset.N]
    end

    @log_msg LOG_INFO "Integration started. Data Points:\t$(dataset.N)\tParameters:\t$(dataset.P)"

    @log_msg LOG_INFO "Data Whitening started."
    whiteningresult = data_whitening(settings.whitening_method, dataset)
    @log_msg LOG_DEBUG "Whitening Result: $(whiteningresult)"


    @log_msg LOG_INFO "Create Data Tree"
    datatree = create_search_tree(dataset)
    @log_msg LOG_DEBUG "Search Tree: $datatree"

    @log_msg LOG_INFO "Find possible Hyperrectangle Centers"
    centerIDs = find_hypercube_centers(dataset, datatree, whiteningresult, settings)
    volumes = Vector{IntegrationVolume{T, I}}(0)

    @log_msg LOG_INFO "Find good tolerances for Hyperrectangle Creation"
    suggTolPts = dataset.N > 1e5 ? round(I, 0.001 * dataset.N) : suggTolPts = round(I, 0.01 * dataset.N)
    suggTol = findtolerance(dataset, datatree, centerIDs, 10, suggTolPts)
    @log_msg LOG_DEBUG "Tolerance: $suggTol"
    maxPoints::I = 0
    totalpoints::I = 0

    use_mt = settings.useMultiThreading && !settings.stop_ifenoughpoints
    nt = use_mt ? nthreads() : 1
    @log_msg LOG_INFO "Create Hyperrectangles using $nt thread(s)"

    thread_volumes = Vector{IntegrationVolume{T, I}}(length(centerIDs))

    mutex = Mutex()
    atomic_centerID = Atomic{I}(1)

    progressbar = Progress(length(centerIDs))
    if use_mt
        @everythread hyperrectangle_creationproccess!(thread_volumes, dataset, datatree, volumes, whiteningresult, suggTol, atomic_centerID, centerIDs, settings, mutex, progressbar)
    else
        hyperrectangle_creationproccess!(thread_volumes, dataset, datatree, volumes, whiteningresult, suggTol, atomic_centerID, centerIDs, settings, mutex, progressbar)
    end
    finish!(progressbar)

    #get volumes
    for i in eachindex(thread_volumes)
        if isassigned(thread_volumes, i) == false
        elseif thread_volumes[i].pointcloud.probfactor == 1.0 || thread_volumes[i].pointcloud.points < dataset.P * 4
        else
            push!(volumes, thread_volumes[i])
            maxPoints = max(maxPoints, thread_volumes[i].pointcloud.points)
            totalpoints += thread_volumes[i].pointcloud.points
        end
    end


    #remove rectangles with less than 10% points of the largest rectangle (in terms of points)
    j = length(volumes)
    for i = 1:length(volumes)
        if volumes[j].pointcloud.points < maxPoints * 0.01
            deleteat!(volumes, j)
        end
        j -= 1
    end


    @log_msg LOG_INFO "Integrating Hyperrectangles"

    nRes = length(volumes)
    IntResults = Array{IntermediateResult, 1}(nRes)


    progressbar = Progress(length(nRes))
    if settings.useMultiThreading
        @threads for i in 1:nRes
            IntResults[i] = integrate_hyperrectangle_noerror(dataset, volumes[i], whiteningresult.determinant)
            lock(mutex) do
                next!(progressbar)
            end
            @log_msg LOG_DEBUG "$i. Integral: $(IntResults[i].integral)\tVolume\t$(IntResults[i].volume)\tPoints:\t$(IntResults[i].points)"
        end
    else
        for i in 1:nRes
            IntResults[i] = integrate_hyperrectangle_noerror(dataset, volumes[i], whiteningresult.determinant)
            @log_msg LOG_DEBUG "$i. Integral: $(IntResults[i].integral)\tVolume\t$(IntResults[i].volume)\tPoints:\t$(IntResults[i].points)"
            next!(progressbar)
        end
    end
    finish!(progressbar)



    #remove integrals with no result
    j = nRes
    for i = 1:nRes
        if isnan(IntResults[j].integral)
            deleteat!(IntResults, j)
        end
        j -= 1
    end

    nRes = length(IntResults)

    rectweights = zeros(T, nRes)

    pweights = create_pointweights(dataset, volumes)
    for i in eachindex(volumes)
        for id in eachindex(volumes[i].pointcloud.pointIDs)
            rectweights[i] += 1.0 / pweights[volumes[i].pointcloud.pointIDs[id]]
        end
    end
    rectnorm = sum(rectweights)

    result = 0.0
    interror = 0.0
    points = 0.0
    volume = 0.

    for i = 1:nRes
        result += IntResults[i].integral * rectweights[i] / rectnorm
        points += IntResults[i].points * rectweights[i] / rectnorm
        volume += IntResults[i].volume * rectweights[i] / rectnorm
    end

    for i in 1:nRes
        interror += (IntResults[i].integral - result)^2
    end
    interror = sqrt(interror / (nRes - 1))
    if nRes == 1
        interror = IntResults[1].error
    end

    @log_msg LOG_INFO "Integration Result:\t $result +- $interror\nRectangles created: $(nRes)\tavg. points used: $points\t avg. volume: $volume"
    return IntegrationResult(result, interror, nRes, points, volume, volumes, centerIDs, suggTol, whiteningresult)
end



function findtolerance{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, datatree::Tree{T, I}, centerIDs::Vector{I}, ntestcubes::I, pts::I)::T
    ntestpts = [2, 4, 8] * pts
    @log_msg LOG_TRACE "Tolerance Test Cube Points: $([pts, ntestpts...])"

    vInc = Vector{T}(ntestcubes * (length(ntestpts) + 1))
    pInc = Vector{T}(ntestcubes * (length(ntestpts) + 1))

    cntr = 1
    for id = 1:ntestcubes
        c = find_density_test_cube(dataset.data[:, centerIDs[id]], dataset, datatree, pts)
        prevv = c[1]^dataset.P
        prevp = c[2].pointcloud.points
        for i in ntestpts
            c = find_density_test_cube(dataset.data[:, centerIDs[id]], dataset, datatree, i)
            v = c[1]^dataset.P
            p = c[2].pointcloud.points

            vInc[cntr] = v/prevv
            pInc[cntr] = p/prevp
            cntr += 1
            prevv = v
            prevp = p
        end
    end
    tols = vInc ./ pInc

    i = length(tols)
    while i > 0
        if isnan(tols[i]) || isinf(tols[i]) || tols[i] > 10 || tols[i] < 0.1
            deleteat!(tols, i)
        end
        i -= 1
    end

    @log_msg LOG_DEBUG "Tolerance List: $tols"

    suggTol::T = mean(tols)
    #suggTol = (suggTol - 1) * 4 + 1

    return suggTol
end


function hyperrectangle_creationproccess!{T<:AbstractFloat, I<:Integer}(results::Vector{IntegrationVolume{T, I}}, dataset::DataSet{T, I}, datatree::Tree{T, I},
        volumes::Vector{IntegrationVolume{T, I}}, whiteningresult::WhiteningResult{T},
        tolerance::T, atomic_centerID::Atomic{I}, centerIDs::Vector{I}, settings::HMIntegrationSettings, mutex::Mutex, progressbar::Progress)

    while true
        #get new center ID
        idc = atomic_add!(atomic_centerID, 1)
        if idc > length(centerIDs)
            break
        end
        id = centerIDs[idc]

        center = dataset.data[:, id]
        inV = false

        lock(mutex) do
            next!(progressbar)
            #check if starting id is inside another rectangle

            if settings.skip_centerIDsinsideHyperRects
                for i in eachindex(results)
                    if isassigned(results, i)
                        if BAT.in(center, results[i].spatialvolume)
                            inV = true
                            break
                        end
                    end
                end
            end
        end

        if inV
            continue
        end
        results[idc] = create_hyperrectangle(center, dataset, datatree, volumes, whiteningresult, tolerance, settings)
        @log_msg LOG_DEBUG "Hyperrectangle created. Points:\t$(results[idc].pointcloud.points)\tVolume:\t$(results[idc].volume)\tProb. Factor:\t$(results[idc].pointcloud.probfactor)"
    end
end

"""
    create_hyperrectangle{T<:AbstractFloat, I<:Integer}(Mode::Vector{T}, dataset::DataSet{T, I}, datatree::Tree{T}, volumes::Vector{IntegrationVolume{T, I}}, whiteningresult::WhiteningResult{T},
        Tolerance::T, settings::HMIntegrationSettings)::IntegrationVolume{T, I}

This function tries to create
 a hyper-rectangle around a starting point. It builds a cube first and adapts each face of individually to fit to the data as good as possible.
If the creation process fails a rectangle with no or only one point might be returned (check probfactor). The creation process might also be stopped because the rectangle overlaps
with another.
"""
function create_hyperrectangle{T<:AbstractFloat, I<:Integer}(Mode::Vector{T}, dataset::DataSet{T, I}, datatree::Tree{T}, volumes::Vector{IntegrationVolume{T, I}}, whiteningresult::WhiteningResult{T},
        Tolerance::T, settings::HMIntegrationSettings)::IntegrationVolume{T, I}

    edgelength::T = 1.0

    cube = HyperCubeVolume(Mode, edgelength)
    vol = IntegrationVolume(dataset, datatree, cube, true)

    while vol.pointcloud.points > 0.01 * dataset.N
        edgelength *= 0.5
        HyperCubeVolume!(cube, Mode, edgelength)
        IntegrationVolume!(vol, dataset, datatree, cube, true)
    end
    tol = 1.0
    step = 0.7
    direction = 0
    PtsIncrease = 0.0
    VolIncrease = 1.0

    it = 0
    while vol.pointcloud.probfactor < whiteningresult.targetprobfactor / tol || vol.pointcloud.probfactor > whiteningresult.targetprobfactor
        tol += 0.01 * 2^it
        it += 1

        if vol.pointcloud.probfactor > whiteningresult.targetprobfactor
            #decrease side length
            VolIncrease = edgelength^dataset.P
            edgelength *= step
            VolIncrease = edgelength^dataset.P / VolIncrease

            step = adjuststepsize!(step, direction == -1)
            direction = -1
        else
            #increase side length
            VolIncrease = edgelength^dataset.P
            edgelength /= step
            VolIncrease = edgelength^dataset.P / VolIncrease

            step = adjuststepsize!(step, direction == 1)
            direction = 1
        end
        PtsIncrease = vol.pointcloud.points
        HyperCubeVolume!(cube, Mode, edgelength)
        IntegrationVolume!(vol, dataset, datatree, cube, true)

        PtsIncrease = vol.pointcloud.points / PtsIncrease

        if vol.pointcloud.points > 0.01 * dataset.N && vol.pointcloud.probfactor < whiteningresult.targetprobfactor
            break
        end
    end



    #LogLow("\tTEST Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")


    wasCubeChanged = true
    newvol::IntegrationVolume{T, I} = deepcopy(vol)

    ptsTolInc::T = Tolerance
    ptsTolDec::T = Tolerance# + (Tolerance - 1) * 1.5

    dimensionsFinished = falses(dataset.P)
    spvol::HyperRectVolume{T} = deepcopy(vol.spatialvolume)
    searchvol::HyperRectVolume{T} = deepcopy(spvol)
    buffer::T = 0.0

    const increase::T = settings.rect_increase
    const decrease::T = 1.0 - 1.0 / (1.0 + increase)

    while wasCubeChanged && vol.pointcloud.probfactor > 1.0

        wasCubeChanged = false


        for p::I = 1:dataset.P
            if dimensionsFinished[p]
                #risky, can improve results but may lead to endless loop
                #dimensionsFinished[p] = false
                continue
            end

            change = true

            #adjust lower bound
            change1 = 0
            while change1 != 0 && vol.pointcloud.probfactor > 1.0
                margin = spvol.hi[p] - spvol.lo[p]
                buffer = spvol.lo[p]
                spvol.lo[p] -= margin * increase

                PtsIncrease = vol.pointcloud.points
                resize_integrationvol!(newvol, vol, dataset, datatree, p, spvol, false, searchvol)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probweightfactor < whiteningresult.targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc) && change1 != -1
                    copy!(vol, newvol)
                    wasCubeChanged = true
                    change = true
                    change1 = 1
                    @log_msg LOG_TRACE "lo inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                else
                    #revert changes
                    spvol.lo[p] = buffer

                    margin = spvol.hi[p] - spvol.lo[p]
                    buffer = spvol.lo[p]
                    spvol.lo[p] += margin * decrease

                    PtsIncrease = vol.pointcloud.points
                    resize_integrationvol!(newvol, vol, dataset, datatree, p, spvol, false, searchvol)

                    PtsIncrease = newvol.pointcloud.points / PtsIncrease

                    if PtsIncrease > (1.0 - decrease / ptsTolDec) && change1 != 1
                        copy!(vol, newvol)
                        wasCubeChanged = true
                        change = true
                        change1 = -1
                        @log_msg LOG_TRACE "lo dec p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                    else
                        #revert changes
                        spvol.lo[p] = buffer
                        change1 = 0
                    end
                end
            end

            #adjust upper bound
            change2 = 0
            while change2!= 0 && vol.pointcloud.probfactor > 1.0
                margin = spvol.hi[p] - spvol.lo[p]
                buffer = spvol.hi[p]
                spvol.hi[p] += margin * increase

                PtsIncrease = vol.pointcloud.points
                resize_integrationvol!(newvol, vol, dataset, datatree, p, spvol, false, searchvol)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probweightfactor < whiteningresult.targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc) && change2 != -1
                    copy!(vol, newvol)
                    wasCubeChanged = true
                    change = true
                    change2 = 1
                    @log_msg LOG_TRACE "up inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                else
                    #revert changes
                    spvol.hi[p] = buffer

                    margin = spvol.hi[p] - spvol.lo[p]
                    buffer = spvol.hi[p]
                    spvol.hi[p] -= margin * decrease

                    PtsIncrease = vol.pointcloud.points
                    resize_integrationvol!(newvol, vol, dataset, datatree, p, spvol, false, searchvol)

                    PtsIncrease = newvol.pointcloud.points / PtsIncrease

                    if PtsIncrease > (1.0 - decrease / ptsTolDec) && change2 != 1
                        copy!(vol, newvol)
                        wasCubeChanged = true
                        change = true
                        change2 = -1
                        @log_msg LOG_TRACE "up dec p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                    else
                        #revert changes
                        spvol.hi[p] = buffer
                        change2 = 0
                    end
                end
            end

            dimensionsFinished[p] = !change
        end

    end


    res = search(dataset, datatree, vol.spatialvolume, true)
    resize!(vol.pointcloud.pointIDs, res.points)
    copy!(vol.pointcloud.pointIDs, res.pointIDs)
    vol.pointcloud.maxLogProb = res.maxLogProb
    vol.pointcloud.minLogProb = res.minLogProb
    vol.pointcloud.probfactor = exp(vol.pointcloud.maxLogProb - vol.pointcloud.minLogProb)
    vol.pointcloud.probweightfactor = exp(vol.pointcloud.maxWeightProb - vol.pointcloud.minWeightProb)

    return vol
end

@inline function adjuststepsize!(Step, Increase::Bool)
    if Increase
        return Step * 0.5
    else
        return Step * 2.0
    end
end

@inline function calculate_overlapping{T<:AbstractFloat, I<:Integer}(intvol::IntegrationVolume{T, I}, other::HyperRectVolume{T})::T
    innerVol::T = 1.0
    rect = intvol.spatialvolume

    for p = 1:ndims(rect)
        if rect.lo[p] > other.hi[p] || rect.hi[p] < other.lo[p]
            return 0.0
        end
        if rect.lo[p] > other.lo[p] && rect.hi[p] < other.hi[p]
            innerVol *= rect.hi[p] - rect.lo[p]
        else
            innerVol *= min(abs(rect.lo[p] - other.hi[p]), abs(rect.hi[p] - other.lo[p]))
        end
    end

    return T(innerVol / intvol.volume)
end

function integrate_hyperrectangle_noerror{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, integrationvol::IntegrationVolume{T, I}, determinant::T)::IntermediateResult{T}
    s::Float64 = 0.0
    count::T = 0.0
    for i in integrationvol.pointcloud.pointIDs
        #for numerical stability
        prob = (dataset.logprob[i] - integrationvol.pointcloud.maxLogProb)
        s += 1.0 / exp(prob) * dataset.weights[i]
        count += dataset.weights[i]
    end

    integral = sum(dataset.weights) * integrationvol.volume / s / determinant / exp(-integrationvol.pointcloud.maxLogProb)

    @log_msg LOG_TRACE "Volume:\t$(integrationvol.volume)\tIntegral: $integral\tPoints:\t$(count)"

    return IntermediateResult(T(integral), T(0), T(integrationvol.pointcloud.points), integrationvol.volume)
end

function integrate_hyperrectangle{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, datatree::Tree{T, I}, integrationvol::IntegrationVolume{T, I}, determinant::T)::IntermediateResult{T}
    Results = Array{IntermediateResult, 1}(5)
    nIntegrals = Float64(length(Results))

    #start 10% larger, stop 10% smaller
    volfactor::T = 0.9^(1/dataset.P)
    step = deepcopy(integrationvol.spatialvolume.lo)


    for p = 1:dataset.P
        margin = integrationvol.spatialvolume.hi[p] - integrationvol.spatialvolume.lo[p]
        integrationvol.spatialvolume.lo[p] -= margin * 0.5 * volfactor
        integrationvol.spatialvolume.hi[p] += margin * 0.5 * volfactor
        step[p] = margin * 0.5 * volfactor / 4
    end

    IntegrationVolume!(integrationvol, dataset, datatree, integrationvol.spatialvolume, true)
    Results[1] = integrate_hyperrectangle_noerror(dataset, integrationvol, determinant)

    p_range = 1:dataset.P
    for i = 1:length(Results) - 1
        for p in p_range
            integrationvol.spatialvolume.lo[p] += step[p]
            integrationvol.spatialvolume.hi[p] -= step[p]
        end
        shrink_integrationvol!(integrationvol, dataset, integrationvol.spatialvolume)
        Results[i + 1] = integrate_hyperrectangle_noerror(dataset, integrationvol, determinant)
    end


    integral = 0.0
    count = 0.0
    for i = 1:length(Results)
        integral += Results[i].integral / nIntegrals
        count += Results[i].points / nIntegrals
    end

    interror = 0.0
    for i = 1:length(Results)
        interror += (Results[i].integral - integral) ^ 2
    end
    interror = sqrt(interror / (nIntegrals - 1))

    if isnan(integral) || integral == Inf || integral == -Inf
        res = integrate_hyperrectangle_noerror(dataset, integrationvol, determinant)
        @log_msg LOG_ERROR "No error could be calculated for integrate_hyperrectangle()"
        return IntermediateResult(res.integral, 0.0, res.points, integrationvol.volume)
    end

    return IntermediateResult(integral, interror, count, integrationvol.volume)
end
