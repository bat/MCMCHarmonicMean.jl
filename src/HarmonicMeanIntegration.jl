# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

"""
    hm_integrate(Data::DataSet, settings::HMIntegrationSettings = HMIntegrationSettings())::IntegrationResult

This function starts the Harmonic Mean Integration. It needs a data set as input (see DataSet documentation). When loading data from a
file (either HDF5 or ROOT) the function
function LoadMCMCData(path::String, params::Array{String}, range = Colon(), modelname::String = "", dataFormat::DataType = Float64)::DataSet()
can be used.
"""
function hm_integrate(bat_input::DensitySampleVector; range = Colon(), settings::HMIntegrationSettings = HMIntegrationStandardSettings())
    return hm_integrate(HMIData(DataSet(bat_input)))
end
function hm_integrate(bat_input::Tuple{DensitySampleVector, MCMCSampleIDVector, MCMCBasicStats})
    return hm_integrate(HMIData(DataSet(bat_input)))
end
function hm_swapdata(result::HMIData{T, I}, data::DataSet{T, I}) where {T<:AbstractFloat, I<:Integer}
    if !data.iswhitened result.whiteningresult = Nullable{WhiteningResult{T}}() end
    result.datatree = Nullable{SearchTree}()
    result.dataset = data
    result.datasetchange = true
    return
end
function hm_reset_hyperrectangles(result::HMIData{T, I}) where {T<:AbstractFloat, I<:Integer}
    result.volumelist = Vector{IntegrationVolume{T, I}}(0)
    result.startingIDs = Vector{I}(0)
    return
end
function hm_reset_tolerance(result::HMIData{T, I}) where {T<:AbstractFloat, I<:Integer}
    result.tolerance = 0.0
end

function hm_integrate(
    result::HMIData{T, I};
    range = Colon(),
    settings::HMIntegrationSettings = HMIntegrationStandardSettings()
)::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

    dataset = result.dataset
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

    if isnull(result.whiteningresult)
        @log_msg LOG_INFO "Data Whitening started."
        result.whiteningresult = data_whitening(settings.whitening_method, dataset)
    end
    whiteningresult = get(result.whiteningresult)
    @log_msg LOG_DEBUG "Whitening Result: $(whiteningresult)"

    if isnull(result.datatree)
        @log_msg LOG_INFO "Create Data Tree"
        result.datatree = create_search_tree(dataset)
    end
    datatree = get(result.datatree)
    @log_msg LOG_TRACE "Search Tree: $datatree"

    if isempty(result.startingIDs)
        @log_msg LOG_INFO "Find possible Hyperrectangle Centers"
        result.startingIDs = find_hypercube_centers(dataset, datatree, whiteningresult, settings)
    end
    startingIDs = result.startingIDs

    if iszero(result.tolerance)
        @log_msg LOG_INFO "Find good tolerances for Hyperrectangle Creation"
        suggTolPts = max(dataset.P * 4, ceil(I, sqrt(dataset.N)))
        result.tolerance = findtolerance(dataset, datatree, startingIDs, min(10, settings.warning_minstartingids), suggTolPts)
    end
    suggTol = result.tolerance
    @log_msg LOG_DEBUG "Tolerance: $suggTol"


    use_mt = settings.useMultiThreading && !settings.stop_ifenoughpoints


    maxPoints::I = 0
    totalpoints::I = 0
    if isempty(result.volumelist)
        @log_msg LOG_INFO "Create Hyperrectangles using $(use_mt ? nthreads() : 1) thread(s)"
        thread_volumes = Vector{IntegrationVolume{T, I}}(length(startingIDs))

        atomic_centerID = Atomic{I}(1)

        progressbar = Progress(length(startingIDs))
        if use_mt
            @everythread hyperrectangle_creationproccess!(thread_volumes, dataset, datatree, result.volumelist, whiteningresult, suggTol, atomic_centerID, startingIDs, settings, progressbar)
        else
            hyperrectangle_creationproccess!(thread_volumes, dataset, datatree, result.volumelist, whiteningresult, suggTol, atomic_centerID, startingIDs, settings, progressbar)
        end
        finish!(progressbar)

        #get suitable integration volumes
        for i in eachindex(thread_volumes)
            if isassigned(thread_volumes, i) == false
            elseif thread_volumes[i].pointcloud.probfactor == 1.0 || thread_volumes[i].pointcloud.points < dataset.P * 4
            else
                push!(result.volumelist, thread_volumes[i])
                maxPoints = max(maxPoints, thread_volumes[i].pointcloud.points)
                totalpoints += thread_volumes[i].pointcloud.points
            end
        end
    else
        if result.datasetchange @log_msg LOG_INFO "Updating $(length(result.volumelist)) Hyperrectangles using $(use_mt ? nthreads() : 1) thread(s)" end
        #update pointIDs for each integrationvolume
        if use_mt
            @threads for i in eachindex(result.volumelist)
                if result.datasetchange update!(result.volumelist[i], dataset, datatree) end

                maxPoints = max(maxPoints, result.volumelist[i].pointcloud.points)
                totalpoints += result.volumelist[i].pointcloud.points
            end
        else
            for i in eachindex(result.volumelist)
                if result.datasetchange update!(result.volumelist[i], dataset, datatree) end

                maxPoints = max(maxPoints, result.volumelist[i].pointcloud.points)
                totalpoints += result.volumelist[i].pointcloud.points
            end
        end
    end
    volumes = result.volumelist
    result.datasetchange = false



    #remove rectangles with less than 1% points of the largest rectangle (in terms of points)
    j = length(volumes)
    for i = 1:length(volumes)
        if volumes[j].pointcloud.points < maxPoints * 0.01 || volumes[j].pointcloud.points < dataset.P * 4
            deleteat!(volumes, j)
        end
        j -= 1
    end

    if !(length(volumes) > 0)
        @log_msg LOG_ERROR "No hyper-rectangles could be created. Try integration with more points or different settings."
    end

    nRes = length(volumes)
    IntResults = Array{IntermediateResult, 1}(nRes)

    @log_msg LOG_INFO "Integrating $nRes Hyperrectangles"

    progressbar = Progress(length(nRes))
    if settings.useMultiThreading
        @threads for i in 1:nRes
            IntResults[i] = integrate_hyperrectangle(dataset, volumes[i], whiteningresult.determinant, settings.nvolumerand)
            lock(BAT.Logging._global_lock) do
                next!(progressbar)
            end
        end
    else
        for i in 1:nRes
            IntResults[i] = integrate_hyperrectangle(dataset, volumes[i], whiteningresult.determinant, settings.nvolumerand)
            next!(progressbar)
        end
    end
    finish!(progressbar)



    #remove integrals with no result
    j = nRes
    for i = 1:nRes
        try
            if !isassigned(IntResults, j) || isnan(IntResults[j].integral)
                deleteat!(IntResults, j)
                deleteat!(volumes, j)
            end
        catch e
            @log_msg LOG_ERROR string(e)
            println(IntResults)
            println(length(IntResults))
            if length(IntResults) >= j
                deleteat!(IntResults, j)
                deleteat(volumes, j)
            end
        finally
            j -= 1
        end
    end

    @assert length(volumes) == length(IntResults)

    nRes = length(IntResults)

    local rectweights::Vector{T}
    if settings.userectweights
        rectweights = zeros(T, nRes)

        pweights = create_pointweights(dataset, volumes)
        for i in eachindex(volumes)
            trw = sum(dataset.weights[volumes[i].pointcloud.pointIDs])
            for id in eachindex(volumes[i].pointcloud.pointIDs)
                rectweights[i] += 1.0 / trw / pweights[volumes[i].pointcloud.pointIDs[id]]# / IntResults[i].error
            end
        end
    else
        rectweights = ones(T, nRes)
    end

    rectnorm = sum(rectweights)

    _results = [IntResults[i].integral for i in eachindex(IntResults)]
    _points  = [IntResults[i].points   for i in eachindex(IntResults)]
    _volumes = [IntResults[i].volume   for i in eachindex(IntResults)]

    integral, point, volume = tmean(_results, _points, _volumes, weights = rectweights)
    error = sqrt(var(_results) / rectnorm)

    pointvar = sqrt(var(_points))

    @log_msg LOG_INFO "Integration Result:\t $integral +- $(error))\nRectangles created: $(nRes)\tavg. points used: $(round(Int64, point)) +- $(round(Int64, pointvar))\t avg. volume: $volume"
    result.integral = integral
    result.error = error
    result.points = point
    result.volume = volume
    result.integrals = IntResults
    return result
end



function findtolerance{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, datatree::SearchTree, startingIDs::Vector{I}, ntestcubes::I, pts::I)::T
    ntestpts = [2, 4, 8] * pts
    @log_msg LOG_TRACE "Tolerance Test Cube Points: $([pts, ntestpts...])"

    vInc = Vector{T}(ntestcubes * (length(ntestpts) + 1))
    pInc = Vector{T}(ntestcubes * (length(ntestpts) + 1))

    cntr = 1
    for id = 1:ntestcubes
        c = find_density_test_cube(dataset.data[:, startingIDs[id]], dataset, datatree, pts)
        prevv = c[1]^dataset.P
        prevp = c[2].pointcloud.points
        for i in ntestpts
            c = find_density_test_cube(dataset.data[:, startingIDs[id]], dataset, datatree, i)
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
        if isnan(tols[i]) || isinf(tols[i]) || tols[i] <= 1.0
            deleteat!(tols, i)
        end
        i -= 1
    end

    @log_msg LOG_TRACE "Tolerance List: $tols"
    if length(tols) < 4
        @log_msg LOG_WARNING "Tolerance calculation failed. Tolerance is set to default to 1.5"
    end
    suggTol::T = length(tols) < 4 ? 1.5 : tmean(tols)[1]
    #suggTol = (suggTol - 1) * 4 + 1

    return suggTol
end


function hyperrectangle_creationproccess!{T<:AbstractFloat, I<:Integer}(
    results::Vector{IntegrationVolume{T, I}},
    dataset::DataSet{T, I},
    datatree::SearchTree,
    volumes::Vector{IntegrationVolume{T, I}},
    whiteningresult::WhiteningResult{T},
    tolerance::T,
    atomic_centerID::Atomic{I},
    startingIDs::Vector{I},
    settings::HMIntegrationSettings,
    progressbar::Progress
)

    while true
        #get new center ID
        idc = atomic_add!(atomic_centerID, 1)
        if idc > length(startingIDs)
            break
        end
        id = startingIDs[idc]

        center = dataset.data[:, id]
        inV = false

        lock(BAT.Logging._global_lock) do
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
    create_hyperrectangle{T<:AbstractFloat, I<:Integer}(Mode::Vector{T}, dataset::DataSet{T, I}, datatree::SearchTree, volumes::Vector{IntegrationVolume{T, I}}, whiteningresult::WhiteningResult{T},
        Tolerance::T, settings::HMIntegrationSettings)::IntegrationVolume{T, I}

This function tries to create
 a hyper-rectangle around a starting point. It builds a cube first and adapts each face of individually to fit to the data as good as possible.
If the creation process fails a rectangle with no or only one point might be returned (check probfactor). The creation process might also be stopped because the rectangle overlaps
with another.
"""
function create_hyperrectangle{T<:AbstractFloat, I<:Integer}(Mode::Vector{T}, dataset::DataSet{T, I}, datatree::SearchTree, volumes::Vector{IntegrationVolume{T, I}}, whiteningresult::WhiteningResult{T},
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
        tol += 0.01 * it
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



    @log_msg LOG_TRACE "Starting Hypercube Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)"


    wasCubeChanged = true
    newvol::IntegrationVolume{T, I} = deepcopy(vol)

    ptsTolInc::T = Tolerance
    ptsTolDec::T = Tolerance * 1.1

    spvol::HyperRectVolume{T} = deepcopy(vol.spatialvolume)
    searchvol::HyperRectVolume{T} = deepcopy(spvol)

    const increase_default::T = settings.rect_increase
    increase = increase_default
    decrease = 1.0 - 1.0 / (1.0 + increase)

    const min_points = 20
    const max_iterations_per_dim = 5

    while wasCubeChanged && vol.pointcloud.probfactor > 1.0

        if vol.pointcloud.points * increase < min_points
            increase *= ceil(I, min_points / (vol.pointcloud.points * increase))
            decrease = 1.0 - 1.0 / (1.0 + increase)
            @log_msg LOG_TRACE "Changed increase to $increase"
        elseif increase > increase_default && vol.pointcloud.points * increase_default > 2 * min_points
            increase = increase_default
            decrease = 1.0 - 1.0 / (1.0 + increase)
            @log_msg LOG_TRACE "Reset increase to $increase"
        end


        wasCubeChanged = false


        for p::I = 1:dataset.P

            change = true

            #adjust lower bound
            change1 = 2
            iteration_per_dimension = 0
            while change1 != 0 && vol.pointcloud.probfactor > 1.0 && iteration_per_dimension < max_iterations_per_dim
                iteration_per_dimension += 1

                margin = spvol.hi[p] - spvol.lo[p]
                buffer = spvol.lo[p]
                spvol.lo[p] -= margin * increase

                PtsIncrease = vol.pointcloud.points
                resize_integrationvol!(newvol, vol, dataset, datatree, p, spvol, false, searchvol)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probfactor < whiteningresult.targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc) && change1 != -1
                    copy!(vol, newvol)
                    wasCubeChanged = true
                    change = true
                    change1 = 1
                    @log_msg LOG_TRACE "lo inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                elseif vol.pointcloud.points > 100 * (1 + increase) && margin > 0.01 * vol.volume^(1.0 / dataset.P)
                    #revert changes - important to also partially revert newvol, because resize_integrationvol function calls update! function which updates the points
                    #by adding only the new number of new points and not by overwriting
                    newvol.pointcloud.points = vol.pointcloud.points
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
                        newvol.pointcloud.points = vol.pointcloud.points
                        spvol.lo[p] = buffer
                        change1 = 0
                    end
                #if there are only very few points, accept if there are points added.
                elseif vol.pointcloud.points < 100 && newvol.pointcloud.probfactor < whiteningresult.targetprobfactor && PtsIncrease > 1.0 && change1 != -1
                    copy!(vol, newvol)
                    wasCubeChanged = true
                    change = true
                    change1 = 1
                    @log_msg LOG_TRACE "(sr) lo inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                else
                    #revert changes
                    newvol.pointcloud.points = vol.pointcloud.points
                    spvol.lo[p] = buffer
                    change1 = 0
                end
            end

            #adjust upper bound
            change2 = 2
            iteration_per_dimension = 0
            while change2!= 0 && vol.pointcloud.probfactor > 1.0 && iteration_per_dimension < max_iterations_per_dim
                iteration_per_dimension += 1
                margin = spvol.hi[p] - spvol.lo[p]
                buffer = spvol.hi[p]
                spvol.hi[p] += margin * increase

                PtsIncrease = vol.pointcloud.points
                resize_integrationvol!(newvol, vol, dataset, datatree, p, spvol, false, searchvol)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probfactor < whiteningresult.targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc) && change2 != -1
                    copy!(vol, newvol)
                    wasCubeChanged = true
                    change = true
                    change2 = 1
                    @log_msg LOG_TRACE "up inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                elseif vol.pointcloud.points > 100 * (1 + increase) && margin > 0.01 * vol.volume^(1.0 / dataset.P)
                    #revert changes
                    newvol.pointcloud.points = vol.pointcloud.points
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
                        newvol.pointcloud.points = vol.pointcloud.points
                        spvol.hi[p] = buffer
                        change2 = 0
                    end
                #if there are only very few points, accept if there are points added.
                elseif vol.pointcloud.points < 100 && newvol.pointcloud.probfactor < whiteningresult.targetprobfactor && PtsIncrease > 1.0 && change2 != -1
                    copy!(vol, newvol)
                    wasCubeChanged = true
                    change = true
                    change2 = 1
                    @log_msg LOG_TRACE "(sr) up inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
                else
                    #revert changes
                    newvol.pointcloud.points = vol.pointcloud.points
                    spvol.hi[p] = buffer
                    change2 = 0
                end
            end
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


function integrate_hyperrectangle(
    dataset::DataSet{T, I},
    integrationvol::IntegrationVolume{T, I},
    determinant::T,
    nsmallervols::I
)::IntermediateResult{T} where {T<:AbstractFloat, I<:Integer}

    integrals = zeros(T, nsmallervols)
    localvolume = deepcopy(integrationvol)

    for i = 1:nsmallervols
    #for i = 2:1+nsmallervols
        #shrinking of 5% per iteration (on average)
        rdec = 1.0 - rand() * 0.1
        dim_change = rdec^(1.0 / dataset.P)

        margins = (localvolume.spatialvolume.hi .- localvolume.spatialvolume.lo) .* (1.0 - dim_change) .* 0.5
        localvolume.spatialvolume.lo .+= margins
        localvolume.spatialvolume.hi .-= margins

        shrink_integrationvol!(localvolume, dataset, localvolume.spatialvolume)
        integrals[i] = integrate_hyperrectangle(dataset, localvolume.pointcloud.pointIDs, localvolume.volume, determinant, localvolume.pointcloud.maxLogProb)

    end

    #no division by sqrt(11) because if empty volume is included the results are not expected to be equal. -> helps finding volumes with empty volume if error is high
    error::T = nsmallervols < 2 ? 0.0 : sqrt(var(integrals))
    integral::T = mean(integrals)

    @log_msg LOG_DEBUG "Integral: $integral\tError: $error"

    return IntermediateResult(integral, error, T(integrationvol.pointcloud.points), integrationvol.volume)
end

function integrate_hyperrectangle{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, pointIDs::Vector{I}, volume::T, determinant::T, maxLogProb::T = 0.0)::T
    s::T = 0.0
    count::T = 0.0
    for i in pointIDs
        #for numerical stability
        prob = (dataset.logprob[i] - maxLogProb)
        s += 1.0 / exp(prob) * dataset.weights[i]
        count += dataset.weights[i]
    end

    totalWeight::T = sum(dataset.weights)
    integral::T = totalWeight * volume / s / determinant / exp(-maxLogProb)

    return integral
end
