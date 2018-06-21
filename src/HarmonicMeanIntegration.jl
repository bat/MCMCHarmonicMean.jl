# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

_global_mt_setting = true

macro mt(expr)
    quote
        if _global_mt_setting
            @everythread $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

macro mt_threads(expr)
    quote
        if _global_mt_setting
            @threads $(esc(expr))
        else
            $(esc(expr))
        end
    end
end
"""
    hm_integrate(Data::DataSet, settings::HMISettings = HMISettings())::IntegrationResult

This function starts the Harmonic Mean Integration. It needs a data set as input (see DataSet documentation). When loading data from a
file (either HDF5 or ROOT) the function
function LoadMCMCData(path::String, params::Array{String}, range = Colon(), modelname::String = "", dataFormat::DataType = Float64)::DataSet()
can be used.
"""

function hm_integrate(
    result::HMIData{T, I};
    settings::HMISettings = HMIPrecisionSettings())::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

    hm_init(result, settings)

    hm_whiteningtransformation(result, settings)

    hm_createpartitioningtree(result)

    hm_findstartingsamples(result, settings)

    hm_determinetolerance(result, settings)

    hm_hyperrectanglecreation(result, settings)

    hm_integratehyperrectangles(result, settings)


    return result
end

function hm_init(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if result.dataset1.N < result.dataset1.P * 50 || result.dataset2.N < result.dataset2.P
        @log_msg LOG_ERROR "Not enough points for integration"
    end
    @assert result.dataset1.P == result.dataset2.P

    global _global_mt_setting = settings.useMultiThreading

    @log_msg LOG_INFO "Harmonic Mean Integration started. Samples in dataset 1 / 2: \t$(result.dataset1.N) / $(result.dataset2.N)\tParameters:\t$(result.dataset1.P)"
end

function hm_whiteningtransformation(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if !isinitialized(result.whiteningresult)
        @log_msg LOG_INFO "Data Whitening."
        result.whiteningresult = settings.whitening_function(result.dataset1)
    end

    if !result.dataset1.iswhitened
        @log_msg LOG_INFO "Apply Whitening Transformation to Data Set 1"
        transform_data(result.dataset1, result.whiteningresult.whiteningmatrix, result.whiteningresult.meanvalue)
    end
    if !result.dataset2.iswhitened
        @log_msg LOG_INFO "Apply Whitening Transformation to Data Set 2"
        transform_data(result.dataset2, result.whiteningresult.whiteningmatrix, result.whiteningresult.meanvalue)
    end
end

function hm_createpartitioningtree(
    result::HMIData{T, I}) where {T<:AbstractFloat, I<:Integer}

    maxleafsize = 200
    progress_steps = ((!isinitialized(result.dataset1.partitioningtree) ? result.dataset1.N / maxleafsize : 0.0)
                    + (!isinitialized(result.dataset2.partitioningtree) ? result.dataset2.N / maxleafsize : 0.0))
    progressbar = Progress(round(Int64, progress_steps))
    progress_steps > 0 && @log_msg LOG_INFO "Create Space Partitioning Tree"
    !isinitialized(result.dataset1.partitioningtree) && create_search_tree(result.dataset1, progressbar, maxleafsize = maxleafsize)
    !isinitialized(result.dataset2.partitioningtree) && create_search_tree(result.dataset2, progressbar, maxleafsize = maxleafsize)
    finish!(progressbar)
end

function hm_findstartingsamples(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if isempty(result.dataset1.startingIDs)
        @log_msg LOG_INFO "Determine Hyperrectangle Starting Samples for Data Set 1"
        find_hypercube_centers(result.dataset1, result.whiteningresult, settings)
    end
    if isempty(result.dataset2.startingIDs)
        @log_msg LOG_INFO "Determine Hyperrectangle Starting Samples for Data Set 2"
        find_hypercube_centers(result.dataset2, result.whiteningresult, settings)
    end
end

function hm_determinetolerance(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    (iszero(result.dataset1.tolerance) || iszero(result.dataset2.tolerance)) && @log_msg LOG_INFO "Determine Tolerances for Hyperrectangle Creation"

    if iszero(result.dataset1.tolerance)
        suggTolPts = max(result.dataset1.P * 4, ceil(I, sqrt(result.dataset1.N)))
        findtolerance(result.dataset1, min(10, settings.warning_minstartingids), suggTolPts)
        @log_msg LOG_DEBUG "Tolerance Data Set 1: $suggTol"
    end
    if iszero(result.dataset2.tolerance)
        suggTolPts = max(result.dataset2.P * 4, ceil(I, sqrt(result.dataset2.N)))
        findtolerance(result.dataset2, min(10, settings.warning_minstartingids), suggTolPts)
        @log_msg LOG_DEBUG "Tolerance Data Set 2: $suggTol"
    end
end

function hm_hyperrectanglecreation(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    maxPoints::I = 0
    totalpoints::I = 0

    if result.dataset1.isnew || result.dataset2.isnew
        @log_msg LOG_INFO "Create Hyperrectangles using $(nthreads()) thread(s) for Data Set 1: $(result.dataset1.isnew)\t Data Set 2: $(result.dataset2.isnew)"

        thread_volumes1 = Vector{IntegrationVolume{T, I}}(result.dataset1.isnew ? length(result.dataset1.startingIDs) : 0)
        thread_volumes2 = Vector{IntegrationVolume{T, I}}(result.dataset2.isnew ? length(result.dataset2.startingIDs) : 0)

        progressbar = Progress(length(thread_volumes1) + length(thread_volumes2))

        if result.dataset1.isnew
            @log_msg LOG_INFO "Create Hyperrectangles using $(nthreads()) thread(s) for Data Set 1"
            atomic_centerID = Atomic{I}(1)

            @mt MCMCHarmonicMean.hyperrectangle_creationproccess!(result.dataset1, result.whiteningresult.targetprobfactor, settings, thread_volumes1, atomic_centerID, progressbar)

            for i in eachindex(thread_volumes1)
                if isassigned(thread_volumes1, i) == false
                elseif thread_volumes1[i].pointcloud.probfactor == 1.0 || thread_volumes1[i].pointcloud.points < result.dataset1.P * 4
                else
                    push!(result.volumelist1, thread_volumes1[i])
                    maxPoints = max(maxPoints, thread_volumes1[i].pointcloud.points)
                    totalpoints += thread_volumes1[i].pointcloud.points
                end
            end
        end

        if result.dataset2.isnew
            @log_msg LOG_INFO "Create Hyperrectangles using $(nthreads()) thread(s) for Data Set 2"
            atomic_centerID = Atomic{I}(1)

            @mt hyperrectangle_creationproccess!(result.dataset2, result.whiteningresult.targetprobfactor, settings, thread_volumes2, atomic_centerID, progressbar)

            for i in eachindex(thread_volumes2)
                if isassigned(thread_volumes2, i) == false
                elseif thread_volumes2[i].pointcloud.probfactor == 1.0 || thread_volumes2[i].pointcloud.points < result.dataset1.P * 4
                else
                    push!(result.volumelist2, thread_volumes2[i])
                    maxPoints = max(maxPoints, thread_volumes2[i].pointcloud.points)
                    totalpoints += thread_volumes2[i].pointcloud.points
                end
            end
        end

        finish!(progressbar)
    end



    if result.dataset1.isnew
        @log_msg LOG_INFO "Updating $(length(result.volumelist1)) Hyperrectangles of Data Set 1 using $(nthreads()) thread(s)"
        #update pointIDs for each integrationvolume

        @mt_threads for i in eachindex(result.volumelist1)
            update!(result.volumelist1[i], result.dataset2)

            maxPoints = max(maxPoints, result.volumelist1[i].pointcloud.points)
            totalpoints += result.volumelist1[i].pointcloud.points
        end

        result.dataset1.isnew = false
    end

    if result.dataset2.isnew
        @log_msg LOG_INFO "Updating $(length(result.volumelist2)) Hyperrectangles of Data Set 2 using $(nthreads()) thread(s)"
        #update pointIDs for each integrationvolume

        @mt_threads for i in eachindex(result.volumelist2)
            update!(result.volumelist2[i], result.dataset1)
            maxPoints = max(maxPoints, result.volumelist2[i].pointcloud.points)
            totalpoints += result.volumelist2[i].pointcloud.points
        end

        result.dataset1.isnew = false
    end




    #remove rectangles with less than 1% points of the largest rectangle (in terms of points)
    j = length(result.volumelist1)
    for i = 1:length(result.volumelist1)
        if result.volumelist1[j].pointcloud.points < maxPoints * 0.01 || result.volumelist1[j].pointcloud.points < result.dataset1.P * 4
            deleteat!(result.volumelist1, j)
        end
        j -= 1
    end
    j = length(result.volumelist2)
    for i = 1:length(result.volumelist2)
        if result.volumelist2[j].pointcloud.points < maxPoints * 0.01 || result.volumelist2[j].pointcloud.points < result.dataset1.P * 4
            deleteat!(result.volumelist2, j)
        end
        j -= 1
    end
end

function hm_integratehyperrectangles(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}


    nRes = length(result.volumelist1) + length(result.volumelist2)
    offset = length(result.volumelist1)

    if nRes < 1
        @log_msg LOG_ERROR "No hyper-rectangles could be created. Try integration with more points or different settings."
    end

    integrationresults = Array{IntermediateResult, 1}(nRes)

    @log_msg LOG_INFO "Integrating $nRes Hyperrectangles"

    progressbar = Progress(nRes)

    @mt_threads for i in eachindex(result.volumelist1)
        integrationresults[i] = integrate_hyperrectangle(result.dataset2, result.volumelist1[i], result.whiteningresult.determinant)
        lock(BAT.Logging._global_lock) do
            next!(progressbar)
        end
    end

    @mt_threads for i in eachindex(result.volumelist2)
        integrationresults[i + offset] = integrate_hyperrectangle(result.dataset1, result.volumelist2[i], result.whiteningresult.determinant)
        lock(BAT.Logging._global_lock) do
            next!(progressbar)
        end
    end
    finish!(progressbar)



        #remove integrals with no result
#=        j = nRes
        for i = 1:nRes
            try
                if !isassigned(IntResults, j) || isnan(IntResults[j].integral)
                    deleteat!(IntResults, j)
                    if i > cntr
                        deleteat!(result.volumelist2, j - cntr)
                    else
                        deleteat!(result.volumelist, j)
                    end
                end
            catch e
                @log_msg LOG_ERROR string(e)

                if length(IntResults) >= j
                    if i > cntr
                        deleteat!(result.volumelist2, j - cntr)
                    else
                        deleteat!(result.volumelist, j)
                    end
                    deleteat!(IntResults, j)
                end
            finally
                j -= 1
            end
        end
=#

    #Standard Integral Result Combination
    rectweights = zeros(T, nRes)
    if settings.userectweights
        pweights1 = create_pointweights(result.dataset2, result.volumelist1)
        pweights2 = create_pointweights(result.dataset1, result.volumelist2)

        for i in eachindex(result.volumelist1)
            trw = sum(result.dataset2.weights[result.volumelist1[i].pointcloud.pointIDs])
            for id in eachindex(result.volumelist1[i].pointcloud.pointIDs)
                rectweights[i] += 1.0 / trw / pweights1[result.volumelist1[i].pointcloud.pointIDs[id]]# / IntResults[i].error
            end
        end

        for i in eachindex(result.volumelist2)
            trw = sum(result.dataset1.weights[result.volumelist2[i].pointcloud.pointIDs])
            for id in eachindex(result.volumelist2[i].pointcloud.pointIDs)
                rectweights[i + offset] += 1.0 / trw / pweights2[result.volumelist2[i].pointcloud.pointIDs[id]]# / IntResults[i].error
            end
        end
    else
        rectweights = ones(T, nRes)
    end

    rectnorm = sum(rectweights)

    _results = [integrationresults[i].integral for i in eachindex(integrationresults)]
    _points  = [integrationresults[i].points   for i in eachindex(integrationresults)]
    _volumes = [integrationresults[i].volume   for i in eachindex(integrationresults)]

    integral, point, volume = tmean(_results, _points, _volumes, weights = rectweights)
    error = sqrt(var(_results) / rectnorm)
    result.result.integral = HMIEstimate(integral, error, rectweights)

    #Integral Combination using analytic weights
    _errors = [integrationresults[i].integral for i in eachindex(integrationresults)]
    wi = 1.0 ./ _errors

    integral_unc = mean(_results, AnalyticWeights(wi))
    error_unc = sqrt(var(_results, AnalyticWeights(wi), corrected=true))
    result.result.integral_analyticweights = HMIEstimate(integral_unc, error_unc, wi)

    #Integral Combination using a linear fit
    r = [result.volumelist1[i].pointcloud.points / result.dataset2.N for i in eachindex(result.volumelist1)]
    append!(r, [result.volumelist2[i].pointcloud.points / result.dataset1.N for i in eachindex(result.volumelist2)])
    intsub = _results .* r
    χ_sq = (intsub .- (r .* integral)).^2
    probablywrongintegralids = sortperm(χ_sq, rev=true)[1:round(Int64, nRes * 0.2)]

    id = length(probablywrongintegralids)
    for i in probablywrongintegralids
        deleteat!(intsub, id)
        deleteat!(r, id)
        id -= 1
    end
    dataframe = DataFrame(intsub = intsub, r = r)


    fit = glm(@formula(intsub ~ r), dataframe, Normal(), IdentityLink(), wts = ones(length(r)))
    res_fit = fit.model.pp.beta0[1] + fit.model.pp.beta0[2]
    err_fit = stderror(fit.model)[1] + stderror(fit.model)[2]
    result.result.integral_linearfit = HMIEstimate(res_fit, err_fit, ones(length(nRes)))


    wts_fit = wi ./ sum(wi) .* length(wi)
    id = length(probablywrongintegralids)
    for i in probablywrongintegralids
        deleteat!(wts_fit, id)
        id -= 1
    end
    fit = glm(@formula(intsub ~ r), dataframe, Normal(), IdentityLink(), wts = wts_fit)
    res_fit_w = fit.model.pp.beta0[1] + fit.model.pp.beta0[2]
    err_fit_w = stderror(fit.model)[1] + stderror(fit.model)[2]
    result.result.integral_weightedfit = HMIEstimate(res_fit_w, err_fit_w, wts_fit)


    result.result.points = point
    result.result.volume = volume
    result.result.integrals = integrationresults
end

function findtolerance(
    dataset::DataSet{T, I},
    ntestcubes::I,
    pts::I) where {T<:AbstractFloat, I<:Integer}

    ntestpts = [2, 4, 8] * pts
    @log_msg LOG_TRACE "Tolerance Test Cube Points: $([pts, ntestpts...])"

    vInc = Vector{T}(ntestcubes * (length(ntestpts) + 1))
    pInc = Vector{T}(ntestcubes * (length(ntestpts) + 1))

    startingIDs = dataset.startingIDs

    cntr = 1
    for id = 1:ntestcubes
        c = find_density_test_cube(dataset.data[:, startingIDs[id]], dataset, pts)
        prevv = c[1]^dataset.P
        prevp = c[2].pointcloud.points
        for i in ntestpts
            c = find_density_test_cube(dataset.data[:, startingIDs[id]], dataset, i)
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
    suggTol = (suggTol - 1) * 4 + 1

    dataset.tolerance = suggTol
end


function hyperrectangle_creationproccess!(
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings,
    integrationvolumes::Vector{IntegrationVolume{T, I}},
    atomic_centerID::Atomic{I},
    progressbar::Progress) where {T<:AbstractFloat, I<:Integer}

    while true
        #get new starting sample
        idc = atomic_add!(atomic_centerID, 1)
        if idc > length(dataset.startingIDs)
            break
        end
        id = dataset.startingIDs[idc]

        #update progress bar
        lock(BAT.Logging._global_lock) do
            next!(progressbar)
        end

        #create hyper-rectangle
        integrationvolumes[idc] = create_hyperrectangle(id, dataset, targetprobfactor, settings)

        @log_msg LOG_DEBUG "Hyperrectangle created. Points:\t$(integrationvolumes[idc].pointcloud.points)\tVolume:\t$(integrationvolumes[idc].volume)\tProb. Factor:\t$(integrationvolumes[idc].pointcloud.probfactor)"
    end
end

"""
    create_hyperrectangle{T<:AbstractFloat, I<:Integer}(Mode::Vector{T}, dataset::DataSet{T, I}, volumes::Vector{IntegrationVolume{T, I}}, whiteningresult::WhiteningResult{T},
        Tolerance::T, settings::HMISettings)::IntegrationVolume{T, I}

This function tries to create
 a hyper-rectangle around a starting point. It builds a cube first and adapts each face of individually to fit to the data as good as possible.
If the creation process fails a rectangle with no or only one point might be returned (check probfactor). The creation process might also be stopped because the rectangle overlaps
with another.
"""
function create_hyperrectangle(
    id::I,
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings)::IntegrationVolume{T, I} where {T<:AbstractFloat, I<:Integer}

    edgelength::T = 1.0

    Mode = dataset.data[:, id]
    Tolerance = dataset.tolerance

    cube = HyperCubeVolume(Mode, edgelength)
    vol = IntegrationVolume(dataset, cube, true)

    while vol.pointcloud.points > 0.01 * dataset.N
        edgelength *= 0.5^(1/dataset.P)
        HyperCubeVolume!(cube, Mode, edgelength)
        IntegrationVolume!(vol, dataset, cube, true)
    end
    tol = 1.0
    step = 0.7
    direction = 0
    PtsIncrease = 0.0
    VolIncrease = 1.0

    it = 0
    while vol.pointcloud.probfactor < targetprobfactor / tol || vol.pointcloud.probfactor > targetprobfactor
        tol += 0.01 * it
        it += 1

        if vol.pointcloud.probfactor > targetprobfactor
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
        IntegrationVolume!(vol, dataset, cube, true)

        PtsIncrease = vol.pointcloud.points / PtsIncrease

        if vol.pointcloud.points > 0.01 * dataset.N && vol.pointcloud.probfactor < targetprobfactor
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

    const min_points = 5
    const max_iterations_per_dim = 20

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
                resize_integrationvol!(newvol, vol, dataset, p, spvol, false, searchvol)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probfactor < targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc) && change1 != -1
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
                    resize_integrationvol!(newvol, vol, dataset, p, spvol, false, searchvol)

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
                elseif vol.pointcloud.points < 100 && newvol.pointcloud.probfactor < targetprobfactor && PtsIncrease > 1.0 && change1 != -1
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
                resize_integrationvol!(newvol, vol, dataset, p, spvol, false, searchvol)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probfactor < targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc) && change2 != -1
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
                    resize_integrationvol!(newvol, vol, dataset, p, spvol, false, searchvol)

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
                elseif vol.pointcloud.points < 100 && newvol.pointcloud.probfactor < targetprobfactor && PtsIncrease > 1.0 && change2 != -1
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


    res = search(dataset, vol.spatialvolume, true)
    resize!(vol.pointcloud.pointIDs, res.points)
    copy!(vol.pointcloud.pointIDs, res.pointIDs)
    vol.pointcloud.points = res.points
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
    determinant::T)::IntermediateResult{T} where {T<:AbstractFloat, I<:Integer}

    s::T = 0.0
    count::T = 0.0
    for i in integrationvol.pointcloud.pointIDs
        #for numerical stability
        prob = (dataset.logprob[i] - integrationvol.pointcloud.maxLogProb)
        s += 1.0 / exp(prob) * dataset.weights[i]
        count += dataset.weights[i]
    end

    totalWeight::T = sum(dataset.weights)
    integral::T = totalWeight * integrationvol.volume / s / determinant / exp(-integrationvol.pointcloud.maxLogProb)

    r = integrationvol.pointcloud.points / dataset.N
    error::T = calculateuncertainty(dataset, integrationvol, determinant, integral * r)

    @log_msg LOG_DEBUG "Integral Estimate: $integral ± $error"

    return IntermediateResult(integral, error, T(integrationvol.pointcloud.points), integrationvol.volume)
end
