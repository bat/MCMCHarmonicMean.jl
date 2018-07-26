# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

#global multithreading setting
_global_mt_setting = true

"""
macro mt(expr)

dependent on the global multithreading setting this macro dynamically evaluates code either multit-hreaded or single-threaded
"""
macro mt(expr)
    quote
        if _global_mt_setting
            @everythread $(esc(expr))
        else
            $(esc(expr))
        end
    end
end

"""
macro mt_threads(expr)

dependent on the global multithreading setting this macro dynamically evaluates a for loop either multit-hreaded or single-threaded
"""
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
function hm_integrate(
    result::HMIData{T, I};
    settings::HMISettings = HMIPrecisionSettings())::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

This function starts the adaptive harmonic mean integration program.
It needs a HMIData struct as input, which can be either filled using BAT.jl samples or by using a DataSet for custom samples. (see DataSet documentation).
When loading data from a file (either HDF5 or ROOT) the function LoadMCMCData can be used.
"""
function hm_integrate(
    result::HMIData{T, I};
    settings::HMISettings = HMIPrecisionSettings())::HMIData{T, I} where {T<:AbstractFloat, I<:Integer}

    #time: <1ms, 3KB
    hm_init(result, settings)

    #time: 20ms, 13 MB
    hm_whiteningtransformation(result, settings)

    #time: 19ms, 18 MB
    hm_createpartitioningtree(result)


    notsinglemode = hm_findstartingsamples(result, settings)

    #time: 223ms, 104MB
    if !notsinglemode
        result.dataset1.tolerance = Inf
        result.dataset2.tolerance = Inf
        @log_msg LOG_WARNING "Tolerance set to Inf for single mode distributions"
    else
        hm_determinetolerance(result, settings)
    end
    #time 850ms, 266 MB
    hm_hyperrectanglecreation(result, settings)

    #time 125ms, 33MB
    hm_integratehyperrectangles(result, settings)


    return result
end

"""
function hm_init(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

Correctly sets to global multithreading setting and ensures that a minimum number of samples when accounting for the number of dimensions are provided.
"""
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

"""
function hm_whiteningtransformation(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

Applies a whitening transformation to the samples. A custom whitening method used can be chosen by replacing the default whitening function (Cholesky)
in the HMISettings struct.
"""
function hm_whiteningtransformation(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if !isinitialized(result.whiteningresult)
        @log_msg LOG_INFO "Data Whitening."
        result.whiteningresult = settings.whitening_function(result.dataset1)
    end

    if !result.dataset1.iswhitened
        @log_msg LOG_INFO "Apply Whitening Transformation to Data Set 1"
        transform_data(result.dataset1, result.whiteningresult.whiteningmatrix, result.whiteningresult.meanvalue, false)
    end
    if !result.dataset2.iswhitened
        @log_msg LOG_INFO "Apply Whitening Transformation to Data Set 2"
        transform_data(result.dataset2, result.whiteningresult.whiteningmatrix, result.whiteningresult.meanvalue, true)
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

    notsinglemode = true
    @log_msg LOG_INFO "Determine Hyperrectangle Starting Samples"
    if isempty(result.dataset1.startingIDs)
        notsinglemode &= find_hypercube_centers(result.dataset1, result.whiteningresult, settings)
    end
    if isempty(result.dataset2.startingIDs)
        notsinglemode &= find_hypercube_centers(result.dataset2, result.whiteningresult, settings)
    end

    notsinglemode
end

function hm_determinetolerance(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    (iszero(result.dataset1.tolerance) || iszero(result.dataset2.tolerance)) && @log_msg LOG_INFO "Determine Tolerances for Hyperrectangle Creation"

    if iszero(result.dataset1.tolerance)
        suggTolPts = max(result.dataset1.P * 4, ceil(I, sqrt(result.dataset1.N)))
        findtolerance(result.dataset1, min(10, settings.warning_minstartingids), suggTolPts)
        @log_msg LOG_DEBUG "Tolerance Data Set 1: $(result.dataset1.tolerance)"
    end
    if iszero(result.dataset2.tolerance)
        suggTolPts = max(result.dataset2.P * 4, ceil(I, sqrt(result.dataset2.N)))
        findtolerance(result.dataset2, min(10, settings.warning_minstartingids), suggTolPts)
        @log_msg LOG_DEBUG "Tolerance Data Set 2: $(result.dataset2.tolerance)"
    end
end

function hm_hyperrectanglecreation(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if isempty(result.volumelist1) || isempty(result.volumelist2)
        nvols = (isempty(result.volumelist1) ? length(result.dataset1.startingIDs) : 0) +
                (isempty(result.volumelist2) ? length(result.dataset2.startingIDs) : 0)

        @log_msg LOG_INFO "Create $nvols Hyperrectangles using $(_global_mt_setting ? nthreads() : 1) thread(s)"
        progressbar = Progress(nvols)

        isempty(result.volumelist1) && hm_hyperrectanglecreation_dataset(result.dataset1, result.volumelist1, result.cubelist1, result.whiteningresult.targetprobfactor, progressbar, settings)
        isempty(result.volumelist2) && hm_hyperrectanglecreation_dataset(result.dataset2, result.volumelist2, result.cubelist2, result.whiteningresult.targetprobfactor, progressbar, settings)

        finish!(progressbar)
    end


    if result.dataset1.isnew || result.dataset2.isnew
        nvols = (result.dataset1.isnew ? length(result.volumelist1) : 0) +
                (result.dataset2.isnew ? length(result.volumelist2) : 0)

        @log_msg LOG_INFO "Updating $nvols Hyperrectangles of Data Set 1 using $(_global_mt_setting ? nthreads() : 1) thread(s)"
        progressbar = Progress(nvols)

        result.dataset2.isnew && hm_updatehyperrectangles_dataset(result.dataset2, result.volumelist1, progressbar)
        result.dataset1.isnew && hm_updatehyperrectangles_dataset(result.dataset1, result.volumelist2, progressbar)

        finish!(progressbar)
    end
end

function hm_hyperrectanglecreation_dataset(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I}, 1},
    cubes::Array{HyperRectVolume{T}, 1},
    targetprobfactor::T,
    progressbar::Progress,
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}


    maxPoints::I = 0
    totalpoints::I = 0

    thread_volumes = Vector{IntegrationVolume{T, I}}(length(dataset.startingIDs))
    thread_cubes = Vector{HyperRectVolume{T}}(length(dataset.startingIDs))

    atomic_centerID = Atomic{I}(1)
    @mt MCMCHarmonicMean.hyperrectangle_creationproccess!(dataset, targetprobfactor, settings, thread_volumes, thread_cubes, atomic_centerID, progressbar)

    for i in eachindex(thread_volumes)
        if isassigned(thread_volumes, i) == false
        elseif thread_volumes[i].pointcloud.probfactor == 1.0 || thread_volumes[i].pointcloud.points < dataset.P * 4
        else
            push!(volumes, thread_volumes[i])
            push!(cubes, thread_cubes[i])
            maxPoints = max(maxPoints, thread_volumes[i].pointcloud.points)
            totalpoints += thread_volumes[i].pointcloud.points
        end
    end

    dataset.isnew = true
end

function hm_updatehyperrectangles_dataset(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I}, 1},
    progressbar::Progress) where {T<:AbstractFloat, I<:Integer}

    maxPoints = zero(T)

    @mt_threads for i in eachindex(volumes)
        update!(volumes[i], dataset)

        maxPoints = max(maxPoints, volumes[i].pointcloud.points)

        lock(BAT.Logging._global_lock) do
            next!(progressbar)
        end
    end

        #remove rectangles with less than 1% points of the largest rectangle (in terms of points)
    j = length(volumes)
    for i = 1:length(volumes)
        if volumes[j].pointcloud.points < maxPoints * 0.01 || volumes[j].pointcloud.points < dataset.P * 4
            deleteat!(volumes, j)
        end
        j -= 1
    end
    dataset.isnew = false
end

function hm_integratehyperrectangles(
    result::HMIData{T, I},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    nRes = length(result.volumelist1) + length(result.volumelist2)
    @log_msg LOG_INFO "Integrating $nRes Hyperrectangles"

    progressbar = Progress(nRes)

    result.integrals1, result.rejectedrects1 = hm_integratehyperrectangles_dataset(result.volumelist1, result.dataset2, result.whiteningresult.determinant, progressbar, settings)
    result.integrals2, result.rejectedrects2 = hm_integratehyperrectangles_dataset(result.volumelist2, result.dataset1, result.whiteningresult.determinant, progressbar, settings)

    finish!(progressbar)

    rectweights = [result.integrals1.weights_overlap..., result.integrals2.weights_overlap...]
    allintegrals = [result.integrals1.integrals..., result.integrals2.integrals...]
    i_std = mean(allintegrals, ProbabilityWeights(rectweights))
    result.integral_standard = HMIEstimate(i_std, sqrt(var(allintegrals)) / sum(rectweights), rectweights)

    σtot = sqrt(result.integrals1.σ + result.integrals2.σ)
    covweights = [result.integrals1.weights_cov..., result.integrals2.weights_cov...]
#=
    covweights1 = zeros(length(result.integrals1.integrals))
    covweights2 = zeros(length(result.integrals2.integrals))
    for i in eachindex(result.integrals1.integrals)
        σinv = inv(result.integrals1.Σ)
        for k in eachindex(result.integrals1.integrals)
            covweights1[i] += σinv[i, k]
        end
        covweights1[i] /= sum(σinv)
    end
    for i in eachindex(result.integrals2.integrals)
        σinv = inv(result.integrals2.Σ)
        for k in eachindex(result.integrals2.integrals)
            covweights2[i] += σinv[i, k]
        end
        covweights2[i] /= sum(σinv)
    end
    covweights = [covweights1..., covweights2...]
=#
    i_cov = mean(allintegrals, AnalyticWeights(covweights))
    result.integral_covweighted = HMIEstimate(i_cov, σtot, covweights)
end

function hm_integratehyperrectangles_dataset(
    volumes::Array{IntegrationVolume{T, I}},
    dataset::DataSet{T, I},
    determinant::T,
    progressbar::Progress,
    settings::HMISettings)::Tuple{IntermediateResults{T}, Vector{I}} where {T<:AbstractFloat, I<:Integer}

    if length(volumes) < 1
        @log_msg LOG_ERROR "No hyper-rectangles could be created. Try integration with more points or different settings."
    end

    integralestimates = IntermediateResults(T, length(volumes))
    integralestimates.Z = Array{T, 2}(dataset.nsubsets, length(volumes))

    pweights = create_pointweights(dataset, volumes)
    integralestimates.weights_overlap::Array{T, 1} = zeros(T, length(volumes))

    @mt_threads for i in eachindex(volumes)
        integralestimates.Z[:, i], integralestimates.integrals[i] = integrate_hyperrectangle_cov(dataset, volumes[i], determinant)
        #integralestimates.integrals[i] = 1.0 / mean(view(integralestimates.Z, :, i))

        trw = sum(dataset.weights[volumes[i].pointcloud.pointIDs])
        for id in eachindex(volumes[i].pointcloud.pointIDs)
            integralestimates.weights_overlap[i] += 1.0 / trw / pweights[volumes[i].pointcloud.pointIDs[id]]
        end

        lock(BAT.Logging._global_lock) do
            next!(progressbar)
        end
    end

    rejectedids = trim(integralestimates)

    @log_msg LOG_TRACE "Rectangle weights: $(integralestimates.weights_overlap))"

    integralestimates.Σ = cov(integralestimates.Z)
    integralestimates.weights_cov = 1 ./ diag(integralestimates.Σ)

    integral_coeff = ones(T, length(integralestimates)) ./ length(integralestimates) ./ dataset.nsubsets #.* integralestimates.integrals.^2
    integralestimates.σ = transpose(integral_coeff) * integralestimates.Σ * integral_coeff

    @log_msg LOG_DEBUG "Covariance σ: $(integralestimates.σ)"

    return integralestimates, rejectedids
end


"""
function findtolerance(
    dataset::DataSet{T, I},
    ntestcubes::I,
    pts::I) where {T<:AbstractFloat, I<:Integer}

This function calculates the parameter λ, which describes the expectation on the number of samples,
that are expected to be added to the hyper-cube in advance before making the change. If the expectation is met,
i.e. sufficient samples are added, the the volume change is performed during the hyper-rectangle creation process.
"""
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
    suggTol::T = length(tols) < 4 ? 1.5 : mean(trim(tols))
    suggTol = (suggTol - 1) * 4 + 1

    dataset.tolerance = suggTol
end

"""
function hyperrectangle_creationproccess!(
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings,
    integrationvolumes::Vector{IntegrationVolume{T, I}},
    atomic_centerID::Atomic{I},
    progressbar::Progress) where {T<:AbstractFloat, I<:Integer}

This function assigns each thread its own hyper-rectangle to build, if in multithreading-mode.
"""
function hyperrectangle_creationproccess!(
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings,
    integrationvolumes::Vector{IntegrationVolume{T, I}},
    cubevolumes::Vector{HyperRectVolume{T}},
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
        integrationvolumes[idc], cubevolumes[idc] = create_hyperrectangle(id, dataset, targetprobfactor, settings)

        @log_msg LOG_DEBUG "Hyperrectangle created. Points:\t$(integrationvolumes[idc].pointcloud.points)\tVolume:\t$(integrationvolumes[idc].volume)\tProb. Factor:\t$(integrationvolumes[idc].pointcloud.probfactor)"
    end
end

"""
function create_hyperrectangle(
    id::I,
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings)::IntegrationVolume{T, I} where {T<:AbstractFloat, I<:Integer}

This function tries to create a hyper-rectangle around each starting sample.
It starts by building a hyper-cube first and subsequently adapts each face individually,
thus turning the hyper-cube into a hyper-rectangle.
The faces are adjusted in a way to match the shape of the distribution as best as possible.
"""
function create_hyperrectangle(
    id::I,
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings)::Tuple{IntegrationVolume{T, I}, HyperRectVolume{T}} where {T<:AbstractFloat, I<:Integer}

    edgelength::T = 1.0

    Mode = dataset.data[:, id]

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

    it = 0
    while vol.pointcloud.probfactor < targetprobfactor / tol || vol.pointcloud.probfactor > targetprobfactor
        tol += 0.01 * it
        it += 1

        if vol.pointcloud.probfactor > targetprobfactor
            #decrease side length
            edgelength *= step

            step = adjuststepsize!(step, direction == -1)
            direction = -1
        else
            #increase side length
            edgelength /= step

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

    finalcube = deepcopy(cube)
    @log_msg LOG_TRACE "Starting Hypercube Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)"


    wasCubeChanged = true
    newvol::IntegrationVolume{T, I} = deepcopy(vol)

    ptsTolInc::T = dataset.tolerance
    ptsTolDec::T = dataset.tolerance * 1.1

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

    return vol, finalcube
end

@inline function adjuststepsize!(Step, Increase::Bool)
    if Increase
        return Step * 0.5
    else
        return Step * 2.0
    end
end

function integrate_hyperrectangle_cov(
    dataset::DataSet{T, I},
    integrationvol::IntegrationVolume{T, I},
    determinant::T)::Tuple{Array{T, 1}, T} where {T<:AbstractFloat, I<:Integer}

    norm_const = Array{T, 1}(dataset.nsubsets)
    totalWeights = zeros(T, dataset.nsubsets)

    for i in eachindex(dataset.weights)
        totalWeights[dataset.ids[i]] += dataset.weights[i]
    end

    s = zeros(T, dataset.nsubsets)
    s_sum = T(0)
    nsamples = zeros(T, dataset.nsubsets)
    for id in integrationvol.pointcloud.pointIDs
        prob = dataset.logprob[id] - integrationvol.pointcloud.maxLogProb
        s[dataset.ids[id]] += T(1) / exp(prob) * dataset.weights[id]
        s_sum += T(1) / exp(prob) * dataset.weights[id]
        nsamples[dataset.ids[id]] += T(1)
    end

    for n = 1:dataset.nsubsets
        norm_const[n] = totalWeights[n] * integrationvol.volume / determinant / exp(-integrationvol.pointcloud.maxLogProb)
    end

    integral::T = sum(dataset.weights) * integrationvol.volume / s_sum / determinant / exp(-integrationvol.pointcloud.maxLogProb)
    integrals_batches = norm_const ./ s

    #replace INF entries with mean integral value
    for i in eachindex(integrals_batches)
        if isnan(integrals_batches[i]) || isinf(integrals_batches[i])
            integrals_batches[i] = integral
        end
    end

    return integrals_batches, integral
end
