function hm_create_integrationvolumes!(
    result::HMIData{T, I, HyperRectVolume{T}},
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}

    if isempty(result.volumelist1) || isempty(result.volumelist2)
        nvols = (isempty(result.volumelist1) ? length(result.dataset1.startingIDs) : 0) +
                (isempty(result.volumelist2) ? length(result.dataset2.startingIDs) : 0)

        @info "Create $nvols Hyperrectangles using $(_global_mt_setting ? nthreads() : 1) thread(s)"
        progressbar = Progress(nvols)

        isempty(result.volumelist1) && hm_create_integrationvolumes_dataset!(
            result.dataset1, result.volumelist1, result.cubelist1, result.whiteningresult.targetprobfactor, progressbar, settings)
        isempty(result.volumelist2) && hm_create_integrationvolumes_dataset!(
            result.dataset2, result.volumelist2, result.cubelist2, result.whiteningresult.targetprobfactor, progressbar, settings)

        finish!(progressbar)
    end


    nvols = (result.dataset1.isnew ? length(result.volumelist1) : 0) +
            (result.dataset2.isnew ? length(result.volumelist2) : 0)

    if nvols > 0
        @info "Updating $nvols Hyperrectangles of Data Set 1 using $(_global_mt_setting ? nthreads() : 1) thread(s)"
        progressbar = Progress(nvols)

        result.dataset2.isnew && hm_update_integrationvolumes_dataset!(result.dataset2, result.volumelist1, progressbar)
        result.dataset1.isnew && hm_update_integrationvolumes_dataset!(result.dataset1, result.volumelist2, progressbar)

        finish!(progressbar)
    end
end

function hm_create_integrationvolumes_dataset!(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I, HyperRectVolume{T}}, 1},
    cubes::Array{HyperRectVolume{T}, 1},
    targetprobfactor::T,
    progressbar::Progress,
    settings::HMISettings) where {T<:AbstractFloat, I<:Integer}


    maxPoints::I = 0
    totalpoints::I = 0

    thread_volumes = Vector{IntegrationVolume{T, I, HyperRectVolume{T}}}(undef, length(dataset.startingIDs))
    thread_cubes = Vector{HyperRectVolume{T}}(undef, length(dataset.startingIDs))

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

function hm_update_integrationvolumes_dataset!(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I, HyperRectVolume{T}}, 1},
    progressbar::Progress) where {T<:AbstractFloat, I<:Integer}

    maxPoints = zero(T)

    @mt for i in threadpartition(eachindex(volumes), mt_nthreads())
        update!(volumes[i], dataset)

        maxPoints = max(maxPoints, volumes[i].pointcloud.points)

        lock(_global_lock) do
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


"""
This function assigns each thread its own hyper-rectangle to build, if in multithreading-mode.
"""
function hyperrectangle_creationproccess!(
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings,
    integrationvolumes::Vector{IntegrationVolume{T, I, HyperRectVolume{T}}},
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
        lock(_global_lock) do
            next!(progressbar)
        end

        #create hyper-rectangle
        integrationvolumes[idc], cubevolumes[idc] = create_hyperrectangle(id, dataset, targetprobfactor, settings)

        #@debug "Hyperrectangle created. Points:\t$(integrationvolumes[idc].pointcloud.points)\tVolume:\t$(integrationvolumes[idc].volume)\tProb. Factor:\t$(integrationvolumes[idc].pointcloud.probfactor)"
    end
end

"""
This function creates a hyper-rectangle around each starting sample.
It starts by building a hyper-cube  and subsequently adapts each face individually,
thus turning the hyper-cube into a hyper-rectangle.
The faces are adjusted in a way to match the shape of the distribution as best as possible.
"""
function create_hyperrectangle(
    id::I,
    dataset::DataSet{T, I},
    targetprobfactor::T,
    settings::HMISettings)::Tuple{IntegrationVolume{T, I, HyperRectVolume{T}}, HyperRectVolume{T}} where {T<:AbstractFloat, I<:Integer}

    edgelength::T = 1.0

    Mode = dataset.data[:, id]

    cube = create_hypercube(Mode, edgelength)
    vol = IntegrationVolume(dataset, cube, true)

    while vol.pointcloud.points > 0.01 * dataset.N
        edgelength *= 0.5^(1/dataset.P)
        modify_hypercube!(cube, Mode, edgelength)
        modify_integrationvolume!(vol, dataset, cube, true)
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
        modify_hypercube!(cube, Mode, edgelength)
        modify_integrationvolume!(vol, dataset, cube, true)

        PtsIncrease = vol.pointcloud.points / PtsIncrease

        if vol.pointcloud.points > 0.01 * dataset.N && vol.pointcloud.probfactor < targetprobfactor
            break
        end
    end

    finalcube = deepcopy(cube)
    #@log_msg LOG_TRACE "Starting Hypercube Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)"


    wasCubeChanged = true
    newvol::IntegrationVolume{T, I} = deepcopy(vol)

    ptsTolInc::T = dataset.tolerance
    ptsTolDec::T = dataset.tolerance * 1.1

    spvol::HyperRectVolume{T} = deepcopy(vol.spatialvolume)
    searchvol::HyperRectVolume{T} = deepcopy(spvol)

    increase_default = settings.rect_increase
    increase = increase_default
    decrease = 1.0 - 1.0 / (1.0 + increase)

    min_points = 5
    max_iterations_per_dim = 20

    while wasCubeChanged && vol.pointcloud.probfactor > 1.0

        if vol.pointcloud.points * increase < min_points
            increase *= ceil(I, min_points / (vol.pointcloud.points * increase))
            decrease = 1.0 - 1.0 / (1.0 + increase)
            #@log_msg LOG_TRACE "Changed increase to $increase"
        elseif increase > increase_default && vol.pointcloud.points * increase_default > 2 * min_points
            increase = increase_default
            decrease = 1.0 - 1.0 / (1.0 + increase)
            #@log_msg LOG_TRACE "Reset increase to $increase"
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
                    #@log_msg LOG_TRACE "lo inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
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
                        #@log_msg LOG_TRACE "lo dec p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
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
                    #@log_msg LOG_TRACE "(sr) lo inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
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
                    #@log_msg LOG_TRACE "up inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
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
                        #@log_msg LOG_TRACE "up dec p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
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
                    #log_msg LOG_TRACE "(sr) up inc p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)\tPtsIncrease=$PtsIncrease"
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
    copyto!(vol.pointcloud.pointIDs, res.pointIDs)
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
