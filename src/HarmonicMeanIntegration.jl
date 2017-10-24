# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

"""
    hm_integrate(Data::DataSet, settings::HMIntegrationSettings = HMIntegrationSettings())::IntegrationResult

This function starts the Harmonic Mean Integration. It needs a data set as input (see DataSet documentation). When loading data from a
file (either HDF5 or ROOT) the function
function LoadMCMCData(path::String, params::Array{String}, range = Colon(), modelname::String = "", dataFormat::DataType = Float64)::DataSet()
can be used.
"""
function hm_integrate(bat_input::DensitySampleVector)
    T = typeof(bat_input.params[1,1])
    hm_integrate(DataSet(
        convert(Array{T, 2}, bat_input.params),
        convert(Array{T, 1}, bat_input.log_value),
        convert(Array{Float64, 1}, bat_input.weight)))
end

function hm_integrate(dataset::DataSet, settings::HMIntegrationSettings = HMIntegrationStandardSettings())::IntegrationResult
    if dataset.N < dataset.P * 50
        error("Not enough points for integration")
    end

    LogHigh("Integration started. Data Points:\t$(dataset.N)\tParameters:\t$(dataset.P)")
    LogHigh("Data Whitening")

    local whiteningresult::WhiteningResult
    if settings.whitening_method == :CholeskyWhitening
        whiteningresult = cholesky_whitening(dataset)
    elseif settings.whitening_method == :StatisticalWhitening
        whiteningresult = statistical_whitening(dataset)
    elseif settings.whitening_method == :NoWhitening
        whiteningresult = no_whitening(dataset)
    else
        error("Unknown whitening method. Use :CholeskyWhitening or :StatisticalWhitening")
    end

    datatree = create_search_tree(dataset)

    LogHigh("Find possible Hyperrectangle Centers")
    centerIDs = find_hypercube_centers(dataset, datatree, whiteningresult, settings)
    volumes = Vector{IntegrationVolume}(0)

    LogHigh("Find good tolerances for Hyperrectangle Creation")

    vInc = zeros(0)
    pInc = zeros(0)
    pts = 10 * dataset.P
    for id = 1:10
        c = find_density_test_cube(dataset.data[:, centerIDs[id]], datatree, pts)
        prevv = c[1]^dataset.P
        prevp = c[2].pointcloud.points
        for i in [2, 3, 4, 6, 8] * pts
            c = find_density_test_cube(dataset.data[:, centerIDs[id]], datatree, i)
            v = c[1]^dataset.P
            p = c[2].pointcloud.points
            if v/prevv/p*prevp < 1
                prevv = v
                prevp = p
                continue
            end
            append!(vInc, v/prevv)
            append!(pInc, p/prevp)
            prevv = v
            prevp = p
        end
    end

    tols = Array{Float64, 1}(length(vInc))
    for i in eachindex(tols)
        tols[i] = vInc[i] / pInc[i]
    end
    sort!(tols, rev=false)
    suggTol = mean(tols[1:round(Int64, length(tols) * 0.5)])


    suggTol = sqrt(suggTol - 1) + 1
    LogHigh("Sugg. Tolerance: $suggTol")


    iterations = 0
    iterations_skipped = 0

    LogHigh("Create Hyperrectangles")
    maxPoints = 0
    totalpoints = 0


    @showprogress for i in centerIDs
        #check if id is inside an existing Hyperrectangle
        insideRectangle = false
        for vol in volumes
            if in(i, vol.pointcloud.pointIDs)
                iterations_skipped += 1
                insideRectangle = true
                break
            end
        end
        if insideRectangle
            continue
        end
        iterations += 1

        vol = create_hyperrectangle(dataset.data[:, i], datatree, volumes, whiteningresult, suggTol, settings)


        isRectInsideAnother = false

        if !settings.use_all_rects
            for other in volumes
                overlap = calculate_overlapping(vol, other.spatialvolume)
                if overlap > 0.75
                    isRectInsideAnother = true
                    break
                end
            end
        end

        if isRectInsideAnother
            iterations_skipped += 1
            LogLow("Hyperrectangle (ID = $i) discarded because it was inside another Rectangle.")
        elseif vol.pointcloud.probfactor == 1.0 || vol.pointcloud.points < dataset.P * 4
            LogLow("Hyperrectangle (ID = $i ) discarded because it has not enough Points.\tPoints:\t$(vol.pointcloud.points)\tProb. Factor:\t$(vol.pointcloud.probfactor)")
        else
            push!(volumes, vol)
            totalpoints += vol.pointcloud.points


            LogMedium("$(length(volumes)). Hyperrectangle (ID = $i ) found. Rectangles discarded: $iterations_skipped\tPoints:\t$(vol.pointcloud.points)\tProb. Factor:\t$(vol.pointcloud.probfactor)")
            if vol.pointcloud.points > maxPoints
                maxPoints = vol.pointcloud.points
            end
        end

        if settings.stop_ifenoughpoints && totalpoints > 0.5 * dataset.N
            LogMedium("hyper-rectangle creation process stopped because the created hyper-rectangles have already enough points")
            break
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


    LogHigh("Integrate Hyperrectangle")

    nRes = length(volumes)
    IntResults = Array{IntermediateResult, 1}(nRes)

    @showprogress for i in 1:nRes
        IntResults[i] = integrate_hyperrectangle(datatree, dataset, volumes[i], whiteningresult.determinant)
        LogMedium("$i. Integral: $(IntResults[i].integral)\tVolume\t$(IntResults[i].volume)\tPoints:\t$(IntResults[i].points)")
    end

    #remove integrals with no result
    j = nRes
    for i = 1:nRes
        if isnan(IntResults[j].integral)
            deleteat!(IntResults, j)
        end
        j -= 1
    end
    nRes = length(IntResults)

    rectweights = ones(nRes)
    rectnorm = nRes

    if settings.use_all_rects
        pweights = create_pointweights(dataset, [volumes[i].pointcloud for i in eachindex(volumes)])
        for i in eachindex(volumes)
            cloud = volumes[i]
            rweight = 0.0
            for p in cloud.pointcloud.pointIDs
                rweight += 1.0 / pweights[p]
            end
            rectweights[i] = rweight
        end
        rectnorm = sum(rectweights)
    end

    result = 0.0
    error = 0.0
    points = 0.0
    volume = 0.0
    #normerror = 0.0
    for i = 1:nRes
        result += IntResults[i].integral * rectweights[i] / rectnorm
        points += IntResults[i].points * rectweights[i] / rectnorm
        volume += IntResults[i].volume * rectweights[i] / rectnorm
    end
    #result /= normerror
    for i in 1:nRes
        error += (IntResults[i].integral - result)^2
    end
    error = sqrt(error / (nRes - 1))
    if nRes == 1
        error = IntResults[1].error
    end

    LogHigh("Integration Result:\t $result +- $error\nRectangles created: $(nRes)\tavg. points used: $points\t avg. volume: $volume")
    return IntegrationResult(result, error, nRes, points, volume, volumes, centerIDs, whiteningresult)
end


"""
    function create_hyperrectangle{T<:Real}(Mode::Vector{T}, datatree::Tree{T}, volumes::Vector{IntegrationVolume}, whiteningresult::WhiteningResult, Tolerance::Float64)::IntegrationVolume

This function tries to create a hyper-rectangle around a starting point. It builds a cube first and adapts each face of individually to fit to the data as good as possible.
If the creation process fails a rectangle with no or only one point might be returned (check probfactor). The creation process might also be stopped because the rectangle overlaps
with another.
"""

function create_hyperrectangle{T<:Real}(Mode::Vector{T}, datatree::Tree{T}, volumes::Vector{IntegrationVolume}, whiteningresult::WhiteningResult,
        Tolerance::Float64, settings::HMIntegrationSettings)::IntegrationVolume

    edgelength = 1.0

    cube = HyperCubeVolume(Mode, edgelength)
    vol = IntegrationVolume(datatree, cube, true)

    while vol.pointcloud.points > 0.01 * datatree.N
        edgelength *= 0.5
        HyperCubeVolume!(cube, Mode, edgelength)
        IntegrationVolume!(vol, datatree, cube, true)
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
            VolIncrease = edgelength^datatree.P
            edgelength *= step
            VolIncrease = edgelength^datatree.P / VolIncrease

            step = adjuststepsize!(step, direction == -1)
            direction = -1
        else
            #increase side length
            VolIncrease = edgelength^datatree.P
            edgelength /= step
            VolIncrease = edgelength^datatree.P / VolIncrease

            step = adjuststepsize!(step, direction == 1)
            direction = 1
        end
        PtsIncrease = vol.pointcloud.points
        HyperCubeVolume!(cube, Mode, edgelength)
        IntegrationVolume!(vol, datatree, cube ,true)

        PtsIncrease = vol.pointcloud.points / PtsIncrease

        if vol.pointcloud.points > 0.01 * datatree.N && vol.pointcloud.probfactor < whiteningresult.targetprobfactor
            break
        end
    end



    LogLow("\tTEST Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")


    wasCubeChanged = true

    ptsTolInc = Tolerance
    ptsTolDec = Tolerance + (Tolerance - 1) * 1.5

    dimensionsFinished = falses(datatree.P)
    spvol = deepcopy(vol.spatialvolume)
    buffer = 0.0

    increase = settings.rect_increase
    decrease = 1.0 - 1.0 / (1.0 + increase)

    while wasCubeChanged && vol.pointcloud.probfactor > 1.0

        wasCubeChanged = false

        isRectInsideAnother = false

        if settings.use_all_rects
            for other in volumes
                if calculate_overlapping(vol, other.spatialvolume) > 0.75
                    isRectInsideAnother = true
                    break
                end
            end
        end

        if isRectInsideAnother
            return vol
        end


        for p = 1:datatree.P
            if dimensionsFinished[p]
                #risky, can improve results but may lead to endless loop
                #dimensionsFinished[p] = false
                continue
            end

            change = true

            #adjust lower bound
            change1 = true
            while change1 && vol.pointcloud.probfactor > 1.0
                change1 = false
                margin = spvol.hi[p] - spvol.lo[p]
                buffer = spvol.lo[p]
                spvol.lo[p] -= margin * increase

                PtsIncrease = vol.pointcloud.points
                newvol = resize_integrationvol(datatree, p, vol, spvol, false)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probweightfactor < whiteningresult.targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc)
                    vol = newvol
                    wasCubeChanged = true
                    change = true
                    change1 = true
                    LogLow("\tTEST up p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")
                else
                    #revert changes
                    spvol.lo[p] = buffer

                    margin = spvol.hi[p] - spvol.lo[p]
                    buffer = spvol.lo[p]
                    spvol.lo[p] += margin * decrease

                    PtsIncrease = vol.pointcloud.points
                    newvol = resize_integrationvol(datatree, p, vol, spvol, false)

                    PtsIncrease = newvol.pointcloud.points / PtsIncrease

                    if PtsIncrease > (1 - decrease / ptsTolDec)
                        vol = newvol
                        wasCubeChanged = true
                        change = true
                        change1 = true
                        LogLow("\tTEST up p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")
                    else
                        #revert changes
                        spvol.lo[p] = buffer
                    end
                end
            end

            #adjust upper bound
            change2 = true
            while change2 && vol.pointcloud.probfactor > 1.0
                change2 = false
                margin = spvol.hi[p] - spvol.lo[p]
                buffer = spvol.hi[p]
                spvol.hi[p] += margin * increase

                PtsIncrease = vol.pointcloud.points
                newvol = resize_integrationvol(datatree, p, vol, spvol, false)

                PtsIncrease = newvol.pointcloud.points / PtsIncrease
                if newvol.pointcloud.probweightfactor < whiteningresult.targetprobfactor && PtsIncrease > (1.0 + increase / ptsTolInc)
                    vol = newvol
                    wasCubeChanged = true
                    change = true
                    change2 = true
                    LogLow("\tTEST up p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")
                else
                    #revert changes
                    spvol.hi[p] = buffer

                    margin = spvol.hi[p] - spvol.lo[p]
                    buffer = spvol.hi[p]
                    spvol.hi[p] -= margin * decrease

                    PtsIncrease = vol.pointcloud.points
                    newvol = resize_integrationvol(datatree, p, vol, spvol, false)

                    PtsIncrease = newvol.pointcloud.points / PtsIncrease

                    if PtsIncrease > (1 - decrease / ptsTolDec)
                        vol = newvol
                        wasCubeChanged = true
                        change = true
                        change2 = true
                        LogLow("\tTEST up p=$p Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")
                    else
                        #revert changes
                        spvol.hi[p] = buffer
                    end
                end
            end

            dimensionsFinished[p] = !change
        end

    end

    LogLow("TEST Hyperrectangle Points:\t$(vol.pointcloud.points)\tVolume:\t$(vol.volume)\tProb. Factor:\t$(vol.pointcloud.probfactor)")

    res = search(datatree, vol.spatialvolume, true)
    vol.pointcloud.pointIDs = res.pointIDs
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

@inline function calculate_overlapping(intvol::IntegrationVolume, other::HyperRectVolume)
    innerVol = 1.0
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

    return innerVol / intvol.volume
end

function integrate_hyperrectangle_noerror(dataset::DataSet, integrationvol::IntegrationVolume, determinant::Float64)::IntermediateResult
    s = 0.0
    count = 0.0
    for i in integrationvol.pointcloud.pointIDs
        #for numerical stability
        prob = (dataset.logprob[i] - integrationvol.pointcloud.maxLogProb)
        s += 1.0 / exp(prob) * dataset.weights[i]
        count += dataset.weights[i]
    end

    I = sum(dataset.weights) * integrationvol.volume / s / determinant / exp(-integrationvol.pointcloud.maxLogProb)

    LogLow("\tVolume:\t$(integrationvol.volume)\tIntegral: $I\tPoints:\t$(count)")

    return IntermediateResult(I, 0.0, Float64(integrationvol.pointcloud.points), integrationvol.volume)
end

function integrate_hyperrectangle(datatree::Tree, dataset::DataSet, integrationvol::IntegrationVolume, determinant::Float64)::IntermediateResult
    nIntegrals = 13

    Results = Array{IntermediateResult, 1}(nIntegrals)

    for i = 1:nIntegrals
        newvol = deepcopy(integrationvol.spatialvolume)
        volfactor = (i - 5) * 0.5 * 0.025 + 1#up to 10% decrease and 20% increase
        volfactor = volfactor^(1.0/dataset.P)
        volfactor -= 1.0
        for p = 1:dataset.P
            margin = newvol.hi[p] - newvol.lo[p]
            newvol.lo[p] += margin * 0.5 * volfactor
            newvol.hi[p] -= margin * 0.5 * volfactor
        end
        newintvol = IntegrationVolume(datatree, newvol, true)

        Results[i] = integrate_hyperrectangle_noerror(dataset, newintvol, determinant)
    end


    I = 0.0
    count = 0.0
    for i = 1:nIntegrals
        I += Results[i].integral / nIntegrals
        count += Results[i].points / nIntegrals
    end

    error = 0.0
    for i = 1:nIntegrals
        error += (Results[i].integral - I) ^ 2
    end
    error = sqrt(error / (nIntegrals - 1))

    if isnan(I) || I == Inf || I == -Inf
        res = integrate_hyperrectangle_noerror(dataset, integrationvol, determinant)
        LogMedium("No error could be calculated for integrate_hyperrectangle()")
        return IntermediateResult(res.integral, 0.0, res.points, integrationvol.volume)
    end

    return IntermediateResult(I, error, count, integrationvol.volume)
end
