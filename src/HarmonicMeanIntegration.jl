



export Integrate
export CreateHyperrectangle


"""
    Integrate(Data::DataSet)::IntegrationResult

This function starts the Harmonic Mean Integration.
"""
function Integrate(Data::DataSet)::IntegrationResult
    if Data.N < Data.P * 50
        error("Not enough points for integration")
    end

    LogHigh("Integration started. Data Points:\t$(Data.N)\tParameters:\t$(Data.P)")
    LogHigh("Data Whitening")

    #whitening_result = CholeskyWhitening(Data)
    whitening_result = StatisticalWhitening(Data)

    tree = CreateSearchTree(Data)

    LogHigh("Find possible Hyperrectangle Centers")
    centerIDs = FindHypercubeCenters(Data, tree, whitening_result.SuggTarProbDiff, whitening_result.BoundingBox)
    Rectangles = Array{Hyperrectangle, 1}()

    LogHigh("Find good tolerances for Hyperrectangle Creation")

    vInc = zeros(0)
    pInc = zeros(0)
    pts = 10 * Data.P
    for id = 1:10
        c = FindDensityTestCube(Data.Data[:, centerIDs[id]], tree, pts)
        prevv = c[1]^Data.P
        prevp = c[2].Points
        for i in [2, 3, 4, 6, 8] * pts
            c = FindDensityTestCube(Data.Data[:, centerIDs[id]], tree, i)
            v = c[1]^Data.P
            p = c[2].Points
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


    @showprogress for i in centerIDs
        #check if id is inside an existing Hyperrectangle
        insideRectangle = false
        for rect in Rectangles
            if in(i, rect.PointIDs)
                iterations_skipped += 1
                insideRectangle = true
                break
            end
        end
        if insideRectangle
            continue
        end
        iterations += 1

        rectangle = CreateHyperrectangle(Data.Data[:, i], tree, Rectangles, whitening_result.SuggTarProbDiff, whitening_result.BoundingBox, suggTol)


        isRectInsideAnother = false
        for other in Rectangles
            overlap = CalculateOverlapping(rectangle, other.Cube)
            if overlap > 0.5
                isRectInsideAnother = true
                break
            end
        end
        if isRectInsideAnother
            iterations_skipped += 1
            LogLow("Hyperrectangle (ID = $i) discarded because it was inside another Rectangle.")
        elseif rectangle.ProbFactor == 1.0 || rectangle.Points < Data.P * 4
            LogLow("Hyperrectangle (ID = $i ) discarded because it has not enough Points.\tPoints:\t" *
                string(rectangle.Points) * "\tProb. Factor:\t" * string(rectangle.ProbFactor))
        else
            push!(Rectangles, rectangle)


            LogMedium(string(length(Rectangles)) * ". Hyperrectangle (ID = $i ) found. Rectangles discarded: $iterations_skipped\tPoints:\t" *
                string(rectangle.Points) * "\tProb. Factor:\t" * string(rectangle.ProbFactor))
            if rectangle.Points > maxPoints
                maxPoints = rectangle.Points
            end
        end


    end


    #remove rectangles with less than 10% points of the largest rectangle (in terms of points)
    j = length(Rectangles)
    for i = 1:length(Rectangles)
        if Rectangles[j].Points < maxPoints * 0.01
            deleteat!(Rectangles, j)
        end
        j -= 1
    end


    LogHigh("Integrate Hyperrectangle")

    nRes = length(Rectangles)
    IntResults = Array{IntermediateResult, 1}(nRes)

    @showprogress for i in 1:nRes
        IntResults[i] = IntegrateHyperrectangleWithError(tree, Data.LogProb, Data.Weights, Rectangles[i], whitening_result.Determinant)
        LogMedium("$i. Integral: "* string(IntResults[i].Integral) * "\tVolume\t" * string(IntResults[i].Volume) * "\tPoints:\t" * string(IntResults[i].Points))
    end

    #remove integrals with no result
    j = nRes
    for i = 1:nRes
        if isnan(IntResults[j].Integral)
            deleteat!(IntResults, j)
        end
        j -= 1
    end
    nRes = length(IntResults)

    result = 0.0
    error = 0.0
    points = 0.0
    volume = 0.0
    #normerror = 0.0
    for i = 1:nRes
        result += IntResults[i].Integral / nRes#(IntResults[i].Error / IntResults[i].Integral)
        #normerror += 1 / (IntResults[i].Error / IntResults[i].Integral)
        points += IntResults[i].Points / nRes
        volume += IntResults[i].Volume / nRes
    end
    #result /= normerror
    for i in 1:nRes
        error += (IntResults[i].Integral - result)^2
    end
    error = sqrt(error / (nRes - 1))
    if nRes == 1
        error = IntResults[1].Error
    end

    return IntegrationResult(result, error, nRes, points, volume, Rectangles, centerIDs, whitening_result)
end

function CreateHyperrectangle(Mode::Array{Float64, 1}, DataTree::Tree, Rectangles::Array{Hyperrectangle, 1},
        ProbFactor::Float64, BoundingBox::Array{Float64, 2}, Tolerance::Float64 = 1.2)
    P = DataTree.P
    N = DataTree.N

    lengthCube = 1.0

    cube = Hypercube(Mode, DataTree, lengthCube, true)

    while cube.Points > 0.01 * N
        lengthCube *= 0.5
        cube = Hypercube(Mode, DataTree, lengthCube, true)
    end
    tol = 1.0
    step = 0.7
    direction = 0
    PtsIncrease = 0.0
    VolIncrease = 1.0

    while cube.ProbFactor < ProbFactor / tol || cube.ProbFactor > ProbFactor
        tol += 0.1

        if cube.ProbFactor > ProbFactor
            #decrease side length
            VolIncrease = lengthCube^P
            lengthCube *= step
            VolIncrease = lengthCube^P / VolIncrease

            step = AdjustStepSize(step, direction == -1)
            direction = -1
        else
            #increase side length
            VolIncrease = lengthCube^P
            lengthCube /= step
            VolIncrease = lengthCube^P / VolIncrease

            step = AdjustStepSize(step, direction == 1)
            direction = 1
        end
        PtsIncrease = cube.Points
        cube = Hypercube(Mode, DataTree, lengthCube, true)
        PtsIncrease = cube.Points / PtsIncrease

        if cube.Points > 0.01 * N && cube.ProbFactor < ProbFactor
            break
        end
    end


    #adjust cube to  bounding box if necessary
    newdimensions = deepcopy(cube.Cube)
    for p = 1:P
        if cube.Cube[p, 1] < BoundingBox[p, 1]
            width = cube.Cube[p, 2] - cube.Cube[p, 1]
            newdimensions[p, 1] = BoundingBox[p, 1]
            cube = ResizeHyperrectangle(DataTree, p, cube, newdimensions, true)
            if cube.Cube[p, 1] >= cube.Cube[p, 2]
                newdimensions[p, 2] = min(cube.Cube[p, 2] + width, BoundingBox[p, 2])
                cube = ResizeHyperrectangle(DataTree, p, cube, newdimensions, true)
            end
        end

        if cube.Cube[p, 2] > BoundingBox[p, 2]
            width = cube.Cube[p, 2] - cube.Cube[p, 1]
            newdimensions[p, 2] = BoundingBox[p, 2]
            cube = ResizeHyperrectangle(DataTree, p, cube, newdimensions, true)
            if cube.Cube[p, 2] <= cube.Cube[p, 1]
                newdimensions[p, 1] = max(cube.Cube[p, 1] - width, BoundingBox[p, 1])
                cube = ResizeHyperrectangle(DataTree, p, cube, newdimensions, true)
            end
        end
    end


    LogLow("\tTEST Hyperrectangle Points:\t" * string(cube.Points) * "\tVolume:\t" * string(cube.Volume) * "\tProb. Factor:\t" * string(cube.ProbFactor))

    #check each dimension
    #1. Is Inside Bounding Box?
    #1.5 Is there already empty volume?
    #2. Are there more points available if one dimension gets larger?
    #2.1 Yes -> Make Box bigger if ProbFactor is within limits
    #2.2 No  -> Make Box Smaller
    #ptsTol = 1.1

    wasCubeChanged = true

    ptsTolInc = Tolerance
    ptsTolDec = Tolerance + (Tolerance - 1) * 1.5

    dimensionsFinished = falses(P)
    dimensions = deepcopy(cube.Cube)
    buffer = 0.0

    increase = 0.1
    decrease = 1.0 - 1.0 / (1.0 + increase)

    while wasCubeChanged && cube.ProbFactor > 1.0

        wasCubeChanged = false

        isRectInsideAnother = false
        for other in Rectangles
            if CalculateOverlapping(cube, other.Cube) > 0.5
                isRectInsideAnother = true
                break
            end
        end
        if isRectInsideAnother
            return cube
        end


        for p = 1:P
            if dimensionsFinished[p]
                #risky, can improve results but may lead to endless loop
                #dimensionsFinished[p] = false
                continue
            end

            change = true

            #adjust lower bound
            change1 = true
            while change1 && cube.ProbFactor > 1.0
                change1 = false
                margin = dimensions[p, 2] - dimensions[p, 1]
                buffer = dimensions[p, 1]
                dimensions[p, 1] -= margin * increase

                PtsIncrease = cube.Points
                newcube = ResizeHyperrectangle(DataTree, p, cube, dimensions)

                PtsIncrease = newcube.Points / PtsIncrease
                if newcube.ProbFactor < ProbFactor && PtsIncrease > (1.0 + increase / ptsTolInc) && newcube.Cube[p, 1] > BoundingBox[p, 1]
                    cube = newcube
                    wasCubeChanged = true
                    change = true
                    change1 = true
                    LogLow("\tTEST up p=$p Hyperrectangle Points:\t" * string(cube.Points) * "\tVolume:\t" * string(cube.Volume) * "\tProb. Factor:\t" * string(cube.ProbFactor))
                else
                    #revert changes
                    dimensions[p, 1] = buffer

                    margin = dimensions[p, 2] - dimensions[p, 1]
                    buffer = dimensions[p, 1]
                    dimensions[p, 1] += margin * decrease

                    PtsIncrease = cube.Points
                    newcube = ResizeHyperrectangle(DataTree, p, cube, dimensions)

                    PtsIncrease = newcube.Points / PtsIncrease

                    if PtsIncrease > (1 - decrease / ptsTolDec)
                        cube = newcube
                        wasCubeChanged = true
                        change = true
                        change1 = true
                        LogLow("\tTEST up p=$p Hyperrectangle Points:\t" * string(cube.Points) * "\tVolume:\t" * string(cube.Volume) * "\tProb. Factor:\t" * string(cube.ProbFactor))
                    else
                        #revert changes
                        dimensions[p, 1] = buffer
                    end
                end
            end

            #adjust upper bound
            change2 = true
            while change2 && cube.ProbFactor > 1.0
                change2 = false
                margin = dimensions[p, 2] - dimensions[p, 1]
                buffer = dimensions[p, 2]
                dimensions[p, 2] += margin * increase

                PtsIncrease = cube.Points
                newcube = ResizeHyperrectangle(DataTree, p, cube, dimensions)

                PtsIncrease = newcube.Points / PtsIncrease
                if newcube.ProbFactor < ProbFactor && PtsIncrease > (1.0 + increase / ptsTolInc) && newcube.Cube[p, 2] < BoundingBox[p, 2]
                    cube = newcube
                    wasCubeChanged = true
                    change = true
                    change2 = true
                    LogLow("\tTEST lo p=$p Hyperrectangle Points:\t" * string(cube.Points) * "\tVolume:\t" * string(cube.Volume) * "\tProb. Factor:\t" * string(cube.ProbFactor))
                else
                    #revert changes
                    dimensions[p, 2] = buffer

                    margin = dimensions[p, 2] - dimensions[p, 1]
                    buffer = dimensions[p, 2]
                    dimensions[p, 2] -= margin * decrease

                    PtsIncrease = cube.Points
                    newcube = ResizeHyperrectangle(DataTree, p, cube, dimensions)

                    PtsIncrease = newcube.Points / PtsIncrease

                    if PtsIncrease > (1 - decrease / ptsTolDec)
                        cube = newcube
                        wasCubeChanged = true
                        change = true
                        change2 = true
                        LogLow("\tTEST lo p=$p Hyperrectangle Points:\t" * string(cube.Points) * "\tVolume:\t" * string(cube.Volume) * "\tProb. Factor:\t" * string(cube.ProbFactor))
                    else
                        #revert changes
                        dimensions[p, 2] = buffer
                    end
                end
            end

            dimensionsFinished[p] = !change
        end

    end

    LogLow("TEST Hyperrectangle Points:\t" * string(cube.Points) * "\tVolume:\t" * string(cube.Volume) * "\tProb. Factor:\t" * string(cube.ProbFactor))

    res = Search(DataTree, cube.Cube, true)
    cube.PointIDs = res.PointIDs
    cube.MaxLogProb = res.MaxLogProb
    cube.MinLogProb = res.MinLogProb
    cube.ProbFactor = exp(cube.MaxLogProb - cube.MinLogProb)

    return cube
end

@inline function AdjustStepSize(Step, Increase::Bool)
    if Increase
        return Step * 0.5
    else
        return Step * 2.0
    end
end

@inline function CalculateOverlapping(Rectangle::Hyperrectangle, Other::Array{Float64, 2})
    P = size(Other)[1]

    innerVol = 1.0

    for p = 1:P
        if Rectangle.Cube[p, 1] > Other[p, 2] || Rectangle.Cube[p, 2] < Other[p, 1]
            return 0.0
        end
        if Rectangle.Cube[p, 1] > Other[p, 1] && Rectangle.Cube[p, 2] < Other[p, 2]
            innerVol *= Rectangle.Cube[p, 2] - Rectangle.Cube[p, 1]
        else
            innerVol *= min(abs(Rectangle.Cube[p, 1] - Other[p, 2]), abs(Rectangle.Cube[p, 2] - Other[p, 1]))
        end
    end

    return innerVol / Rectangle.Volume
end

function IntegrateHyperrectangle(LogProb::Array{Float64, 1}, Weights::Array{Float64, 1}, Rectangle::Hyperrectangle, Determinant::Float64)::IntermediateResult
    s = 0.0
    count = 0
    for i in Rectangle.PointIDs
        #for numerical sot tability
        prob = (LogProb[i] - Rectangle.MaxLogProb)
        s += 1.0 / exp(prob) * Weights[i]
        count += round(Weights[i])
    end

    I = sum(Weights) * Rectangle.Volume / s / Determinant / exp(-Rectangle.MaxLogProb)

    LogLow("\tVolume\t" * string(Rectangle.Volume) * "\tIntegral: "* string(I) * "\tPoints:\t" * string(count))

    return IntermediateResult(I, 0.0, Float64(Rectangle.Points), Rectangle.Volume)
end

function IntegrateHyperrectangleWithError(DataTree::Tree, LogProb::Array{Float64, 1}, Weights::Array{Float64, 1}, Rectangle::Hyperrectangle, Determinant::Float64)::IntermediateResult
    P = DataTree.P

    nIntegrals = 13

    Results = Array{IntermediateResult, 1}(nIntegrals)

    for i = 1:nIntegrals
        newdimensions = deepcopy(Rectangle.Cube)
        volfactor = (i - 5) * 0.5 * 0.05 + 1#up to 20% decrease and increase
        volfactor = volfactor^(1.0/P)
        volfactor -= 1.0
        for p = 1:P
            margin = newdimensions[p, 2] - newdimensions[p, 1]
            newdimensions[p, 1] += margin * 0.5 * volfactor
            newdimensions[p, 2] -= margin * 0.5 * volfactor
        end
        Rectangle = Hyperrectangle(DataTree, newdimensions, true)

        Results[i] = IntegrateHyperrectangle(LogProb, Weights, Rectangle, Determinant)
    end


    I = 0.0
    count = 0.0
    for i = 1:nIntegrals
        I += Results[i].Integral / nIntegrals
        count += Results[i].Points / nIntegrals
    end

    error = 0.0
    for i = 1:nIntegrals
        error += (Results[i].Integral - I) ^ 2
    end
    error = sqrt(error / (nIntegrals - 1))

    if isnan(I) || I == Inf || I == -Inf
        res = IntegrateHyperrectangle(LogProb, Weights, Rectangle, Determinant)
        return IntermediateResult(res.Integral, 0.0, res.Points, Rectangle.Volume)
        LogMedium("No error could be calculated for IntegrateHyperrectangleWithError()")
    end

    return IntermediateResult(I, error, count, Rectangle.Volume)
end


function CPUtime_ms()
    rusage = Libc.malloc(4*sizeof(Clong) + 14*sizeof(UInt64))
    ccall(:uv_getrusage, Cint, (Ptr{Void},), rusage)
    utime = UInt64(1000000)*unsafe_load(convert(Ptr{Clong}, rusage + 0*sizeof(Clong))) +    # user CPU time
                                    unsafe_load(convert(Ptr{Clong}, rusage + 1*sizeof(Clong)))
    stime = UInt64(1000000)*unsafe_load(convert(Ptr{Clong}, rusage + 2*sizeof(Clong))) +    # system CPU time
                                    unsafe_load(convert(Ptr{Clong}, rusage + 3*sizeof(Clong)))
    ttime = utime + stime  # total CPU time
    Libc.free(rusage)
    return ttime
end
