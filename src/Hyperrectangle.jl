


export ResizeHyperrectangle

function Hyperrectangle(DataTree::Tree, Cube::Array{Float64, 2}, SearchPoints::Bool = false)::Hyperrectangle
    P = size(Cube)[1]

    volume = 1.0
    for p = 1:P
        volume *= Cube[p, 2] - Cube[p, 1]
    end

    res = Search(DataTree, Cube, SearchPoints)

    points = res.Points
    pointIDs = res.PointIDs

    maxProb = res.MaxLogProb
    minProb = res.MinLogProb

    probFactor = exp(maxProb - minProb)

    return Hyperrectangle(deepcopy(Cube), volume, maxProb, minProb, probFactor, points, pointIDs)
end


#only works correct if the resized dim gets changed and nothing else
function ResizeHyperrectangle(DataTree::Tree, ResizeDim::Int64, Rectangle::Hyperrectangle, ResizeVol::Array{Float64, 2}, SearchPoints::Bool = false)
    P = DataTree.P
    N = DataTree.N

    searchVol = deepcopy(ResizeVol)
    increase = true

    #increase
    if Rectangle.Cube[ResizeDim, 1] > ResizeVol[ResizeDim, 1]
        searchVol[ResizeDim, 2] = Rectangle.Cube[ResizeDim, 1]
        searchVol[ResizeDim, 1] = ResizeVol[ResizeDim, 1]
    elseif Rectangle.Cube[ResizeDim, 2] < ResizeVol[ResizeDim, 2]
        searchVol[ResizeDim, 1] = Rectangle.Cube[ResizeDim, 2]
        searchVol[ResizeDim, 2] = ResizeVol[ResizeDim, 2]
    else
        increase = false
        if Rectangle.Cube[ResizeDim, 1] < ResizeVol[ResizeDim, 1]
            searchVol[ResizeDim, 1] = Rectangle.Cube[ResizeDim, 1]
            searchVol[ResizeDim, 2] = ResizeVol[ResizeDim, 1]
        elseif Rectangle.Cube[ResizeDim, 2] > ResizeVol[ResizeDim, 2]
            searchVol[ResizeDim, 2] = Rectangle.Cube[ResizeDim, 2]
            searchVol[ResizeDim, 1] = ResizeVol[ResizeDim, 2]
        else
            error("Cube didn't change.")
        end
    end

    minProb = 0.0
    maxProb = 0.0
    local pointIDs::Array{Int64, 1}
    points = 0

    res = Search(DataTree, searchVol, SearchPoints)

    if increase

        maxProb = Rectangle.MaxLogProb
        minProb = Rectangle.MinLogProb

        points = Rectangle.Points + res.Points

        if SearchPoints
            pointIDs = deepcopy(Rectangle.PointIDs)
            append!(pointIDs, res.PointIDs)
        else
            pointIDs = Array{Int64, 1}(0)
        end

        if res.MaxLogProb > maxProb
            maxProb = res.MaxLogProb
        end
        if res.MinLogProb < minProb
            minProb = res.MinLogProb
        end
    else

        points = Rectangle.Points - res.Points

        if SearchPoints
            res = Search(DataTree, ResizeVol, SearchPoints)
            maxProb = res.MaxLogProb
            minProb = res.MinLogProb
            pointIDs = res.PointIDs
        else
            maxProb = Rectangle.MaxLogProb
            minProb = Rectangle.MinLogProb
            pointIDs = Array{Int64, 1}(0)
        end
    end

    volume = 1.0

    for p = 1:P
        volume *= ResizeVol[p, 2] - ResizeVol[p, 1]
    end
    probFactor = exp(maxProb - minProb)

    return Hyperrectangle(deepcopy(ResizeVol), volume, maxProb, minProb, probFactor, points, pointIDs)
end

function Hypercube(Mode::Array{Float64, 1}, DataTree::Tree, Length::Float64, SearchPoints::Bool)::Hyperrectangle
    P = DataTree.P

    cube = Array{Float64, 2}(P, 2)
    for p = 1:P
        cube[p, 1] = Mode[p] - Length * 0.5
        cube[p, 2] = Mode[p] + Length * 0.5
    end

    return Hyperrectangle(DataTree, cube, SearchPoints)
end

function FindHypercubeCenters(Data::DataSet, DataTree::Tree, ProbFactor::Float64, BoundingBox::Array{Float64, 2})::Array{Int64, 1}
    weight_Prob = 1.0
    weight_Dens = 1.0
    weight_Loca = 10.0
    weights = [-Inf for i=1:Data.N]

    sortLogProb = sortperm(Data.LogProb, rev = true)

    NMax = ceil(Int64, min(10000, Data.N * 0.05))

    ignorePoint = falses(Data.N)

    testlength = FindDensityTestCube(Data.Data[:, sortLogProb[1]], DataTree, max(round(Int64, Data.N * 0.001), Data.P * 10))[1]

    @showprogress for n in sortLogProb[1:NMax]
        if ignorePoint[n]
            continue
        end

        mode = view(Data.Data, :, n)



        if IsInHyperrectangle(Data.Data[:, n], BoundingBox)
            weights[n] = weight_Prob * Data.LogProb[n]

            cube =  Hypercube(Data.Data[:, n], DataTree, testlength, true)
            for id in cube.PointIDs
                ignorePoint[id] = true
            end
        end
    end

    sortIdx = sortperm(weights, rev = true)

    stop = 1
    for i = 1:Data.N
        if weights[sortIdx[i]] == -Inf
            stop = i
            break
        end
    end
    NMax = stop - 1
    if stop > min(1000, Data.N * 0.005)
        NMax = round(Int64, min(1000, Data.N * 0.005))
    end
    stop = Data.LogProb[sortIdx[1]] - log(ProbFactor)
    for i = 1:NMax
        if Data.LogProb[sortIdx[i]] < stop
            NMax = i
            break
        end
    end

    #return at least 10 possible hyper-rect centers
    if NMax < 10 && stop >= 10
        NMax = 10
    elseif NMax < 10 && stop < 10
        return sortLogProb[1:10]
    end

    LogMedium("Possible Hypersphere Centers: $NMax out of $(Data.N) points")

    return sortIdx[1:NMax]
end

function IsInHyperrectangle(Point::Array{Float64, 1}, BoundingBox::Array{Float64, 2})
    P = size(BoundingBox)[1]


    for p = 1:P
        if Point[p] < BoundingBox[p, 1] || Point[p] > BoundingBox[p, 2]
            return false
        end
    end

    return true
end

function IsInHyperrectangle(Point::Array{Float64, 1}, Rectangle::Hyperrectangle)
    P = size(Rectangle.Cube)[1]

    for p = 1:P
        if Rectangle.Cube[p, 1] > Point[p] || Rectangle.Cube[p, 2] < Point[p]
            return false
        end
    end

    return true
end

function FindDensityTestCube(Mode::Array{Float64, 1}, DataTree::Tree, Points::Int64 = 100)
    P = DataTree.P

    l = 1.0
    tol = 1.0
    mult = 1.2^(1.0 / P)

    cube = Hypercube(Mode, DataTree, l, false)
    pt = cube.Points
    #pt = length(cube.PointIDs)

    while pt < Points / tol || pt > Points * tol
        tol += 0.001
        if pt > Points
            l /= mult
        else
            l *= mult
        end
        cube = Hypercube(Mode, DataTree, l, false)
        pt = cube.Points
        #pt = length(cube.PointIDs)

    end

    return l, cube
end
