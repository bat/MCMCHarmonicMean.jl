

mutable struct Tree
    SortedData::Matrix{Float64}
    SortedLogProb::Vector{Float64}
    SortedIDs::Vector{Int64}

    Cuts::Int64
    Leafsize::Int64

    DimensionList::Vector{Int64}
    RecursionDepth::Int64
    CutList::Vector{Float64}
    N::Int64
    P::Int64
end

function CreateSearchTree(Data::DataSet, MinCuts::Integer = 8, MaxLeafsize::Integer = 200)::Tree
    println("Create Search Tree")
    newData = deepcopy(Data.Data)
    newLogProb = deepcopy(Data.LogProb)
    ids = [i for i = 1:Data.N]

    suggCuts = (Data.N / MaxLeafsize)^(1.0 / Data.P)
    Cuts = ceil(Int64, max(MinCuts, suggCuts))

    recDepth = ceil(Int64, log(Data.N / MaxLeafsize) / log(Cuts))

    #define dimension list
    local DimensionList::Vector{Int64}
    if Cuts > MinCuts
        DimensionList=[i for i=1:Data.P]
    else
        DimensionList = [i for i=1:recDepth]
    end

    Leafsize = ceil(Int64, Data.N / Cuts^recDepth)
    println("Cuts $Cuts\tLeafsize $Leafsize\tRec. Depth $recDepth")
    CutList = Vector{Float64}(0)

    createSearchTree(newData, newLogProb, ids, DimensionList, CutList, Cuts, Leafsize, 1)

    return Tree(newData, newLogProb, ids, Cuts, Leafsize, DimensionList, length(DimensionList), CutList, Data.N, Data.P)
end

function createSearchTree{T}(Data::Matrix{T}, LogProb::Vector{T}, IDs::Vector{Int64}, DimensionList::Vector{Int64}, CutList::Vector{Float64},
        Cuts::Int64, Leafsize::Int64, StartID::Int64 = 0)
    remainingRec = length(DimensionList)

    stopMax = length(LogProb)
    thisLeaf = Leafsize * Cuts^(remainingRec - 1)
    bigLeaf = Leafsize * Cuts^remainingRec

    startInt = StartID
    stopInt = StartID + bigLeaf - 1
    if stopInt > stopMax
        stopInt = stopMax
    end

    sortID = sortperm(Data[DimensionList[1], startInt:stopInt])

    Data[:, startInt:stopInt] = Data[:, sortID+startInt-1]
    LogProb[startInt:stopInt] = LogProb[sortID+startInt-1]
    IDs[startInt:stopInt] = IDs[sortID+startInt-1]


    if remainingRec >= 1
        start = 0
        stop = StartID - 1

        for i = 1:Cuts
            if stop == stopMax
                continue
            end

            start = stop + 1
            stop += thisLeaf
            if stop > stopMax
                stop = stopMax
            end

            push!(CutList, Data[DimensionList[1], start])

            if remainingRec > 1
                createSearchTree(Data, LogProb, IDs, DimensionList[2:end], CutList, Cuts, Leafsize, start)
            end
        end
    end
end

function Search(DataTree::Tree, Volume::Matrix{Float64}, SearchPoints::Bool = false)::SearchResult
    points = 0
    pointIDs = Vector{Int64}()
    maxprob = -Inf
    minprob = Inf

    currentRecursion = 0
    currentDimension = 0
    treePos = Vector{Int64}()

    maxRecursion = DataTree.RecursionDepth
    maxI = length(DataTree.CutList)

    i = 1
    while i <= maxI
        if currentRecursion < maxRecursion
            currentRecursion += 1
        end

        if length(treePos) < currentRecursion
            push!(treePos, 1)
        else
            treePos[currentRecursion] += 1
        end
        while treePos[currentRecursion] > DataTree.Cuts
            deleteat!(treePos, currentRecursion)
            currentRecursion -= 1
            treePos[currentRecursion] += 1
        end
        currentDimension = DataTree.DimensionList[currentRecursion]


        diff = 1
        for r = 1:(maxRecursion - currentRecursion)
            diff += DataTree.Cuts^r
        end
        low = DataTree.CutList[i]
        high = i + diff > maxI ? DataTree.CutList[end] : DataTree.CutList[i+diff]
        if treePos[currentDimension] == DataTree.Cuts
            high = Inf
        end


        if Volume[currentDimension, 1] > high || Volume[currentDimension, 2] < low
            #skip this interval
            i+=diff
            if currentDimension < maxRecursion
                currentRecursion -= 1
            end
            continue
        end

        #if on deepest recursion check for points
        if currentRecursion == maxRecursion
            startID, stopID = getDataPositions(DataTree, treePos)

            res = searchInterval(DataTree, Volume, startID, stopID, SearchPoints)
            points += res.Points
            maxprob = res.MaxLogProb > maxprob ? res.MaxLogProb : maxprob
            minprob = res.MinLogProb < minprob ? res.MinLogProb : minprob
            if SearchPoints
                append!(pointIDs, res.PointIDs)
            end
        end
        i += 1
    end

    return SearchResult(pointIDs, points, maxprob, minprob)
end

@inline function getDataPositions(DataTree::Tree, TreePos::Vector{Int64})
    maxRecursion = DataTree.RecursionDepth
    startID = 1
    recCntr = maxRecursion
    for t in TreePos
        recCntr -= 1
        startID += DataTree.Leafsize * DataTree.Cuts^recCntr * (t-1)
    end
    stopID = startID + DataTree.Leafsize - 1
    if stopID > DataTree.N
        stopID = DataTree.N
    end

    return startID, stopID
end


function searchInterval(DataTree::Tree, Volume::Matrix{Float64}, start::Int64, stop::Int64, SearchPoints::Bool)::SearchResult
    points = 0
    pointIDs = Vector{Int64}(0)
    maxprob = -Inf
    minprob = Inf

    dimsort = DataTree.DimensionList[DataTree.RecursionDepth]

    for i = start:stop
        if DataTree.SortedData[dimsort, i] > Volume[dimsort, 2]
            break
        end
        inVol = true
        for p = 1:DataTree.P
            if DataTree.SortedData[p, i] < Volume[p, 1] || DataTree.SortedData[p, i] > Volume[p, 2]
                inVol = false
                break
            end
        end

        if inVol
            points += 1

            if SearchPoints
                push!(pointIDs, DataTree.SortedIDs[i])
            end
            prob = DataTree.SortedLogProb[i]
            maxprob = prob > maxprob ? prob : maxprob
            minprob = prob < minprob ? prob : minprob
        end
    end

    return SearchResult(pointIDs, points, maxprob, minprob)
end
