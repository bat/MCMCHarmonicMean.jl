# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


mutable struct Tree{T<:Real}
    SortedData::Matrix{T}
    SortedLogProb::Vector{T}
    SortedWeights::Vector{T}
    SortedIDs::Vector{Int64}

    Cuts::Int64
    Leafsize::Int64

    DimensionList::Vector{Int64}
    RecursionDepth::Int64
    CutList::Vector{T}
    N::Int64
    P::Int64
end

function create_search_tree(dataset::DataSet, MinCuts::Integer = 8, MaxLeafsize::Integer = 200)::Tree
    LogMedium("Create Search Tree")
    newData = deepcopy(dataset.data)
    newLogProb = deepcopy(dataset.logprob)
    newWeights = deepcopy(dataset.weights)
    ids = [i for i = 1:dataset.N]

    suggCuts = (dataset.N / MaxLeafsize)^(1.0 / dataset.P)
    Cuts = ceil(Int64, max(MinCuts, suggCuts))

    recDepth = ceil(Int64, log(dataset.N / MaxLeafsize) / log(Cuts))

    #define dimension list
    local DimensionList::Vector{Int64}
    if Cuts > MinCuts
        DimensionList= [i for i=1:dataset.P]
    else
        DimensionList = [i for i=1:recDepth]
    end

    Leafsize = ceil(Int64, dataset.N / Cuts^recDepth)
    LogMedium("Cuts $Cuts\tLeafsize $Leafsize\tRec. Depth $recDepth")
    CutList = Vector{Float64}(0)

    if recDepth > 0
        createSearchTree(newData, newLogProb, newWeights, ids, DimensionList, CutList, Cuts, Leafsize, 1)
    end

    return Tree(newData, newLogProb, newWeights, ids, Cuts, Leafsize, DimensionList, length(DimensionList), CutList, dataset.N, dataset.P)
end

function createSearchTree{T}(Data::Matrix{T}, LogProb::Vector{T}, Weights::Vector{T}, IDs::Vector{Int64}, DimensionList::Vector{Int64}, CutList::Vector{Float64},
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
    Weights[startInt:stopInt] = Weights[sortID+startInt-1]
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
                createSearchTree(Data, LogProb, Weights, IDs, DimensionList[2:end], CutList, Cuts, Leafsize, start)
            end
        end
    end
end

function search(datatree::Tree, searchvol::HyperRectVolume, SearchPoints::Bool = false)::SearchResult
    points = 0
    pointIDs = Vector{Int64}()
    maxprob = -Inf
    minprob = Inf
    maxwp = -Inf
    minwp = Inf

    currentRecursion = 0
    currentDimension = 0
    treePos = Vector{Int64}()

    maxRecursion = datatree.RecursionDepth
    maxI = length(datatree.CutList)

    i = 1
    if maxI == 0
        #only on leaf
        for n = 1:datatree.N
            inVol = true
            for p = 1:datatree.P
                if datatree.SortedData[p, n] < searchvol.lo[p] || datatree.SortedData[p, n] > searchvol.hi[p]
                    inVol = false
                    break
                end
            end

            if inVol
                points += 1

                if SearchPoints
                    push!(pointIDs, datatree.SortedIDs[n])
                end
                prob = datatree.SortedLogProb[n]
                w = datatree.SortedWeights[n]
                maxprob = prob > maxprob ? prob : maxprob
                maxwp = prob * w > maxwp ? prob * w : maxwp
                minprob = prob < minprob ? prob : minprob
                minwp = prob * w < minwp ? prob * w : minwp
            end
        end
        return SearchResult(pointIDs, points, maxprob, minprob, maxwp, minwp)

    end
    while i <= maxI
        if currentRecursion < maxRecursion
            currentRecursion += 1
        end

        if length(treePos) < currentRecursion
            push!(treePos, 1)
        else
            treePos[currentRecursion] += 1
        end
        while treePos[currentRecursion] > datatree.Cuts
            deleteat!(treePos, currentRecursion)
            currentRecursion -= 1
            treePos[currentRecursion] += 1
        end
        currentDimension = datatree.DimensionList[currentRecursion]


        diff = 1
        for r = 1:(maxRecursion - currentRecursion)
            diff += datatree.Cuts^r
        end
        low = datatree.CutList[i]
        high = i + diff > maxI ? datatree.CutList[end] : datatree.CutList[i+diff]
        if treePos[currentDimension] == datatree.Cuts
            high = Inf
        end


        if searchvol.lo[currentDimension] > high || searchvol.hi[currentDimension] < low
            #skip this interval
            i+=diff
            if currentDimension < maxRecursion
                currentRecursion -= 1
            end
            continue
        end

        #if on deepest recursion check for points
        if currentRecursion == maxRecursion
            startID, stopID = getDataPositions(datatree, treePos)

            res = searchInterval(datatree, searchvol, startID, stopID, SearchPoints)
            points += res.points
            maxprob = res.maxLogProb > maxprob ? res.maxLogProb : maxprob
            minprob = res.minLogProb < minprob ? res.minLogProb : minprob
            maxwp = res.maxWeightProb > maxwp ? res.maxWeightProb : maxwp
            minwp = res.minWeightProb < minwp ? res.minWeightProb : minwp
            if SearchPoints
                append!(pointIDs, res.pointIDs)
            end
        end
        i += 1
    end

    return SearchResult(pointIDs, points, maxprob, minprob, maxwp, minwp)
end

@inline function getDataPositions(datatree::Tree, TreePos::Vector{Int64})
    maxRecursion = datatree.RecursionDepth
    startID = 1
    recCntr = maxRecursion
    for t in TreePos
        recCntr -= 1
        startID += datatree.Leafsize * datatree.Cuts^recCntr * (t-1)
    end
    stopID = startID + datatree.Leafsize - 1
    if stopID > datatree.N
        stopID = datatree.N
    end

    return startID, stopID
end


function searchInterval(datatree::Tree, searchvol::HyperRectVolume, start::Int64, stop::Int64, SearchPoints::Bool)::SearchResult
    points = 0
    pointIDs = Vector{Int64}(0)
    maxprob = -Inf
    minprob = Inf
    maxwp = -Inf
    minwp = Inf

    dimsort = datatree.DimensionList[datatree.RecursionDepth]

    for i = start:stop
        if datatree.SortedData[dimsort, i] > searchvol.hi[dimsort]
            break
        end
        inVol = true
        for p = 1:datatree.P
            if datatree.SortedData[p, i] < searchvol.lo[p] || datatree.SortedData[p, i] > searchvol.hi[p]
                inVol = false
                break
            end
        end

        if inVol
            points += 1

            if SearchPoints
                push!(pointIDs, datatree.SortedIDs[i])
            end
            prob = datatree.SortedLogProb[i]
            w = datatree.SortedWeights[i]
            maxprob = prob > maxprob ? prob : maxprob
            maxwp = prob * w > maxwp ? prob * w : maxwp
            minprob = prob < minprob ? prob : minprob
            minwp = prob * w < minwp ? prob * w : minwp
        end
    end

    return SearchResult(pointIDs, points, maxprob, minprob, maxwp, minwp)
end
