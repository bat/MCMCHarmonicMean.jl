# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


mutable struct Tree{T<:AbstractFloat, I<:Integer}
    SortedData::Matrix{T}
    SortedLogProb::Vector{T}
    SortedWeights::Vector{T}
    SortedIDs::Vector{I}

    Cuts::I
    Leafsize::I

    DimensionList::Vector{I}
    RecursionDepth::I
    CutList::Vector{T}
    N::I
    P::I
end

function create_search_tree{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, MinCuts::I = 8, MaxLeafsize::I = 200)::Tree{T, I}
    LogMedium("Create Search Tree")
    newData = deepcopy(dataset.data)
    newLogProb = deepcopy(dataset.logprob)
    newWeights = deepcopy(dataset.weights)
    ids = [i for i::I = 1:dataset.N]

    suggCuts = (dataset.N / MaxLeafsize)^(1.0 / dataset.P)
    Cuts = ceil(I, max(MinCuts, suggCuts))

    recDepth = ceil(I, log(dataset.N / MaxLeafsize) / log(Cuts))

    #define dimension list
    local DimensionList::Vector{I}
    if Cuts > MinCuts
        DimensionList= [i for i::I = 1:dataset.P]
    else
        DimensionList = [i for i::I = 1:recDepth]
    end

    Leafsize = ceil(I, dataset.N / Cuts^recDepth)
    LogMedium("Cuts $Cuts\tLeafsize $Leafsize\tRec. Depth $recDepth")
    CutList = Vector{T}(0)

    if recDepth > 0
        createSearchTree(newData, newLogProb, newWeights, ids, DimensionList, CutList, Cuts, Leafsize, 1)
    end

    return Tree(newData, newLogProb, newWeights, ids, Cuts, Leafsize, DimensionList, length(DimensionList), CutList, dataset.N, dataset.P)
end

function createSearchTree{T<:AbstractFloat, I<:Integer}(Data::Matrix{T}, LogProb::Vector{T}, Weights::Vector{T}, IDs::Vector{I}, DimensionList::Vector{I}, CutList::Vector{T},
        Cuts::I, Leafsize::I, StartID::I = 0)
    remainingRec::I = length(DimensionList)

    stopMax::I = length(LogProb)
    thisLeaf::I = Leafsize * Cuts^(remainingRec - 1)
    bigLeaf::I = Leafsize * Cuts^remainingRec

    startInt::I = StartID
    stopInt::I = StartID + bigLeaf - 1
    if stopInt > stopMax
        stopInt = stopMax
    end

    sortID = sortperm(Data[DimensionList[1], startInt:stopInt])

    Data[:, startInt:stopInt] = Data[:, sortID+startInt-1]
    LogProb[startInt:stopInt] = LogProb[sortID+startInt-1]
    Weights[startInt:stopInt] = Weights[sortID+startInt-1]
    IDs[startInt:stopInt] = IDs[sortID+startInt-1]


    if remainingRec >= 1
        start::I = 0
        stop::I = StartID - 1

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

function search{T<:AbstractFloat, I<:Integer}(datatree::Tree{T, I}, searchvol::HyperRectVolume{T}, searchpoints::Bool = false)::SearchResult
    res = SearchResult(T, I)
    search!(res, datatree, searchvol, searchpoints)
    return res
end

function search!{T<:AbstractFloat, I<:Integer}(result::SearchResult{T, I}, datatree::Tree{T, I}, searchvol::HyperRectVolume{T}, searchpoints::Bool = false)
    result.points = 0
    resize!(result.pointIDs, 0)
    result.maxLogProb = -Inf
    result.minLogProb = Inf
    result.maxWeightProb = -Inf
    result.minWeightProb = Inf

    currentRecursion::I = 0
    currentDimension::I = 0
    treePos = Vector{I}(0)

    maxRecursion::I = datatree.RecursionDepth
    maxI::I = length(datatree.CutList)

    i::I = 1
    if maxI == 0
        #only on leaf
        for n::I = 1:datatree.N
            inVol = true
            for p::I = 1:datatree.P
                if datatree.SortedData[p, n] < searchvol.lo[p] || datatree.SortedData[p, n] > searchvol.hi[p]
                    inVol = false
                    break
                end
            end

            if inVol
                result.points += 1

                if searchpoints
                    push!(result.pointIDs, datatree.SortedIDs[n])
                end
                prob = datatree.SortedLogProb[n]
                w = datatree.SortedWeights[n]
                result.maxLogProb = max(result.maxLogProb, prob)
                result.minLogProb = min(result.minLogProb, prob)
                result.maxWeightProb = max(result.maxWeightProb, w)
                result.minWeightProb = min(result.minWeightProb, w)
            end
        end
        return

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


        diff::I = 1
        for r::I = 1:(maxRecursion - currentRecursion)
            diff += datatree.Cuts^r
        end
        low::T = datatree.CutList[i]
        high::T = i + diff > maxI ? datatree.CutList[end] : datatree.CutList[i+diff]
        if treePos[currentDimension] == datatree.Cuts
            high = Inf
        end


        if searchvol.lo[currentDimension] > high || searchvol.hi[currentDimension] < low
            #skip this interval
            i += diff
            if currentDimension < maxRecursion
                currentRecursion -= 1
            end
            continue
        end

        #if on deepest recursion check for points
        if currentRecursion == maxRecursion
            startID, stopID = getDataPositions(datatree, treePos)

            searchInterval!(result, datatree, searchvol, startID, stopID, searchpoints)
        end
        i += 1
    end

end

@inline function getDataPositions{T<:AbstractFloat, I<:Integer}(datatree::Tree{T, I}, TreePos::Vector{I})
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


function searchInterval!{T<:AbstractFloat, I<:Integer}(result::SearchResult{T, I}, datatree::Tree{T, I}, searchvol::HyperRectVolume{T}, start::I, stop::I, searchpoints::Bool)
    dimsort::I = datatree.DimensionList[datatree.RecursionDepth]

    for i::I = start:stop
        if datatree.SortedData[dimsort, i] > searchvol.hi[dimsort]
            break
        end
        inVol = true
        for p::I = 1:datatree.P
            if datatree.SortedData[p, i] < searchvol.lo[p] || datatree.SortedData[p, i] > searchvol.hi[p]
                inVol = false
                break
            end
        end

        if inVol
            result.points += 1

            if searchpoints
                push!(result.pointIDs, datatree.SortedIDs[i])
            end
            prob = datatree.SortedLogProb[i]
            w = datatree.SortedWeights[i]
            result.maxLogProb = max(result.maxLogProb, prob)
            result.minLogProb = min(result.minLogProb, prob)
            result.maxWeightProb = max(result.maxWeightProb, w)
            result.minWeightProb = min(result.minWeightProb, w)
        end
    end
end
