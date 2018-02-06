# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).


mutable struct DataTree{
    T<:AbstractFloat,
    I<:Integer
} <: SearchTree
    Cuts::I
    Leafsize::I

    DimensionList::Vector{I}
    RecursionDepth::I
    CutList::Vector{T}
end

function create_search_tree(dataset::DataSet{T, I}, MinCuts::I = 8, MaxLeafsize::I = 200)::DataTree{T, I} where {T<:AbstractFloat, I<:Integer}
    suggCuts = (dataset.N / MaxLeafsize) ^ (1 / dataset.P)
    Cuts = ceil(I, max(MinCuts, suggCuts))

    recDepth = ceil(I, log(dataset.N / MaxLeafsize) / log(Cuts))

    #define dimension list
    DimensionList = if Cuts > MinCuts
        [i for i = 1:dataset.P]
    else
        [i for i = 1:recDepth]
    end

    Leafsize = ceil(I, dataset.N / Cuts^recDepth)
    @log_msg LOG_DEBUG "Cuts $Cuts\tLeafsize $Leafsize\tRec. Depth $recDepth"
    CutList = Vector{T}(0)

    if recDepth > 0
        createleafs(dataset, DimensionList, CutList, Cuts, Leafsize, 1)
    end

    return DataTree(Cuts, Leafsize, DimensionList, length(DimensionList), CutList)
end

function createleafs{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, DimensionList::Vector{I}, CutList::Vector{T}, Cuts::I, Leafsize::I, StartID::I = 0)
    remainingRec::I = length(DimensionList)

    thisLeaf::I = Leafsize * Cuts^(remainingRec - 1)
    bigLeaf::I = Leafsize * Cuts^remainingRec

    startInt::I = StartID
    stopInt::I = StartID + bigLeaf - 1
    if stopInt > dataset.N
        stopInt = dataset.N
    end

    sortID = sortperm(dataset.data[DimensionList[1], startInt:stopInt])

    dataset.data[:, startInt:stopInt] = dataset.data[:, sortID+startInt-1]
    dataset.logprob[startInt:stopInt] = dataset.logprob[sortID+startInt-1]
    dataset.weights[startInt:stopInt] = dataset.weights[sortID+startInt-1]

    if remainingRec >= 1
        start::I = 0
        stop::I = StartID - 1

        for i = 1:Cuts
            if stop == dataset.N
                continue
            end

            start = stop + 1
            stop += thisLeaf
            if stop > dataset.N
                stop = dataset.N
            end

            push!(CutList, dataset.data[DimensionList[1], start])

            if remainingRec > 1
                createleafs(dataset, DimensionList[2:end], CutList, Cuts, Leafsize, start)
            end
        end
    end
end

function search{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, datatree::DataTree{T, I}, searchvol::HyperRectVolume{T}, searchpoints::Bool = false)::SearchResult
    res = SearchResult(T, I)
    #searchpoints = false
    search!(res, dataset, datatree, searchvol, searchpoints)

    return res
end

function search!{T<:AbstractFloat, I<:Integer}(result::SearchResult{T, I}, dataset::DataSet{T, I}, datatree::DataTree{T, I}, searchvol::HyperRectVolume{T}, searchpoints::Bool = false)
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
        for n::I = 1:dataset.N
            inVol = true
            for p::I = 1:dataset.P
                if dataset.data[p, n] < searchvol.lo[p] || dataset.data[p, n] > searchvol.hi[p]
                    inVol = false
                    break
                end
            end

            if inVol
                result.points += 1

                if searchpoints
                    push!(result.pointIDs, n)
                end
                prob = dataset.logprob[n]
                w = log(dataset.weights[n]) + prob
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
            startID, stopID = getDataPositions(dataset, datatree, treePos)

            searchInterval!(result, dataset, datatree, searchvol, startID, stopID, searchpoints)
        end
        i += 1
    end

end

@inline function getDataPositions{T<:AbstractFloat, I<:Integer}(dataset::DataSet{T, I}, datatree::DataTree{T, I}, TreePos::Vector{I})
    maxRecursion = datatree.RecursionDepth
    startID = 1
    recCntr = maxRecursion
    for t in TreePos
        recCntr -= 1
        startID += datatree.Leafsize * datatree.Cuts^recCntr * (t-1)
    end
    stopID = startID + datatree.Leafsize - 1
    if stopID > dataset.N
        stopID = dataset.N
    end

    return startID, stopID
end


function searchInterval!{T<:AbstractFloat, I<:Integer}(result::SearchResult{T, I}, dataset::DataSet{T, I}, datatree::DataTree{T, I}, searchvol::HyperRectVolume{T}, start::I, stop::I, searchpoints::Bool)
    dimsort = datatree.DimensionList[datatree.RecursionDepth]

    for i = start:stop
        if dataset.data[dimsort, i] > searchvol.hi[dimsort]
            break
        end
        inVol = true
        for p = 1:dataset.P
            if dataset.data[p, i] < searchvol.lo[p] || dataset.data[p, i] > searchvol.hi[p]
                inVol = false
                break
            end
        end

        if inVol
            result.points += 1

            if searchpoints
                push!(result.pointIDs, i)
            end
            prob = dataset.logprob[i]
            w = log(dataset.weights[i]) + prob
            result.maxLogProb = max(result.maxLogProb, prob)
            result.minLogProb = min(result.minLogProb, prob)
            result.maxWeightProb = max(result.maxWeightProb, w)
            result.minWeightProb = min(result.minWeightProb, w)
        end
    end
end
