# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).



struct TreeNode{T<:Real}
    Cuts::Vector{T}
    Dimension::Int64

    SubNodes::Array{TreeNode{T}, 1}
    IsEndPoint::Bool

    Positions::Matrix{T}
    LogProb::Vector{T}
    PointIDs::Array{Int64, 1}
    Points::Int64
end

struct Tree{T<:Real}
    DimensionList::Vector{Int64}
    Cuts::Int64

    Leafsize::Int64
    StartingNode::TreeNode{T}
    N::Int64
    P::Int64
end


"""
    CreateSearchTree(Data, LogProb, StartingDimension, Cuts = 4, Leafsize = 100)

Creates a Search Tree
"""
function create_search_tree(Data::DataSet, MinCuts::Integer = 8, MaxLeafsize::Integer = 200)::Tree
    #find number of cuts per dimension
    suggCuts = (Data.N / MaxLeafsize)^(1.0 / Data.P)
    Cuts = round(Int64, max(MinCuts, suggCuts))

    #define dimension list
    local DimensionList::Vector{Int64}
    if Cuts > MinCuts
        DimensionList=[i for i=1:Data.P]
    else
        #TODO find algorithm to divide dimension
        DimensionList=[i for i=1:Data.P]
    end

    endpoints = divide_dimension(DimensionList[1], Data.Data, Data.LogProb, [i for i=1:Data.N], Cuts)
    nodes = Array{TreeNode, 1}(Cuts)


    cutx = Array{Float64, 1}(Cuts + 1)
    cutx[1] = endpoints[1].Positions[DimensionList[1], 1]
    for i = 2:Cuts
        cutx[i] = (endpoints[i - 1].Positions[DimensionList[1], end] + endpoints[i].Positions[DimensionList[1], 1]) * 0.5
    end
    cutx[Cuts + 1] = endpoints[Cuts].Positions[DimensionList[1], end]

    for i = 1:Cuts
        nodes[i] = createTree(Data.Data, Data.LogProb, endpoints[i].PointIDs, DimensionList[2:end], Cuts, MaxLeafsize)
    end

    tree = Tree(DimensionList, Cuts, MaxLeafsize, TreeNode(cutx, DimensionList[1], nodes, false, Array{Float64, 2}(0, 0), Array{Float64, 1}(0), Array{Int64, 1}(0), 0), Data.N, Data.P)

    return tree
end

function createTree{T}(Data::Matrix{T}, LogProb::Vector{T}, PointIDs::Vector{Int64}, DimensionList::Vector{Int64}, Cuts::Int64, Leafsize::Int64)
    P = size(Data)[1]
    N = size(Data)[2]

    subData = Data[:, PointIDs]
    endpoints = divide_dimension(DimensionList[1], subData, LogProb[PointIDs], PointIDs, Cuts)

    cutx = Array{Float64, 1}(Cuts + 1)
    cutx[1] = endpoints[1].Positions[DimensionList[1], 1]
    for i = 2:Cuts
        cutx[i] = (endpoints[i - 1].Positions[DimensionList[1], end] + endpoints[i].Positions[DimensionList[1], 1]) * 0.5
    end
    cutx[Cuts + 1] = endpoints[Cuts].Positions[DimensionList[1], end]

    if endpoints[1].Points > Leafsize && length(DimensionList) > 1
        for i  = 1:Cuts
            endpoints[i] = createTree(Data, LogProb, endpoints[i].PointIDs, DimensionList[2:end], Cuts, Leafsize)
        end
    end

    return TreeNode(cutx, DimensionList[1], endpoints, false, Array{Float64, 2}(0, 0), Array{Float64, 1}(0), Array{Int64, 1}(0), 0)
end

function divide_dimension(Dim::Int64, DataSubSet::Array{Float64, 2}, LogProb::Array{Float64, 1}, PointIDs::Array{Int64, 1}, Cuts::Int64)
    N = length(PointIDs)

    sortIdx = sortperm(DataSubSet[Dim, :])

    start = 0
    stop = 0

    intervals = Array{TreeNode, 1}(Cuts)

    for i = 1:Cuts
        start = stop + 1
        stop = round(Int64, i / Cuts * N)

        intervals[i] = TreeNode(Array{Float64, 1}(), Dim, Array{TreeNode, 1}(), true,
                                DataSubSet[:, sortIdx[start:stop]],
                                LogProb[sortIdx[start:stop]],
                                PointIDs[sortIdx[start:stop]],
                                stop - start + 1)

    end

    return intervals
end

function search(DataTree::Tree, Volume::Array{Float64, 2}, SearchPoints::Bool = false)::SearchResult
    #global c = 0
    res = searchDim(DataTree.StartingNode, Volume, SearchPoints)
    #LogMedium("Search count: $c")
    return res
end

function searchDimEndPoint(Node::TreeNode, Volume::Array{Float64, 2}, SearchPoints::Bool)::SearchResult
    P = size(Volume)[1]

    #global c += 1

    result = SearchResult(Array{Int64, 1}(0), 0, -Inf, Inf)

    cntr = 0
    for i = 1:Node.Points
        inVol = true
        for p = 1:P
            if Node.Positions[p, i] < Volume[p, 1] || Node.Positions[p, i] > Volume[p, 2]
                inVol = false
                break
            end
        end

        if inVol
            cntr += 1

            if SearchPoints
                append!(result.PointIDs, Node.PointIDs[i])
            end

            if Node.LogProb[i] > result.MaxLogProb
                result.MaxLogProb = Node.LogProb[i]
            elseif Node.LogProb[i] < result.MinLogProb
                result.MinLogProb = Node.LogProb[i]
            end

        end
    end

    result.Points = cntr

    return result
end


function searchDim(Node::TreeNode, Volume::Array{Float64, 2}, SearchPoints::Bool)::SearchResult
    P::Int64 = size(Volume)[1]

    result = SearchResult(Array{Int64, 1}(0), 0, -Inf, Inf)

    Dim = Node.Dimension

    for i in eachindex(Node.SubNodes)
        if Node.Cuts[i] <= Volume[Dim, 2] && Node.Cuts[i + 1] >= Volume[Dim, 1]
            local newPts::SearchResult
            if  Node.SubNodes[i].IsEndPoint
                newPts = searchDimEndPoint(Node.SubNodes[i], Volume, SearchPoints)
            else
                newPts = searchDim(Node.SubNodes[i], Volume, SearchPoints)
            end

            if newPts.MaxLogProb > result.MaxLogProb
                result.MaxLogProb = newPts.MaxLogProb
            end
            if newPts.MinLogProb < result.MinLogProb
                result.MinLogProb = newPts.MinLogProb
            end

            result.Points += newPts.Points
            if SearchPoints
                append!(result.PointIDs, newPts.PointIDs)
            end
        end
    end

    return result
end
