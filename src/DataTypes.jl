
export DataSet
export IntegrationResult
export WhiteningResult

"""
    DataSet{T<:AbstractFloat}

Holds the MCMC output.
# Arguments
- 'Data::Matrix{T}' : An P x N array with N data points with P parameters.
- 'LogProb::Vector{T}' : The logarithmic probability for each data point
- 'Weights::Vector{T}' : How often each sample occurred. Set to an array of ones if working directly on MCMC output
- 'N::Integer' : number of data points.
- 'P::Integer' : number of parameters.
"""
mutable struct DataSet{T<:AbstractFloat}
    Data::Matrix{T}
    LogProb::Vector{T}
    Weights::Vector{T}
    N::Integer
    P::Integer
end


mutable struct Hyperrectangle
    Cube::Array{Float64, 2}
    Volume::Float64
    MaxLogProb::Float64
    MinLogProb::Float64
    ProbFactor::Float64
    Points::Int64
    PointIDs::Array{Int64, 1}
end


struct WhiteningResult
    Determinant::Float64
    BoundingBox::Array{Float64, 2}
    LogProbDiff::Float64
    SuggTarProbDiff::Float64
    WhiteningMatrix::Array{Float64, 2}
    Mean::Array{Float64, 1}
end

mutable struct SearchResult
    PointIDs::Array{Int64, 1}
    Points::Int64
    MaxLogProb::Float64
    MinLogProb::Float64
end

struct IntegrationResult
    Integral::Float64
    Error::Float64
    Rectangles::Int64
    Points::Float64
    Volume::Float64
    RectangleList::Array{Hyperrectangle, 1}
    StartingIDs::Array{Int64, 1}
    WhiteningResult::WhiteningResult
end

struct IntermediateResult
    Integral::Float64
    Error::Float64
    Points::Float64
    Volume::Float64
end
