


#trims dataset before performing mean and variance calculation to increase stability
function tmean(
    x::Vector{T},
    params...;
    tr::T = T(0.1),
    weights::Vector{T} = Vector{T}(0),
    calculateVar::Bool = false
)::Tuple where {T<:AbstractFloat}

    if tr < 0.0 || tr > 0.5
        @log_msg LOG_ERROR "tr is not allowed to be smaller than 0.0 or larger than 0.5"
    end
    for i in eachindex(params)
        if length(params[i]) != length(x)
            @log_msg LOG_ERROR "$i. entry of params has a different length than x"
        end
    end
    if length(weights) != 0 && length(weights) != length(x)
        @log_msg LOG_ERROR "length of weights doesn't match length of x"
    end

    perm = sortperm(x)

    l = length(x)
    lo = floor(Int64, tr * l + 1)
    hi = ceil(Int64, (1-tr) * l)

    norm::Bool = length(weights) > 0

    additionalreturns = Vector{T}(length(params))
    variances = Vector{T}((calculateVar ? length(params) + 1 : 0))
    for i in eachindex(params)
        additionalreturns[i] = norm ? mean(params[i][perm][lo:hi], FrequencyWeights(weights[perm][lo:hi])) : mean(params[i][perm][lo:hi])
    end
    if calculateVar
        variances[1] = norm ? var(x[perm][lo:hi], FrequencyWeights(weights[perm][lo:hi]), corrected = true) : var(x[perm][lo:hi])
        for i = 2:length(variances)
            variances[i] = norm ? var(params[i-1][perm][lo:hi], FrequencyWeights(params[i-1][perm][lo:hi]), corrected = true) : var(params[i-1][perm][lo:hi])
        end
    end
    return ((norm ? T(mean(x[perm][lo:hi], FrequencyWeights(weights[perm][lo:hi]))) : mean(T, x[perm][lo:hi])), additionalreturns..., variances...)
end



function calculateNess(dataset::DataSet{T, I}) where {T<:AbstractFloat, I<:Integer}
    pj = zeros(sum(dataset.weights))
    μ = zeros(dataset.P)
    for p = 1:dataset.P
        μ[p] = mean(view(dataset.data, p, :), FrequencyWeights(dataset.weights))
    end


    σ = zeros(dataset.P)
    for p=1:dataset.P
        σ[p] = var(view(dataset.data, p, :), FrequencyWeights(dataset.weights))
    end

    ∑_σ = sum(σ)

    cntr = 1
    for n = 1:dataset.N
        for w=1:dataset.weights[n]
            for p=1:dataset.P
                pj[cntr] += dataset.data[p, n] * dataset.data[p, n+1>dataset.N?1:n+1] - μ[p]^2
            end
            pj[cntr] \= ∑_σ
            cntr += 1
        end
    end

    Ness = sum(dataset.weights) / (1 + 2 * sum(pj))

    return Ness
end


function split_samples(
    samples::DensitySampleVector{T, T, I, ElasticArray{T, 2,1}, Array{T, 1},Array{I ,1}},
    sampleIDs::MCMCSampleIDVector,
    mcmcstats::MCMCBasicStats
)::Tuple{DataSet{T, I}, DataSet{T, I}} where {T<:Real, I<:Integer}

    N = length(sampleIDs)
    n = floor(Int64, N / 2)

    ds1 = DataSet(samples.params[:, 1:n], samples.log_value[1:n], samples.weight[1:n])
    ds2 = DataSet(samples.params[:, n+1:N], samples.log_value[n+1:N], samples.weight[n+1:N])

    return ds1, ds2
end
