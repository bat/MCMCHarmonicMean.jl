


function calculate_overlap(
    dataset::DataSet{T, I},
    volumes::Array{IntegrationVolume{T, I, V}, 1},
    integralestimates::IntermediateResults)::Array{T, 2} where{T<:AbstractFloat, I<:Integer, V<:SpatialVolume}


    M = length(integralestimates)

    overlap = zeros(T, M, M)

    sortedsets = SortedSet.([volumes[integralestimates.volumeID[i]].pointcloud.pointIDs for i = 1:M])

    p = Progress(M)
    @info "test"
    @onallthreads for i = threadpartition(1:M)
        for j = 1:M
            lock(_global_lock) do
                next!(p)
            end
            intersectpts = intersect(sortedsets[i], sortedsets[j])
            unionpts = union(sortedsets[i], sortedsets[j])

            overlap[i, j] = sum(dataset.weights[collect(intersectpts)]) / sum(dataset.weights[collect(unionpts)])
        end
    end
    finish!(p)

    overlap
end







function pdf_gauss(Z::Float64, μ::Float64,  σ_sq::Float64)::Float64
    return 1.0 / sqrt(2 * pi * σ_sq) * exp(-0.5 * (Z - μ)^2 / σ_sq)
end


function binomial_p_estimate_wald(n_total, n_success, nsigmas = 1)
    p_hat = n_success / n_total
    @info p_hat
    p_val = p_hat
    p_err = nsigmas * sqrt(p_hat * (1 - p_hat) / n_total)
end

function binomial_p_estimate_wilson(n_total, n_success, nsigmas = 1)
    n = n_total
    n_S = n_success
    n_F = n_total - n_S
    z = nsigmas
    z2 = z^2

    p_val = (n_S + z2/2) / (n + z2)

    p_err = z / (n + z2) * sqrt(n_S * n_F / n + z2 / 4)
end


function calculateuncertainty(dataset::DataSet{T, I}, volume::IntegrationVolume{T, I}, determinant::T, integral::T) where {T<:AbstractFloat, I<:Integer}
    f = 1 ./ exp.(dataset.logprob[volume.pointcloud.pointIDs])

    μ_Z = mean(f)
    σ_Z_sq = var(f)

    f_max = maximum(f)
    f_min = minimum(f)
    @info "$f_min, $f_max"

    y = (x::Float64) -> Float64(f_min + (f_max - f_min) * x)

    integrand1 = (x, f) -> f[1] = pdf_gauss(y(x[1]), μ_Z, σ_Z_sq) / y(x[1])^2 * (f_max - f_min)
    integrand2 = (x, f) -> f[1] = pdf_gauss(y(x[1]), μ_Z, σ_Z_sq) / y(x[1]) * (f_max - f_min)

    display(cuhre(integrand1, reltol = 0.02))
    display(cuhre(integrand2, reltol = 0.02))
    integral1 = vegas(integrand1, reltol = 0.02).integral[1]
    integral2 = vegas(integrand2, reltol = 0.02).integral[1]

    x_min = 1.0 / dataset.N
    x_max = 1.0

    #resorting the samples to undo the reordering of the space partitioning tree
    reorderscheme = sortperm(dataset.sortids[volume.pointcloud.pointIDs], rev = true)
    original_ordered_sample_ids = volume.pointcloud.pointIDs[reorderscheme]

    vol_samples = dataset.data[:, original_ordered_sample_ids]
    vol_weights = dataset.weights[original_ordered_sample_ids]

    vol_weight = sum(vol_weights)
    total_weight = sum(dataset.weights)
    r = vol_weight / total_weight

    y_r = (x::Float64) -> Float64(x_min + (x_max - x_min) * x)



    uncertainty_Z = (integral1 - integral2^2) * volume.volume / r / determinant



    ess = BAT.ESS(vol_samples, vol_weights)
    @info "ess: $ess"

    scaling_factor = ess / vol_weight

    uncertainty_r = binomial_p_estimate_wald(scaling_factor * total_weight, scaling_factor * vol_weight)
    @info "uncertainties: $uncertainty_r wald"

    uncertainty_r = binomial_p_estimate_wilson(scaling_factor * total_weight, scaling_factor * vol_weight)
    @info "uncertainties: $uncertainty_r wilson"

    uncertainty_r = integral / r * uncertainty_r
    @info "uncertainty using binomial error: $uncertainty_r"

    @info "uncertainties: $uncertainty_Z"

    uncertainty = sqrt(uncertainty_Z^2 + uncertainty_r^2)

    @info "Analytic Uncertainty Calculation: 1/f Mean:$(μ_Z)\tVar:$(sqrt(σ_Z_sq))\tInt: [$x_min, $x_max]\tVar Z:$(integral1-integral2^2)"
    @info "Integrals: $integral1 \t $integral2"
    @info "Final Analytic Uncertainty: $uncertainty"

    return uncertainty
end
