
using Cuba


function pdf_gauss(Z::Float64, μ::Float64,  σ_sq::Float64)::Float64
    return 1.0 / sqrt(2 * pi * σ_sq) * exp(-0.5 * (Z - μ)^2 / σ_sq)
end

function pdf_binomial(x::Float64, N::Float64, p::Float64)
    if x <= 0
        return 0.0
    elseif x > N + 1
        return 1.0
    else
        #TODO use continuous form for binompdf using (incomplete) beta function
        return binompdf(round(Int64, N), p, round(Int64, x))
    end
end

function calculateuncertainty(dataset::DataSet{T, I}, volume::IntegrationVolume{T, I}, determinant::T, int_sub::T) where {T<:AbstractFloat, I<:Integer}
    f = 1./exp.(dataset.logprob[volume.pointcloud.pointIDs])

    μ_Z = mean(f)
    σ_Z_sq = var(f)

    f_max = maximum(f)
    f_min = minimum(f)

    y = (x::Float64) -> Float64(f_min + (f_max - f_min) * x)

    integrand1 = (x, f) -> f[1] = pdf_gauss(y(x[1]), μ_Z, σ_Z_sq) / y(x[1])^2 * (f_max - f_min)
    integrand2 = (x, f) -> f[1] = pdf_gauss(y(x[1]), μ_Z, σ_Z_sq) / y(x[1]) * (f_max - f_min)

    integral1 = vegas(integrand1, reltol = 0.02).integral[1]
    integral2 = vegas(integrand2, reltol = 0.02).integral[1]

    x_min = 1.0 / dataset.N
    x_max = 1.0
    r = volume.pointcloud.points / dataset.N
    y_r = (x::Float64) -> Float64(x_min + (x_max - x_min) * x)

    integrand1_r = (x, f) -> f[1] = pdf_binomial(y(x[1]), x_max, r) / y(x[1])^2 * (x_max - x_min)
    integrand2_r = (x, f) -> f[1] = pdf_binomial(y(x[1]), x_max, r) / y(x[1]) * (x_max - x_min)


    integral1_r = vegas(integrand1_r, reltol = 0.02).integral[1]
    integral2_r = vegas(integrand2_r, reltol = 0.02).integral[1]


    uncertainty_Z = (integral1 - integral2^2) * volume.volume / r / determinant

    uncertainty_r = (integral1_r - integral2_r^2) * int_sub

    @log_msg LOG_DEBUG "uncertainties: $uncertainty_Z\t $uncertainty_r"

    uncertainty = sqrt(uncertainty_Z^2 + uncertainty_r^2)

    @log_msg LOG_TRACE "Analytic Uncertainty Calculation: 1/f Mean:$(μ_Z)\tVar:$(sqrt(σ_Z_sq))\tInt: [$x_min, $x_max]\tVar Z:$(integral1-integral2^2)"
    @log_msg LOG_TRACE "Analytic Uncertainty: $uncertainty"

    return uncertainty
end
