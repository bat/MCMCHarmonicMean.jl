using BAT
using MCMCHarmonicMean

#Model definition to generate samples from a n-dim gaussian shell
struct GaussianShellDensity<:AbstractDensity
    lambda::Vector{Float64}
    r::Float64
    sigma::Float64
    dimensions::Int64
end
BAT.nparams(model::GaussianShellDensity) = model.dimensions

#optional define exec_capabilities of our likelihood
BAT.exec_capabilities(::typeof(BAT.unsafe_density_logval), target::GaussianShellDensity, params::AbstractVector{<:Real}) = ExecCapabilities(0, true, 0, true)

#define likelihood for the Gaussian Shell
function BAT.unsafe_density_logval(target::GaussianShellDensity, params::Vector{Float64}, exec_context::ExecContext = ExecContext())
    diff::Float64 = 0
    for i in eachindex(params)
        diff += (target.lambda[i] - params[i]) * (target.lambda[i] - params[i])
    end
    diff = sqrt(diff)
    expo::Float64 = exp(-(diff - target.r) * (diff - target.r) / (2 * target.sigma^2))
    return log(1.0 / sqrt(2 * pi * target.sigma^2) * expo)
end

algorithm = MetropolisHastings()
#algorithm = MetropolisHastings(MHAccRejProbWeights{Float64}())

#define model and #dimensions
dim = 2
model = GaussianShellDensity(zeros(dim), 5.0, 2.0, dim)

#define boundaries
lo_bounds = [-30.0 for i = 1:dim]
hi_bounds = [ 30.0 for i = 1:dim]
bounds = HyperRectBounds(lo_bounds, hi_bounds, reflective_bounds)

chainspec = MCMCSpec(algorithm, model, bounds)
chains = 8
nsamples = 10^5

#define function to generate samples
sample() = rand(chainspec, nsamples, chains)


#Harmonic Mean Integration
#True integral value for 2D Gaussian Shell I = 31.4411
#True integral value for 10D Gaussian Shell I = 1.1065e9


#BAT.jl samples
bat_samples = sample()
data = HMIData(bat_samples)
hm_integrate(data)

#you can also re-run the integration without modifiying the already created hyper-rectangles, using different
#integration settings or different data sets
bat_samples2 = sample()
ds1, ds2 = split_samples(bat_samples...)
data.dataset1 = ds1
data.dataset2 = ds2
hm_integrate(data)


#other samples
nsamples = 10000
pdf_gauss(x, σ, μ) = log(1.0 / sqrt(2 * pi * σ^2) * exp(-(x-μ)^2 / (2σ^2)))
samples = randn(1, nsamples)
ds = DataSet(samples, pdf_gauss.(samples[1, :], 1.0, 0.0), ones(nsamples))
data = HMIData(ds)

#or alternatively two datasets, if the dataset is preferred to be manually splitted
#data = HMIData(ds1, ds2)

#use custom settings
setting = HMIPrecisionSettings()
setting.useMultiThreading = false
hm_integrate(data, settings = setting)
