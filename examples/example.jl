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

#easy way
data = sample()
result = hm_integrate(data)

#normal way - enables the option to reevaluate the integral for different samples, plotting and more
dataset = DataSet(sample())
result = HMIData(dataset)

hm_integrate(result)

#redo integration with the same integration volumes but a different data set
newdataset = DataSet(sample())
hm_swapdata(result, newdataset)
hm_integrate(result)

#redo integration with custom settings
setting = HMIntegrationPrecisionSettings()
setting.userectweights = false  #doesn't use rect weights which compensate for overlapping integration volumes
setting.nvolumerand = 5         #use 5 random volume variations during the integration process
hm_reset_hyperrectangles(result)
#hm_reset_tolerance(result)
hm_integrate(result, settings = setting)


#result holds all information neccessary to redo the integration process, the chosen hyper-rectangle volumes including the ids of the samples in it,
#the whitening result and its determinant to normalize the integral, the (resorted) dataset and the datatree which stores information how the dataset was resorted, ...
#manually integrate the integral of j.th volume and compare it to the integral of result.integrals[1]
j = 2
s=0.0
for i in result.volumelist[j].pointcloud.pointIDs
	inV = true
	for p=1:dataset.P
		if dataset.data[p, i] < result.volumelist[j].spatialvolume.lo[p] || dataset.data[p, i] > result.volumelist[j].spatialvolume.hi[p]
			inV = false
			break
		end
	end
	if inV
	 	s += 1.0 / exp(result.dataset.logprob[i]) * result.dataset.weights[i]
	end
end
totalWeight = sum(result.dataset.weights)
integral = totalWeight * result.volumelist[j].volume / s / get(result.whiteningresult).determinant


#stored integral, not the same because the hm_integrate does a small randomization to the integration volume before calculating the integral
result.integrals[j]




#plot the 2D Gaussian shell, the starting IDs and the hyper-rectangles
#for plotting you need to construct the HMIResult before the integration process, otherwise plotting will yield wrong results, before information on how the original
#data was resorted is lost (as well as the whitening information and more)
using Plots
pyplot() #don't include Plot before the integration is done, otherwise segmentation faults may occur

scatter(result.dataset.data[1, :], result.dataset.data[2, :], c=:red, marker=:circle, markersize = 0.2 * sqrt.(result.dataset.weights), size=(1000,1000), markerstrokewidth=0, legend=false)

for vol in result.volumelist
    points = Array{Float64, 2}(2, 5)
    points[:, 1] = [vol.spatialvolume.lo[1], vol.spatialvolume.lo[2]]
    points[:, 2] = [vol.spatialvolume.lo[1], vol.spatialvolume.hi[2]]
    points[:, 3] = [vol.spatialvolume.hi[1], vol.spatialvolume.hi[2]]
    points[:, 4] = [vol.spatialvolume.hi[1], vol.spatialvolume.lo[2]]
    points[:, 5] = [vol.spatialvolume.lo[1], vol.spatialvolume.lo[2]]
    plot!(points[1, :], points[2, :], c=:blue, linewidth=1.0)
end

pts = result.dataset.data[:, result.startingIDs]
scatter!(pts[1, :], pts[2, :], c=:black, markersize=2.0, marker=:rect, linewidth=0.0)

#savefig("path/example.png")
