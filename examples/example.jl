using MCMCHarmonicMean, Distributions

#Normal Distribution
function sample_normal(nsamples)
    pdf_gauss(x, σ, μ) = log(1.0 / sqrt(2 * pi * σ^2) * exp(-(x-μ)^2 / (2σ^2)))
    samples = randn(1, nsamples)
    DataSet(samples, pdf_gauss.(samples[1, :], 1.0, 0.0), ones(nsamples))
end
data = HMIData(sample_normal(100000))
hm_integrate!(data)


#2D Multivariate Normal Distribution
function sample_mvnormal(nsamples)
    mvnormal = MvNormal(zeros(2), reshape([0.5 0.3 0.3 0.5], 2, 2))
    samples = rand(mvnormal, nsamples)
    logprob = log.([pdf(mvnormal, samples[:, i]) for i =1:nsamples])
    DataSet(samples, logprob, ones(nsamples))
end
data = HMIData(sample_mvnormal(100000))
hm_integrate!(data)



#custom settings and replacing data set
setting = HMIPrecisionSettings()
setting.useMultiThreading = false
setting.max_startingIDs = 32
setting.whitening_function! = MCMCHarmonicMean.statistical_whitening! #default: cholesky_whitening! alternatively: no_whitening!
setting.dotrimming = false
newdataset = sample_mvnormal(100000)
#replace data sets
data.dataset1, data.dataset2 = MCMCHarmonicMean.split_dataset(newdataset)
#integrate the new data set using the already predefined hyper-rectangles
hm_integrate!(data, settings = setting)

#plotting
using Plots #include Plots after calling hm_integrate!, otherwise a Segmentation Fault may occur during integration
pyplot()
plot(data)#multiple keyword arguments possible, see plot_recipes.jl
