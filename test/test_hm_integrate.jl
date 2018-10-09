# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

using MCMCHarmonicMean
using Random
using Test

@testset "hm_integrate" begin

    nsamples = 10000
    pdf_gauss(x, σ, μ) = log(1.0 / sqrt(2 * pi * σ^2) * exp(-(x-μ)^2 / (2σ^2)))
    samples = randn(MersenneTwister(0), 1, nsamples)
    ds = DataSet(samples, pdf_gauss.(samples[1, :], 1.0, 0.0), ones(nsamples))
    data = HMIData(ds)
    hm_integrate!(data)
    @test data.integralestimates["cov. weighted result"].final.estimate ≈ 1.013448711024494
end
