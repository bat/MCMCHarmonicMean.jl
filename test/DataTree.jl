# This file is a part of MCMCHarmonicMean.jl, licensed under the MIT License (MIT).

using Compat.Test
using MCMCHarmonicMean
using BAT

@testset "DataTree" begin
    dataset = DataSet(rand(MersenneTwister(1), 2,100), ones(100), ones(100))
    tree = MCMCHarmonicMean.create_search_tree(dataset, 2, 25)

    @test typeof(tree) == MCMCHarmonicMean.DataTree{Float64, Int64}
    @test tree.Cuts == 2
    @test tree.Leafsize == 25
    @test tree.CutList == [0.0135403, 0.00790928, 0.521387, 0.46335, 0.0507632, 0.550098]

     res = MCMCHarmonicMean.search(dataset, tree, HyperRectVolume([0.0, 0.0], [0.5, 0.5]))
     @test res.points == 25
end
