

function create_rectangle(rect::MCMCHarmonicMean.HyperRectVolume, dim1::Integer, dim2::Integer)
    xpoints = zeros(Float64, 6)
    ypoints = zeros(Float64, 6)

    xpoints[1] = rect.lo[FirstDim]
    xpoints[2] = rect.lo[FirstDim]
    xpoints[3] = rect.hi[FirstDim]
    xpoints[4] = rect.hi[FirstDim]
    xpoints[5] = rect.lo[FirstDim]
    xpoints[6] = NaN    #necessary to group rectangles under same label
    ypoints[1] = rect.lo[SecondDim]
    ypoints[2] = rect.hi[SecondDim]
    ypoints[3] = rect.hi[SecondDim]
    ypoints[4] = rect.lo[SecondDim]
    ypoints[5] = rect.lo[SecondDim]
    ypoints[6] = NaN

    xpoints, ypoints
end


@recipe function f(dataset::DataSet, dim1::Integer = 1, dim2::Integer = 2)
    @assert dataset.P > 1

    xlimits=(minimum(dataset.data[dim1, :]), maximum(dataset.data[dim1, :]))
    ylimits=(minimum(dataset.data[dim2, :]), maximum(dataset.data[dim2, :]))

    x, y = dataset.data[dim1, :], dataset.data[dim2, :]
    #c:=:red
    #marker=:circle
    #markersize = 0.5 * sqrt.(dataset.weights)
    #size=(1000,1000)
    #markerstrokewidth=0
    #xlim=xlimits
    #ylim=ylimits
    #legend=:topright
end


rejected = length(rejectedids)
accepted = length(volumes) - rejected
rectangles_x = Array{Float64, 1}(accepted * 6)
rectangles_y = Array{Float64, 1}(accepted * 6)
rectanglesrej_x = Array{Float64, 1}(rejected * 6)
rectanglesrej_y = Array{Float64, 1}(rejected * 6)
cntr = 1
cntrrej = 1
for i in eachindex(volumes)
    if i in rejectedids
        rectanglesrej_x[cntrrej:cntrrej+5], rectanglesrej_y[cntrrej:cntrrej+5] = createRectangle(volumes[i].spatialvolume, FirstDim, SecondDim)
        cntrrej += 6
    else
        rectangles_x[cntr:cntr+5], rectangles_y[cntr:cntr+5] = createRectangle(volumes[i].spatialvolume, FirstDim, SecondDim)
        cntr += 6
    end
end
plot!(rectangles_x[1:end-1], rectangles_y[1:end-1], c=:blue, linewidth=1.5, label = "Accepted Rectangles")
plot!(rectanglesrej_x[1:end-1], rectanglesrej_y[1:end-1], c=:green, linewidth=1.5, label = "Rejected Rectangles")


pts = dataset.data[:, dataset.startingIDs]
scatter!(pts[FirstDim, :], pts[SecondDim, :], c=:black, markersize=5.0, marker=:rect, linewidth=0.0, label = "Seed Samples")


cubes_x = Array{Float64, 1}(length(volumes) * 6)
cubes_y = Array{Float64, 1}(length(volumes) * 6)
cntr = 1
for i in eachindex(cubes)
    cubes_x[cntr:cntr+5], cubes_y[cntr:cntr+5] = createRectangle(cubes[i], FirstDim, SecondDim)
    cntr += 6
end
plot!(cubes_x[1:end-1], cubes_y[1:end-1], c=:blue, linewidth=1.5, label = "Initial Cubes around Seed Samples")


title!("Visualization of the AHMI Technique")
xlabel!(L"$\lambda_1$")
ylabel!(L"$\lambda_2$")


#space partitioning tree

tree=dataset.partitioningtree

xcutpos = 1:(1 + tree.cuts):length(tree.cutlist)
xcuts=tree.cutlist[xcutpos]

xlimits=(minimum(dataset.data[1, :]), maximum(dataset.data[1, :]))
ylimits=(minimum(dataset.data[2, :]), maximum(dataset.data[2, :]))

push!(xcuts, xlimits[2])


for i=1:tree.cuts
    plot!([xcuts[i], xcuts[i]], [ylimits[1], ylimits[2]], color=:blue, linewidth=1.0)


    ycuts = if i < tree.cuts
        [tree.cutlist[xcutpos[i] + 1:xcutpos[i+1] - 1]..., ylimits[2]]
    else
        [tree.cutlist[xcutpos[i] + 1:length(tree.cutlist)]..., ylimits[2]]
    end

    #push!(ycuts, maximum(dataset.data[tree.leafsize*tree.cuts*(i - 1)+1:tree.leafsize*tree.cuts*(i)]))


    for j=1:tree.cuts
        plot!([xcuts[i], xcuts[i+1]], [ycuts[j], ycuts[j]], color=:green, linewidth=1.0)
    end
    #plot!([xcuts[i], xcuts[i+1]], [ycuts[tree.cuts + 1], ycuts[tree.cuts + 1]], color=:green, linewidth=1.0)
end
plot!([xlimits[2], xlimits[2]], [ylimits[1], ylimits[2]], color=:blue, linewidth=1.0)
