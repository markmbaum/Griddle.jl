module Griddle

using AxisArrays
using Distributions
using Random: Xoshiro
using StatsBase: ProbabilityWeights, sample
using Base.Iterators: product, ProductIterator, flatten, Zip

export gridapproximation, sampleposterior

const UC = UnivariateDistribution{Continuous}
const UD = UnivariateDistribution{Discrete}

#------------------------------------------------------------------------------
# grid selection for priors, dispatching on named tuple arguments & types

gridlimit(xsupport::𝒯, xq::𝒯) where {𝒯<:Real} = isinf(xsupport) ? xq : xsupport

priorgrid(prior::UD, args...) = prior |> support

function priorgrid(prior::UC; n::Int=101, q::Real=0.99)
    @assert 0 < q < 1 "The quantile used for prior range selection must be greater than zero and less than one, but you chose $q. This number selects the middle qth of the probabibility mass for the grid of prior/parameter values, unless it is bounded on one or both ends."
    range(
        gridlimit(prior |> support |> minimum, quantile(prior, (1 - q)/2)),
        gridlimit(prior |> support |> maximum, quantile(prior, (1 + q)/2)),
        length=n
    ) .|> Float64
end

function priorgrid(t::NamedTuple{(:prior,:n), Tuple{𝒟,ℐ}})::Vector{Float64} where {𝒟 <: UC, ℐ <: Int}
    priorgrid(t.prior, n=t.n)
end

function priorgrid(t::NamedTuple{(:prior,:q), Tuple{𝒟,ℛ}})::Vector{Float64} where {𝒟 <: UC, ℛ <: Real}
    priorgrid(t.prior, q=t.q)
end

function priorgrid(t::NamedTuple{(:prior,:n,:q), Tuple{𝒟,ℐ,ℛ}})::Vector{Float64} where {𝒟 <: UC, ℐ <: Int, ℛ <: Real}
    priorgrid(t.prior, n=t.n, q=t.q)
end

function priorgrid(t::NamedTuple{(:prior,:q,:n), Tuple{𝒟,ℛ,ℐ}})::Vector{Float64} where {𝒟 <: UC, ℛ <: Real, ℐ <: Int}
    priorgrid(t.prior, n=t.n, q=t.q)
end

function priorgrid(t::NamedTuple{(:prior,:grid), Tuple{𝒟,𝒱}})::Vector{Float64} where {𝒟 <: UC, 𝒱 <: AbstractVector}
    t.grid |> collect
end

#------------------------------------------------------------------------------
# dispatching to pull out prior distributions from parameter definitions

extractdistribution(prior::UD) = prior

extractdistribution(prior::UC) = prior

extractdistribution(prior::NamedTuple) = getfield(prior, :prior)

#------------------------------------------------------------------------------
# probability calculation

logprior(dist::UnivariateDistribution, grid)::Vector{Float64} = map(g -> logpdf(dist, g), grid)

function bayes(
        loglikelihood::ℱ,
        grids::ProductIterator{NTuple{𝒩,Vector{Float64}}},
        logpriors::ProductIterator{NTuple{𝒩,Vector{Float64}}}
    ) where {ℱ <: Function, 𝒩}

    #form the output grid by computing the joint log prior probabilities
    P = map(sum, logpriors)

    #density calculations
    for (x, idx) ∈ zip(grids, P |> CartesianIndices)
        @inbounds P[idx] += loglikelihood(x...)
    end
    #take out of log units
    P = map!(exp, P, P)
    #normalize for total mass of 1
    s = 1/sum(P)
    map!(p -> s * p, P, P)

    return P
end

#------------------------------------------------------------------------------
# drivers

#posterior approximation
function gridapproximation(loglikelihood::ℱ; parameters...)::AxisArray where {ℱ <: Function}

    #pull parameter names (as symbols) and their values out of the tuple
    nparam = length(parameters)
    names = parameters |> keys
    vals = parameters |> values
    #extract the distribution structs from the parameter inputs
    dists = map(extractdistribution, vals)
    #form a grid of sample locations for each parameter
    grids = map(priorgrid, vals)
    #evaluate log prob density/mass at all grid locations
    logpriors = ntuple(i -> logprior(dists[i], grids[i]), nparam)

    AxisArray(
        bayes(
            loglikelihood,
            product(grids...),
            product(logpriors...)
        ),
        (Axis{n}(g) for (n,g) ∈ zip(names,grids))...
    )
end

#joint prior approximation
gridapproximation(; parameters...) = gridapproximation((x...) -> 0.; parameters...)

function sampleposterior(post::AxisArray, N::Int=10_000; rng=Xoshiro())::Dict

    #names/symbols of the parameters
    names = map(ax -> typeof(ax).parameters[1], post.axes)
    #a big matrix for samples
    samp = Matrix{Float64}(undef, length(names), N)
    #sampling weights
    weights = post.data |> vec |> ProbabilityWeights
    #multidimensional indices along the axes to sample
    indices = post |> CartesianIndices |> vec |> collect
    for r ∈ eachcol(samp)
        idx = Tuple(sample(rng, indices, weights))
        for i ∈ eachindex(idx)
            r[i] = post.axes[i][idx[i]]
        end
    end

    Dict(zip(names, samp |> eachrow))
end

end
