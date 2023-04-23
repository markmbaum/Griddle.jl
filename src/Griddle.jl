module Griddle

using AxisArrays
using Distributions
using Random: Xoshiro
using StatsBase: ProbabilityWeights, sample
using Base.Iterators: product, ProductIterator, flatten

export prior, gridapproximation, sampleposterior

const UC = UnivariateDistribution{Continuous}
const UD = UnivariateDistribution{Discrete}


gridlimit(xsupport::𝒯, xq::𝒯) where {𝒯<:Real} = isinf(xsupport) ? xq : xsupport

priorgrid(prior::UD, args...) = prior |> support

function priorgrid(prior::UC; n::Int=101, q::Real=0.5e-2)
    @assert 0 < q < 1 "The quantile used for prior range selection must be greater than zero and less than one, but you chose $q. This number should be a small number close to zero, like q=1e-2, which will select a range of values between the prior distribution's qth and (1-q)th quantiles, unless it is bounded on one or both ends."
    range(
        gridlimit(prior |> support |> minimum, quantile(prior, q)),
        gridlimit(prior |> support |> maximum, quantile(prior, 1 - q)),
        length=n
    )
end

function priorgrid(t::NamedTuple{(:prior,:n), Tuple{𝒟,ℐ}}) where {𝒟 <: UC, ℐ <: Int}
    priorgrid(t.prior, n=t.n)
end

function priorgrid(t::NamedTuple{(:prior,:q), Tuple{𝒟,ℛ}}) where {𝒟 <: UC, ℛ <: Real}
    priorgrid(t.prior, q=t.q)
end

function priorgrid(t::NamedTuple{(:prior,:n,:q), Tuple{𝒟,ℐ,ℛ}}) where {𝒟 <: UC, ℐ <: Int, ℛ <: Real}
    priorgrid(t.prior, n=t.n, q=t.q)
end

extractdistribution(prior::UD) = prior

extractdistribution(prior::UC) = prior

extractdistribution(prior::NamedTuple) = prior.prior

logprior(dist::UnivariateDistribution, grid) = map(g -> logpdf(dist, g), grid)


function _gridapproximation(
        loglikelihood::ℱ,
        grid::ProductIterator,
        logp::ProductIterator{NTuple{𝒩,Vector{Float64}}}
    ) where {𝒩, ℱ <: Function}

    @assert size(grid) == size(logp)
    #big fat labeled array for the posterior
    post = zeros(size(logp))
    #density calculations
    indices = CartesianIndices(post)
    for (g, l, idx) ∈ zip(grid, logp, indices)
        post[idx] = exp(sum(l) + loglikelihood(g...))
    end
    return post / sum(post)
end

function gridapproximation(loglikelihood::ℱ; params...)::AxisArray where {ℱ <: Function}

    name = params |> keys
    vals = params |> values
    grid = map(priorgrid, vals)
    dist = map(extractdistribution, vals)
    logp = map(x -> logprior(x...), zip(dist,grid))
    AxisArray(
        _gridapproximation(
            loglikelihood,
            product(grid...),
            product(logp...)
        ),
        [Axis{n}(g) for (n,g) ∈ zip(name,grid)]...
    )
end

function sampleposterior(post::AxisArray, N::Int=10_000; rng=Xoshiro())::Dict

    #names/symbols of the parameters
    names = map(ax -> typeof(ax).parameters[1], post.axes)
    #a big matrix for samples
    samp = Matrix{Float64}(undef, N, length(names))
    #sampling weights
    weights = post.data |> vec |> ProbabilityWeights
    #multidimensional indices along the axes to sample
    indices = post |> CartesianIndices |> vec |> collect

    for r ∈ eachrow(samp)
        idx = Tuple(sample(rng, indices, weights))
        for i ∈ eachindex(idx)
            r[i] = post.axes[i][idx[i]]
        end
    end

    #return DataFrame(samp, names |> collect)
    return Dict(zip(names, samp |> eachcol))

end

end
