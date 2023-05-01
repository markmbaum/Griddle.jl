using Griddle
using Griddle: priorgrid
using Distributions
using Random: Xoshiro
using Test

@testset "Priors/Parameters" begin

    g = priorgrid(Uniform())
    @test minimum(g) == 0
    @test maximum(g) == 1

    g = priorgrid(Exponential())
    @test minimum(g) == 0

    g = priorgrid(Normal())
    @test minimum(g) ≈ -maximum(g)

    g = priorgrid((prior=Normal(), n=10, q=0.5))
    @test length(g) == 10
    @test minimum(g) ≈ -0.6744897501960818
    @test maximum(g) ≈ 0.6744897501960818

    g = priorgrid(Bernoulli())
    @test length(g) == 2
    @test all(g .== (0,1))

    g = priorgrid(Binomial(10))
    @test length(g) == 11
    @test all(g .== 0:10)

    g = priorgrid((prior=Uniform(), grid=LinRange(0, 1, 10)))
    @test all(g .== LinRange(0, 1, 100))

end

@testset "Inference" begin

    post = gridapproximation(
        p -> logpdf(Binomial(2,p), 1),
        p = Uniform()
    )
    p = post.axes[1] |> collect
    exact = pdf.(Beta(2,2), p)
    @test all(post.data .≈ (exact / sum(exact)))

end

@testset "Sampling" begin

    rng = Xoshiro(1)
    x = randn(rng, 16)
    n = length(x)
    β = 0.5
    y = β*x .+ randn(rng, n)/5

    post = gridapproximation(
        (β,σ) -> logpdf.(Normal.(β*x, σ), y) |> sum,
        β = Normal(),
        σ = Exponential()
    )

    samp = sampleposterior(post, rng=rng)

    @test 0.45 < median(samp[:β]) < 0.55
    @test 0.1 < median(samp[:σ]) < 0.3

end