using Laya: mlx_erf, gelu, rope, softmax
using PythonCall

# MLX's own GPU erf, evaluated on the reference Python environment.
function mlx_erf_gpu(xs::Vector{Float32})
    mx, np = pyimport("mlx.core"), pyimport("numpy")
    y = mx.erf(mx.array(np.asarray(xs)), stream=mx.gpu)
    pyconvert(Vector{Float32}, np.asarray(y))
end

@testset "mlx_erf" begin
    xs = Float32.(range(-6, 6; length=20001))
    @test mlx_erf.(xs) == mlx_erf_gpu(xs)   # bitwise identical to MLX's Metal kernel
    @test mlx_erf(0.0f0) === 0.0f0
    @test mlx_erf(-0.0f0) === -0.0f0
    @test mlx_erf(10.0f0) == 1.0f0
end

@testset "rope" begin
    x = randn(Float32, 8, 2, 5, 3)
    y = rope(x, 10000)
    @test y[:, :, 1, :] ≈ x[:, :, 1, :]                 # position 0 is not rotated
    @test sum(abs2, y; dims=1) ≈ sum(abs2, x; dims=1)   # rotations preserve norms
end

@testset "softmax" begin
    x = randn(Float32, 7, 4)
    @test all(sum(softmax(x); dims=1) .≈ 1)
end
