# Scalar math that must match MLX's Metal kernels. Written with Float32 `fma` only so the
# same functions can later run inside Metal.jl kernels.
#
# Ported from MLX v0.32.2 `mlx/backend/metal/kernels/{erf,expm1f}.h` (MIT, Apple Inc.),
# which in turn follow Norbert Juffa's faithfully rounded erff/expm1f.

@inline function mlx_expm1f(a::Float32)
    j = fma(1.442695f0, a, 12582912.0f0)
    j = j - 12582912.0f0
    i = unsafe_trunc(Int32, j)
    f = fma(j, -6.93145752f-1, a)
    s = a == 0.0f0 ? a : f * f
    r = 1.97350979f-4
    r = fma(r, f, 1.39309070f-3)
    r = fma(r, f, 8.33343994f-3)
    r = fma(r, f, 4.16668020f-2)
    r = fma(r, f, 1.66666716f-1)
    r = fma(r, f, 4.99999970f-1)
    u = j == 1.0f0 ? f + 0.5f0 : f
    v = fma(r, s, u)
    s = 0.5f0
    t = ldexp(s, i)
    y = t - s
    x = (t - y) - s
    r = fma(v, t, x) + y
    r = r + r
    j == 0.0f0 && (r = v)
    j == 1.0f0 && (r = v + v)
    if abs(a - 1.0f0) > 88.0f0
        r = exp2(a)
        r = fma(r, r, -1.0f0)
    end
    return r
end

@inline function mlx_erf(a::Float32)
    t = abs(a)
    s = a * a
    if t > 0.927734375f0
        r = fma(-1.72853470f-5, t, 3.83197126f-4)
        u = fma(-3.88396438f-3, t, 2.42546219f-2)
        r = fma(r, s, u)
        r = fma(r, t, -1.06777877f-1)
        r = fma(r, t, -6.34846687f-1)
        r = fma(r, t, -1.28717512f-1)
        r = fma(r, t, -t)
        r = -mlx_expm1f(r)
        return copysign(r, a)
    else
        r = -5.96761703f-4
        r = fma(r, s, 4.99119423f-3)
        r = fma(r, s, -2.67681349f-2)
        r = fma(r, s, 1.12819925f-1)
        r = fma(r, s, -3.76125336f-1)
        r = fma(r, s, 1.28379166f-1)
        return fma(r, a, a)
    end
end

mlx_erf(a::Real) = oftype(float(a), mlx_erf(Float32(a)))

"""Exact (erf-based) GELU, as `mlx.nn.gelu`: `x * (1 + erf(x / √2)) / 2`."""
@inline gelu(x::T) where {T<:Real} = x * (one(T) + mlx_erf(x / T(sqrt(2)))) / T(2)

@inline relu(x::T) where {T<:Real} = max(x, zero(T))
