# Thin high-level layer over LibMLX.
#
# Conventions: an MLX array of shape (b, L, d) (row-major) shares its memory layout with a
# Julia array of size (d, L, b). `MLXArray(::Array)` and `Array(::MLXArray)` reverse the
# shape and never permute data. All ops below take *MLX* axes (0-based, negative allowed)
# and MLX shapes, so model code reads like the Python original; `shape(x)` is the MLX
# shape and `size(x)` its reverse (the Julia view).

module MX

using ..LibMLX
using ..LibMLX: mlx_array, mlx_stream, mlx_vector_array, mlx_dtype

export MLXArray, MLXError, BFloat16, shape, astype, eval!, set_device!, with_device

struct MLXError <: Exception
    msg::String
end
Base.showerror(io::IO, e::MLXError) = print(io, "MLXError: ", e.msg)

const LAST_ERROR = Ref("")

function _error_handler(msg::Cstring, ::Ptr{Cvoid})::Cvoid
    LAST_ERROR[] = unsafe_string(msg)
    return nothing
end

__init__() = LibMLX.mlx_set_error_handler(
    @cfunction(_error_handler, Cvoid, (Cstring, Ptr{Cvoid})), C_NULL, C_NULL)

"""Throw an `MLXError` carrying the captured mlx-c message when `status != 0`."""
@inline function check(status::Integer)
    status == 0 && return nothing
    msg = LAST_ERROR[]
    LAST_ERROR[] = ""
    throw(MLXError(isempty(msg) ? "mlx-c call failed with status $status" : msg))
end

# ---------------------------------------------------------------------------- streams

const STREAMS = Dict{Symbol,mlx_stream}()
const CURRENT = Ref(:gpu)

function stream(dev::Symbol=CURRENT[])
    get!(STREAMS, dev) do
        dev === :gpu ? LibMLX.mlx_default_gpu_stream_new() :
        dev === :cpu ? LibMLX.mlx_default_cpu_stream_new() :
        throw(ArgumentError("device must be :gpu or :cpu, got $dev"))
    end
end

"""Set the device (`:gpu` or `:cpu`) used by subsequent ops."""
set_device!(dev::Symbol) = (stream(dev); CURRENT[] = dev)

"""Run `f()` with ops dispatched to `dev`, restoring the previous device afterwards."""
function with_device(f, dev::Symbol)
    old = CURRENT[]
    set_device!(dev)
    try
        f()
    finally
        CURRENT[] = old
    end
end

synchronize(dev::Symbol=CURRENT[]) = check(LibMLX.mlx_synchronize(stream(dev)))

# ---------------------------------------------------------------------------- memory

"""Release MLX's buffer cache (`mx.clear_cache()`)."""
clear_cache() = check(LibMLX.mlx_clear_cache())
for (f, c) in ((:active_memory, :mlx_get_active_memory), (:peak_memory, :mlx_get_peak_memory),
               (:cache_memory, :mlx_get_cache_memory))
    @eval $f() = (r = Ref{Csize_t}(0); check(LibMLX.$c(r)); Int(r[]))
end
reset_peak_memory() = check(LibMLX.mlx_reset_peak_memory())

"""MLX version string reported by the linked libmlx (e.g. "0.32.2")."""
function version()
    s = Ref(LibMLX.mlx_string_new())
    try
        check(LibMLX.mlx_version(s))
        unsafe_string(LibMLX.mlx_string_data(s[]))
    finally
        LibMLX.mlx_string_free(s[])
    end
end

# ---------------------------------------------------------------------------- arrays

mutable struct MLXArray
    h::mlx_array
    function MLXArray(h::mlx_array)
        h.ctx == C_NULL && throw(MLXError("null mlx_array"))
        x = finalizer(free!, new(h))
        arena = get(task_local_storage(), :mlx_arena, nothing)
        arena === nothing || push!(arena::Vector{MLXArray}, x)
        x
    end
end

"""Release the array handle now (idempotent; the finalizer does the same later)."""
function free!(x::MLXArray)
    if x.h.ctx != C_NULL
        LibMLX.mlx_array_free(x.h)
        x.h = mlx_array(C_NULL)
    end
    nothing
end

"""
    scoped(f)

Run `f()` and free every `MLXArray` created meanwhile (by this task) when it returns.
Julia's GC does not see the device memory behind an `MLXArray`, so without this the
intermediates of a forward pass pile up until a collection happens. `f` must return plain
Julia data (or arrays created outside the scope); arrays created inside are invalid after.
"""
function scoped(f)
    tls = task_local_storage()
    outer = get(tls, :mlx_arena, nothing)
    arena = MLXArray[]
    tls[:mlx_arena] = arena
    try
        f()
    finally
        outer === nothing ? delete!(tls, :mlx_arena) : (tls[:mlx_arena] = outer)
        foreach(free!, arena)
    end
end

# Passing an MLXArray to a ccall expecting `mlx_array` keeps it rooted during the call.
Base.cconvert(::Type{mlx_array}, x::MLXArray) = x
function Base.unsafe_convert(::Type{mlx_array}, x::MLXArray)
    x.h.ctx == C_NULL && throw(MLXError("use of a freed MLXArray"))
    x.h
end

const NULL_ARRAY = mlx_array(C_NULL)   # "may be null" optional array arguments

"""Call an mlx-c op `f(res, args..., stream)` and wrap the result."""
function op(f, args...)
    res = Ref(NULL_ARRAY)
    check(f(res, args..., stream()))
    MLXArray(res[])
end

# dtypes
struct BFloat16Type end
"""Marker for MLX's bfloat16 (no native Julia type); use with `astype`."""
const BFloat16 = BFloat16Type()

const DTYPES = Dict{Any,mlx_dtype}(
    Bool => LibMLX.MLX_BOOL, UInt8 => LibMLX.MLX_UINT8, UInt32 => LibMLX.MLX_UINT32,
    Int8 => LibMLX.MLX_INT8, Int16 => LibMLX.MLX_INT16, Int32 => LibMLX.MLX_INT32,
    Int64 => LibMLX.MLX_INT64, Float16 => LibMLX.MLX_FLOAT16, Float32 => LibMLX.MLX_FLOAT32,
    Float64 => LibMLX.MLX_FLOAT64, BFloat16 => LibMLX.MLX_BFLOAT16,
)
const JLTYPES = Dict(v => k for (k, v) in DTYPES)

mlxdtype(T) = get(() -> throw(ArgumentError("unsupported element type $T")), DTYPES, T)
mlxdtype(d::mlx_dtype) = d
dtype(x::MLXArray) = LibMLX.mlx_array_dtype(x)
"""Julia element type of `x` (`BFloat16` marker for bfloat16)."""
Base.eltype(x::MLXArray) = JLTYPES[dtype(x)]
isfloating(d::mlx_dtype) = d in (LibMLX.MLX_FLOAT16, LibMLX.MLX_FLOAT32, LibMLX.MLX_FLOAT64, LibMLX.MLX_BFLOAT16)

Base.ndims(x::MLXArray) = Int(LibMLX.mlx_array_ndim(x))
Base.length(x::MLXArray) = Int(LibMLX.mlx_array_size(x))
"""MLX (row-major) shape."""
function shape(x::MLXArray)
    p = LibMLX.mlx_array_shape(x)
    ntuple(i -> Int(unsafe_load(p, i)), ndims(x))
end
"""Julia size, i.e. the reversed MLX shape."""
Base.size(x::MLXArray) = reverse(shape(x))
Base.size(x::MLXArray, i::Integer) = size(x)[i]

"""Copy a Julia array into MLX with reversed shape (no permutation)."""
function MLXArray(a::AbstractArray{T}) where {T}
    a = convert(Array{T}, a)  # dense, column-major
    shp = Cint[reverse(size(a))...]
    h = GC.@preserve a shp LibMLX.mlx_array_new_data(pointer(a), pointer(shp), length(shp), mlxdtype(T))
    MLXArray(h)
end
MLXArray(x::MLXArray) = x

"""A 0-d MLX array holding `v` converted directly to `T` (or bfloat16)."""
scalar(v, T::Type) = MLXArray(fill(convert(T, v)))
scalar(v, ::BFloat16Type) = astype(scalar(v, Float32), BFloat16)
scalar(v, d::mlx_dtype) = scalar(v, JLTYPES[d])

function Base.show(io::IO, x::MLXArray)
    print(io, "MLXArray{", eltype(x), "} shape=", shape(x), " (Julia size ", size(x), ")")
end

eval!(x::MLXArray) = (check(LibMLX.mlx_array_eval(x)); x)
function eval!(xs::MLXArray...)
    v = vector(collect(xs))
    try
        check(LibMLX.mlx_eval(v))
    finally
        LibMLX.mlx_vector_array_free(v)
    end
    xs
end

const DATA = Dict{mlx_dtype,Function}(
    LibMLX.MLX_BOOL => LibMLX.mlx_array_data_bool, LibMLX.MLX_UINT8 => LibMLX.mlx_array_data_uint8,
    LibMLX.MLX_UINT32 => LibMLX.mlx_array_data_uint32, LibMLX.MLX_INT8 => LibMLX.mlx_array_data_int8,
    LibMLX.MLX_INT16 => LibMLX.mlx_array_data_int16, LibMLX.MLX_INT32 => LibMLX.mlx_array_data_int32,
    LibMLX.MLX_INT64 => LibMLX.mlx_array_data_int64, LibMLX.MLX_FLOAT16 => LibMLX.mlx_array_data_float16,
    LibMLX.MLX_FLOAT32 => LibMLX.mlx_array_data_float32, LibMLX.MLX_FLOAT64 => LibMLX.mlx_array_data_float64,
)

"""Copy to a Julia `Array` of size `reverse(shape(x))`. bfloat16 is returned as Float32."""
function Base.Array(x::MLXArray)
    dtype(x) == LibMLX.MLX_BFLOAT16 && return Array(astype(x, Float32))
    y = eval!(op(LibMLX.mlx_contiguous, x, false))
    T = eltype(y)
    out = Array{T}(undef, size(y))
    GC.@preserve y begin
        p = Ptr{T}(DATA[dtype(y)](y))
        p == C_NULL && throw(MLXError("array has no data"))
        unsafe_copyto!(pointer(out), p, length(out))
    end
    out
end
Base.Array{T}(x::MLXArray) where {T} = convert(Array{T}, Array(x))
Base.collect(x::MLXArray) = Array(x)
item(x::MLXArray) = (length(x) == 1 || throw(ArgumentError("item of non-scalar")); only(Array(x)))

# vector_array helpers (caller frees)
vector(xs::AbstractVector{MLXArray}) =
    GC.@preserve xs LibMLX.mlx_vector_array_new_data([Base.unsafe_convert(mlx_array, x) for x in xs], length(xs))

function unvector(v::mlx_vector_array)
    map(0:Int(LibMLX.mlx_vector_array_size(v))-1) do i
        res = Ref(NULL_ARRAY)
        check(LibMLX.mlx_vector_array_get(res, v, i))
        MLXArray(res[])
    end
end

function with_vector(f, xs)
    v = vector(collect(MLXArray, xs))
    try
        f(v)
    finally
        LibMLX.mlx_vector_array_free(v)
    end
end

# ---------------------------------------------------------------------------- ops

cints(v) = Cint[v...]

astype(x::MLXArray, T) = dtype(x) == mlxdtype(T) ? x : op(LibMLX.mlx_astype, x, mlxdtype(T))

# Python-style weak scalars: a Real adopts the array's dtype (float scalars promote ints to Float32).
lift(x::MLXArray, y::MLXArray) = y
function lift(x::MLXArray, y::Real)
    d = dtype(x)
    isfloating(d) || !(y isa AbstractFloat) ? scalar(y, d) : scalar(y, Float32)
end

for (jf, cf) in ((:+, :mlx_add), (:-, :mlx_subtract), (:*, :mlx_multiply), (:/, :mlx_divide),
                 (:maximum, :mlx_maximum), (:minimum, :mlx_minimum), (:less, :mlx_less),
                 (:less_equal, :mlx_less_equal), (:greater, :mlx_greater),
                 (:greater_equal, :mlx_greater_equal), (:equal, :mlx_equal),
                 (:logical_and, :mlx_logical_and), (:logical_or, :mlx_logical_or))
    f = jf in (:+, :-, :*, :/) ? :(Base.$jf) : jf
    @eval begin
        $f(a::MLXArray, b::MLXArray) = op(LibMLX.$cf, a, b)
        $f(a::MLXArray, b::Real) = op(LibMLX.$cf, a, lift(a, b))
        $f(a::Real, b::MLXArray) = op(LibMLX.$cf, lift(b, a), b)
    end
end
Base.:&(a::MLXArray, b::MLXArray) = logical_and(a, b)
Base.:|(a::MLXArray, b::MLXArray) = logical_or(a, b)

for (jf, cf) in ((:-, :mlx_negative), (:abs, :mlx_abs), (:log, :mlx_log), (:exp, :mlx_exp),
                 (:erf, :mlx_erf), (:sqrt, :mlx_sqrt), (:logical_not, :mlx_logical_not))
    f = jf in (:-, :abs, :log, :exp, :sqrt) ? :(Base.$jf) : jf
    @eval $f(a::MLXArray) = op(LibMLX.$cf, a)
end
Base.:~(a::MLXArray) = logical_not(a)
Base.:!(a::MLXArray) = logical_not(a)

where(c::MLXArray, a, b) = (a isa Real && b isa Real) ? throw(ArgumentError("where needs an array branch")) :
    a isa Real ? op(LibMLX.mlx_where, c, lift(b, a), b) :
    b isa Real ? op(LibMLX.mlx_where, c, a, lift(a, b)) : op(LibMLX.mlx_where, c, a, b)

matmul(a::MLXArray, b::MLXArray) = op(LibMLX.mlx_matmul, a, b)
"""`alpha * (a @ b) + beta * c`"""
addmm(c::MLXArray, a::MLXArray, b::MLXArray; alpha=1f0, beta=1f0) =
    op(LibMLX.mlx_addmm, c, a, b, Float32(alpha), Float32(beta))

reshape(x::MLXArray, shp...) = (s = cints(shp); op(LibMLX.mlx_reshape, x, s, length(s)))
"""Permute MLX axes (0-based); without `perm`, reverse all axes."""
transpose(x::MLXArray, perm...) = isempty(perm) ? op(LibMLX.mlx_transpose, x) :
    (p = cints(perm); op(LibMLX.mlx_transpose_axes, x, p, length(p)))
expand_dims(x::MLXArray, axes...) = (a = cints(axes); op(LibMLX.mlx_expand_dims_axes, x, a, length(a)))
squeeze(x::MLXArray, axis::Integer) = op(LibMLX.mlx_squeeze_axis, x, Cint(axis))
broadcast_to(x::MLXArray, shp...) = (s = cints(shp); op(LibMLX.mlx_broadcast_to, x, s, length(s)))

"""
Python-style slice along the leading MLX axes: `start`/`stop` per axis (0-based, stop
exclusive, negatives count from the end); trailing axes are kept whole.
"""
function slice(x::MLXArray, start, stop, strides=ones(Int, length(start)))
    shp = shape(x); n = length(shp); m = length(start)
    (length(stop) == m == length(strides) && m <= n) || throw(ArgumentError("slice: bad index lengths"))
    norm(v, d) = clamp(v < 0 ? v + d : v, 0, d)
    s = cints([norm.(start, shp[1:m]); zeros(Int, n - m)])
    e = cints([norm.(stop, shp[1:m]); collect(shp[m+1:n])])
    t = cints([strides; ones(Int, n - m)])
    op(LibMLX.mlx_slice, x, s, length(s), e, length(e), t, length(t))
end
"""`x[..., i, ...]` along `axis` (drops the axis)."""
function index(x::MLXArray, axis::Integer, i::Integer)
    n = ndims(x); axis = mod(axis, n); shp = shape(x)
    start = zeros(Int, n); stop = collect(shp)
    start[axis+1] = i < 0 ? i + shp[axis+1] : i
    stop[axis+1] = start[axis+1] + 1
    squeeze(slice(x, start, stop), axis)
end

take(x::MLXArray, idx::MLXArray, axis::Integer) = op(LibMLX.mlx_take_axis, x, idx, Cint(axis))
take(x::MLXArray, idx::MLXArray) = op(LibMLX.mlx_take, x, idx)
"""Rows of `weight` (vocab, d) at integer `ids`: shape(ids) × d."""
embedding(weight::MLXArray, ids::MLXArray) = take(weight, ids, 0)

sum(x::MLXArray; axis::Integer, keepdims::Bool=false) = op(LibMLX.mlx_sum_axis, x, Cint(axis), keepdims)
softmax(x::MLXArray; axis::Integer=-1, precise::Bool=false) = op(LibMLX.mlx_softmax_axis, x, Cint(axis), precise)
sort(x::MLXArray; axis::Integer=-1) = op(LibMLX.mlx_sort_axis, x, Cint(axis))

function split(x::MLXArray, n::Integer; axis::Integer=-1)
    res = Ref(LibMLX.mlx_vector_array_new())
    try
        check(LibMLX.mlx_split(res, x, Cint(n), Cint(axis), stream()))
        unvector(res[])
    finally
        LibMLX.mlx_vector_array_free(res[])
    end
end
concatenate(xs; axis::Integer=0) = with_vector(v -> op(LibMLX.mlx_concatenate_axis, v, Cint(axis)), xs)
stack(xs; axis::Integer=0) = with_vector(v -> op(LibMLX.mlx_stack_axis, v, Cint(axis)), xs)

arange(start, stop, step=1; dtype=Int32) = op(LibMLX.mlx_arange, Float64(start), Float64(stop), Float64(step), mlxdtype(dtype))
arange(n::Integer; dtype=Int32) = arange(0, n; dtype)

# ---------------------------------------------------------------------------- compile

const KEEP = Any[]   # payloads referenced from C closures

function _closure_cb(res::Ptr{mlx_vector_array}, input::mlx_vector_array, payload::Ptr{Cvoid})::Cint
    try
        f = unsafe_pointer_to_objref(payload)[]
        ys = f(unvector(input)...)
        with_vector(v -> check(LibMLX.mlx_vector_array_set(res, v)), ys isa MLXArray ? (ys,) : ys)
        return Cint(0)
    catch e
        LAST_ERROR[] = sprint(showerror, e)
        return Cint(1)
    end
end

"""An `mx.compile`d function of MLXArrays (one compiled closure per device)."""
struct Compiled
    f::Any
    shapeless::Bool
    closures::Dict{Symbol,LibMLX.mlx_closure}
end

"""`mx.compile(f; shapeless)`: `f` maps MLXArrays to an MLXArray or a tuple of them."""
compile(f; shapeless::Bool=false) = Compiled(f, shapeless, Dict{Symbol,LibMLX.mlx_closure}())

function closure(c::Compiled)
    get!(c.closures, CURRENT[]) do
        payload = Ref{Any}(c.f)
        push!(KEEP, payload)
        cb = @cfunction(_closure_cb, Cint, (Ptr{mlx_vector_array}, mlx_vector_array, Ptr{Cvoid}))
        raw = LibMLX.mlx_closure_new_func_payload(cb, pointer_from_objref(payload), C_NULL)
        res = Ref(LibMLX.mlx_closure_new())
        try
            check(LibMLX.mlx_compile(res, raw, c.shapeless))
        finally
            LibMLX.mlx_closure_free(raw)
        end
        res[]
    end
end

function (c::Compiled)(xs::MLXArray...)
    cls = closure(c)
    out = Ref(LibMLX.mlx_vector_array_new())
    try
        with_vector(v -> check(LibMLX.mlx_closure_apply(out, cls, v)), xs)
        ys = unvector(out[])
        length(ys) == 1 ? only(ys) : Tuple(ys)
    finally
        LibMLX.mlx_vector_array_free(out[])
    end
end

# `mlx.nn.relu` / `mlx.nn.gelu` are `mx.compile(shapeless=True)`d; compiling them here too
# makes the fused kernels (and hence rounding) identical to Python's.
relu_eager(x::MLXArray) = maximum(x, 0)
"""Exact (erf) GELU, same expression as `mlx.nn.gelu`."""
gelu_eager(x::MLXArray) = x * (1 + erf(x / sqrt(2.0))) / 2
const RELU = compile(relu_eager; shapeless=true)
const GELU = compile(gelu_eager; shapeless=true)
relu(x::MLXArray) = RELU(x)
gelu(x::MLXArray) = GELU(x)

layer_norm(x::MLXArray, w, b, eps) =
    op(LibMLX.mlx_fast_layer_norm, x, something(w, NULL_ARRAY), something(b, NULL_ARRAY), Float32(eps))

"""`mx.fast.rope(x, dims, traditional, base, scale, offset)` over the last axis."""
rope(x::MLXArray, dims::Integer; traditional::Bool=false, base::Real, scale::Real=1, offset::Integer=0) =
    op(LibMLX.mlx_fast_rope, x, Cint(dims), traditional, LibMLX.mlx_optional_float(Float32(base), true),
       Float32(scale), Cint(offset), NULL_ARRAY)

"""`mx.fast.scaled_dot_product_attention` with an optional boolean/additive mask array."""
sdpa(q::MLXArray, k::MLXArray, v::MLXArray; scale::Real, mask::Union{Nothing,MLXArray}=nothing) =
    op(LibMLX.mlx_fast_scaled_dot_product_attention, q, k, v, Float32(scale),
       mask === nothing ? "" : "array", something(mask, NULL_ARRAY), NULL_ARRAY, false)

"""Load a safetensors file into `Dict(name => MLXArray)` (read on the CPU stream)."""
function load_safetensors(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    arrays = Ref(LibMLX.mlx_map_string_to_array_new())
    meta = Ref(LibMLX.mlx_map_string_to_string_new())
    out = Dict{String,MLXArray}()
    try
        check(LibMLX.mlx_load_safetensors(arrays, meta, path, stream(:cpu)))
        it = LibMLX.mlx_map_string_to_array_iterator_new(arrays[])
        try
            key = Ref{Ptr{Cchar}}(C_NULL)
            while true
                val = Ref(NULL_ARRAY)
                st = LibMLX.mlx_map_string_to_array_iterator_next(key, val, it)
                st == 2 && break
                check(st)
                out[unsafe_string(key[])] = MLXArray(val[])
            end
        finally
            LibMLX.mlx_map_string_to_array_iterator_free(it)
        end
    finally
        LibMLX.mlx_map_string_to_array_free(arrays[])
        LibMLX.mlx_map_string_to_string_free(meta[])
    end
    out
end

end # module MX
