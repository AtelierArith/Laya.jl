# Minimal safetensors reader: 8-byte little-endian header length, JSON header, raw data.
# Tensors are returned with reversed axes (row-major `(out, in)` becomes Julia `(in, out)`),
# so no data is permuted.

const SAFETENSORS_DTYPES = Dict(
    "F64" => Float64, "F32" => Float32, "F16" => Float16,
    "I64" => Int64, "I32" => Int32, "I16" => Int16, "I8" => Int8,
    "U8" => UInt8, "BOOL" => Bool,
)

"""
    load_safetensors(path) -> Dict{String,Array}

Read every tensor in a `.safetensors` file. BF16 tensors are widened to `Float32`.
"""
function load_safetensors(path::AbstractString)
    open(path, "r") do io
        n = Int(ltoh(read(io, UInt64)))
        header = JSON.parse(String(read(io, n)))
        base = 8 + n
        tensors = Dict{String,Array}()
        for (name, info) in header
            name == "__metadata__" && continue
            dtype = String(info["dtype"])
            shape = Tuple(reverse(Int.(info["shape"])))
            start, stop = Int.(info["data_offsets"])
            seek(io, base + start)
            if dtype == "BF16"
                raw = read!(io, Vector{UInt16}(undef, (stop - start) ÷ 2))
                data = reinterpret(Float32, UInt32.(ltoh.(raw)) .<< 16)
            else
                T = get(SAFETENSORS_DTYPES, dtype) do
                    error("Unsupported safetensors dtype $dtype for $name")
                end
                data = ltoh.(read!(io, Vector{T}(undef, (stop - start) ÷ sizeof(T))))
            end
            length(data) == prod(shape) || error("Size mismatch for tensor $name")
            tensors[String(name)] = reshape(data, shape)
        end
        tensors
    end
end
