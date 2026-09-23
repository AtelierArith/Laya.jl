"""
    LayaMLXReference

Test-only bridge to the Python `laya_mlx` implementation in `extern/laya-mlx`, used to
produce ground truth for the pure-Julia port.

Array convention: a NumPy array of shape `(b, L, d)` is returned as a Julia array of size
`(d, L, b)` with the same memory order, i.e. axes are reversed. This matches handing a
column-major Julia array to MLX with its shape reversed.

Token ids and marker positions are returned exactly as Python produces them (0-based).
"""
module LayaMLXReference

using JSON
using PythonCall

const laya_ref = PythonCall.pynew()

function __init__()
    sys = pyimport("sys")
    sys.path.insert(0, joinpath(@__DIR__, "..", "python"))
    PythonCall.pycopy!(laya_ref, pyimport("laya_ref"))
end

struct ReferenceAgent
    py::Py
end

Base.show(io::IO, a::ReferenceAgent) = print(io, "ReferenceAgent(", pyconvert(String, a.py.model_id), ")")

"""
    load(path; dtype="float32", device="gpu") -> ReferenceAgent

Load a checkpoint with the Python `laya_mlx.Agent`. `path` may be a local directory or a
Hugging Face repository id.
"""
load(path; dtype::AbstractString="float32", device::AbstractString="gpu") =
    ReferenceAgent(laya_ref.load_agent(string(path), dtype, device))

"""
    load_tokenizer(dir) -> ReferenceAgent

Only the Rust tokenizer of a local checkpoint directory; supports `tokenize` and
`special_tokens` without loading any weights.
"""
load_tokenizer(dir) = ReferenceAgent(laya_ref.load_tokenizer(string(dir)))

"""Rust-tokenizer ids for `text`, without special tokens."""
tokenize(a::ReferenceAgent, text::AbstractString) = pyconvert(Vector{Int}, laya_ref.tokenize(a.py, text))

"""Special tokens as `Dict(name => (token=..., id=...))`."""
function special_tokens(a::ReferenceAgent)
    d = pyconvert(Dict{String,Dict{String,Any}}, laya_ref.special_tokens(a.py))
    Dict(k => (token=String(v["token"]), id=Int(v["id"])) for (k, v) in d)
end

"""
    prepare(a, state, questions) -> Vector{NamedTuple}

Tokenized sequences `(ids, markers, qtype)`, one per question, in question order.
`state` and `questions` are serialized with JSON.jl before crossing into Python.
"""
function prepare(a::ReferenceAgent, state, questions)
    raw = JSON.parse(pyconvert(String, laya_ref.prepare(a.py, JSON.json(state), JSON.json(questions))))
    [(ids=Vector{Int}(i["ids"]), markers=Vector{Int}(i["markers"]), qtype=Int(i["qtype"])) for i in raw]
end

"""
    collate(a, items; pad_to_multiple=nothing) -> Dict{String,Array}

Padded batch tensors, axes reversed (e.g. `input_ids` is `(L, n)`).
"""
function collate(a::ReferenceAgent, items; pad_to_multiple=nothing)
    batch = laya_ref.collate(a.py, JSON.json(items), pad_to_multiple)
    Dict(pyconvert(String, k) => to_julia(v) for (k, v) in batch.items())
end

"""
    trace(a, batch) -> Dict{String,Array{Float32 or Bool}}

Every intermediate activation of one forward pass: `embeddings`, `layer_i`, `encoder`,
`typed`, `head_i`, `logits`, `action`, plus the attention masks. Activations are cast to
Float32 on the Python side.
"""
function trace(a::ReferenceAgent, batch::AbstractDict)
    out = laya_ref.trace(a.py, pydict(Dict(k => v for (k, v) in batch)))
    Dict(pyconvert(String, k) => to_julia(v) for (k, v) in out.items())
end

"""Full `predict` result as parsed JSON."""
predict(a::ReferenceAgent, state, questions) =
    JSON.parse(pyconvert(String, laya_ref.predict(a.py, JSON.json(state), JSON.json(questions))))

"""Write the tiny random checkpoint from `tests/conftest.py` to `path` (must not exist)."""
tiny_checkpoint(path; seed::Integer=7) = pyconvert(String, laya_ref.tiny_checkpoint(string(path), seed))

"""Convert a C-contiguous NumPy array to a Julia array with reversed axes."""
function to_julia(x::Py)
    a = PyArray(x)
    N = ndims(a)
    N <= 1 ? Array(a) : permutedims(Array(a), N:-1:1)
end

end # module LayaMLXReference
