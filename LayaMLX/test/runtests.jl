using LayaMLX
using LayaMLX: MX
using LayaMLX.MX: MLXArray, shape, astype
using LayaMLXReference
using SpecialFunctions: erf
using Test

const R = LayaMLXReference
maxdiff(a, b) = maximum(abs.(Float64.(a) .- Float64.(b)))

# Julia-side reference implementations, written on Julia-layout arrays.
jl_layernorm(x, w, b, eps) = (μ = sum(x; dims=1) / size(x, 1);
    v = sum(abs2, x .- μ; dims=1) / size(x, 1); (x .- μ) ./ sqrt.(v .+ eps) .* w .+ b)
function jl_rope(x, base)  # x: (hd, L, ...) Julia layout; non-traditional RoPE, offset 0
    hd = size(x, 1); h = hd ÷ 2
    y = similar(x)
    for I in CartesianIndices(size(x)[2:end]), i in 1:h
        θ = (I[1] - 1) * Float64(base)^(-(i - 1) / h)
        a, b = x[i, I], x[i+h, I]
        y[i, I] = a * cos(θ) - b * sin(θ)
        y[i+h, I] = a * sin(θ) + b * cos(θ)
    end
    y
end
function jl_sdpa(q, k, v, scale, mask)  # (hd, L, H, b) Julia layout, mask (Lk, Lq, 1, b)
    out = similar(q)
    for bi in axes(q, 4), hi in axes(q, 3)
        s = (k[:, :, hi, bi]' * q[:, :, hi, bi]) .* scale          # (Lk, Lq)
        s[.!mask[:, :, 1, bi]] .= -Inf
        p = exp.(s .- maximum(s; dims=1)); p ./= sum(p; dims=1)
        out[:, :, hi, bi] = v[:, :, hi, bi] * p
    end
    out
end

const QUESTIONS = Dict(
    "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
    "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
    "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
    "one" => Dict("type" => "choice", "instructions" => "Only", "criteria" => ["x"]),
)

# Real checkpoints are large (~1.7 GB per Float32 copy, held by both LayaMLX and the Python
# reference). Select ONE repository per process with LAYAMLX_TEST_REPOS (default: none) and
# the dtypes with LAYAMLX_TEST_DTYPES (default "float32,float16"; models are freed between).
const TEST_REPOS = filter(!isempty, split(get(ENV, "LAYAMLX_TEST_REPOS", ""), ","))
const TEST_DTYPES = filter(!isempty, split(get(ENV, "LAYAMLX_TEST_DTYPES", "float32,float16"), ","))
length(TEST_REPOS) <= 1 || error("LAYAMLX_TEST_REPOS: test one real checkpoint per process")

function cached_snapshot(repo)
    root = joinpath(homedir(), ".cache", "huggingface", "hub", "models--" * replace(repo, "/" => "--"), "snapshots")
    isdir(root) || return nothing
    snaps = filter(d -> isfile(joinpath(d, "model.safetensors")), readdir(root; join=true))
    isempty(snaps) ? nothing : first(snaps)
end

"""Release checkpoints on both the Julia/mlx-c and the Python side."""
function free_models()
    GC.gc(true)
    MX.clear_cache()
    R.PythonCall.pyimport("gc").collect()
    R.PythonCall.pyimport("mlx.core").clear_cache()
    nothing
end

"""
Run the reference `trace` and our forward on `batch`; return per-stage relative errors
(max abs error over valid tokens / max |reference|). Masks must match exactly.
"""
function compare(ref, model, batch)
    tr = R.trace(ref, batch)
    logits, action, t = LayaMLX.forward(model, batch; trace=true)
    @test Set(keys(t)) == Set(keys(tr))
    @test logits == t["logits"] && action == t["action"]
    valid = Bool.(batch["attention_mask"])
    errs = Dict{String,Float64}()
    for (k, e) in tr
        @test size(t[k]) == size(e)
        size(t[k]) == size(e) || (errs[k] = Inf; continue)
        if startswith(k, "mask")
            @test t[k] == e
            continue
        end
        a, e = ndims(e) == 3 ? (t[k][:, valid], e[:, valid]) : (t[k], e)
        errs[k] = maxdiff(a, e) / maximum(abs, e)
    end
    tr, logits, action, errs
end

"""Stage summary for logging: embeddings, first/middle/last layer, encoder, head, outputs."""
function summarize(errs)
    n = count(k -> startswith(k, "layer_"), keys(errs))
    ks = unique(["embeddings", "layer_0", "layer_$(n ÷ 2)", "layer_$(n - 1)", "encoder", "typed",
                 sort!(filter(k -> startswith(k, "head_"), collect(keys(errs))))..., "logits", "action"])
    [k => round(errs[k]; sigdigits=3) for k in ks if haskey(errs, k)]
end

const REAL_QUESTIONS = Dict(
    "department" => Dict("type" => "choice", "instructions" => "Which team should handle this request?",
        "criteria" => Dict("billing" => "invoices, payments, refunds", "technical" => "bugs and outages",
                           "sales" => "new purchases")),
    "urgency" => Dict("type" => "score", "instructions" => "How urgent is this request?",
        "criteria" => ["not urgent", "soon", "critical"]),
    "refund" => Dict("type" => "noul", "instructions" => "Does the customer ask for money back?"),
)
const REAL_STATES = [
    "I was billed twice. Please refund the duplicate today.",
    "The app crashes every time I open the settings page after the latest update. "^8,  # sliding window
]
# fp32: same MLX kernels as Python, so errors are ~0; MLX's own GPU-vs-CPU difference is
# ~1e-5 relative. fp16: the half-precision rounding of the reference itself.
const RTOL = Dict("float32" => 3e-5, "float16" => 1e-2)

function real_checkpoint(repo, dir, dt)
    T = Dict("float32" => Float32, "float16" => Float16)[dt]
    ref = R.load(dir; dtype=dt, device="gpu")
    model = LayaMLX.load(dir; dtype=T, device=:gpu)
    for state in REAL_STATES
        items = R.prepare(ref, state, REAL_QUESTIONS)
        batch = R.collate(ref, items)
        tr, logits, action, errs = compare(ref, model, batch)
        @test all(e -> e <= RTOL[dt], values(errs))
        # selected answers: argmax over each question's valid options, and the act decision
        k = [length(it.markers) for it in items]
        @test [argmax(logits[1:k[j], j]) for j in eachindex(k)] == [argmax(tr["logits"][1:k[j], j]) for j in eachindex(k)]
        @test [argmax(action[:, j]) for j in eachindex(k)] == [argmax(tr["action"][:, j]) for j in eachindex(k)]
        @info "$repo $dt, L=$(size(batch["input_ids"], 1)), n=$(length(items)): relative errors" stages = join(["$k=$v" for (k, v) in summarize(errs)], " ")
            worst = maximum(values(errs))
    end
    ref = model = nothing
    free_models()
end

@testset "LayaMLX" begin

@testset "ops vs Julia ($dev)" for dev in (:gpu, :cpu)
    MX.with_device(dev) do
        a = randn(Float32, 5, 4, 3)
        x = MLXArray(a)
        @test shape(x) == (3, 4, 5) && size(x) == (5, 4, 3)
        @test Array(x) == a
        for T in (Bool, Int32, Float16, Float32)
            b = T === Bool ? rand(Bool, 3, 2) : T.(rand(-3:3, 3, 2))
            y = MLXArray(b)
            @test eltype(y) == T && Array(y) == b
        end
        bf = astype(x, MX.BFloat16)
        @test eltype(bf) == MX.BFloat16 && maxdiff(Array(bf), a) < 0.02 * maximum(abs, a)
        @test_throws MX.MLXError MX.matmul(x, x)
        # scoped: arrays created inside are freed on return, outer ones survive
        v, z = MX.scoped(() -> (z = x + 1; (Array(z), z)))
        @test v == a .+ 1 && Array(x) == a
        @test_throws MX.MLXError Array(z)

        @test Array(MX.transpose(x)) == permutedims(a, (3, 2, 1))
        @test Array(MX.transpose(x, 0, 2, 1)) == permutedims(a, (2, 1, 3))
        @test Array(MX.reshape(x, 3, 20)) == reshape(a, 20, 3)
        @test Array(MX.index(x, 1, 2)) == a[:, 3, :]
        @test Array(MX.slice(x, [0, 1], [3, 3])) == a[:, 2:3, :]
        @test Array.(MX.split(x, 2; axis=1)) == [a[:, 1:2, :], a[:, 3:4, :]]
        @test Array(MX.concatenate([x, x]; axis=-1)) == cat(a, a; dims=1)
        @test Array(MX.stack([x, x]; axis=0)) == cat(a, a; dims=4)
        @test Array(MX.sort(x; axis=-1)) == sort(a; dims=1)
        @test Array(MX.sum(x; axis=-1)) ≈ dropdims(sum(a; dims=1); dims=1)
        @test Array(MX.softmax(x; axis=-1)) ≈ exp.(a) ./ sum(exp.(a); dims=1)
        @test Array(MX.maximum(x, 0)) == max.(a, 0) == Array(MX.relu(x))
        @test Array(abs(x)) == abs.(a)
        @test Array(log(abs(x))) ≈ log.(abs.(a))
        @test Array(MX.gelu(x)) ≈ a .* (1 .+ erf.(a ./ sqrt(2f0))) ./ 2
        @test Array(MX.arange(5)) == Int32[0, 1, 2, 3, 4]
        m = MX.less_equal(x, 0)
        @test Array(m) == (a .<= 0) && Array(~m) == (a .> 0)
        @test Array(m & MX.less_equal(x, 1)) == (a .<= 0) .& (a .<= 1)
        @test Array(m | MX.greater(x, 1)) == (a .<= 0) .| (a .> 1)
        @test Array(MX.where(m, x, -1e4)) == ifelse.(a .<= 0, a, -1f4)
        @test Array(x * 2 - 1 / x) ≈ a .* 2 .- 1 ./ a
        @test eltype(MLXArray(Float16.(a)) + 1.5) == Float16          # weak scalars

        W = randn(Float32, 5, 7); bias = randn(Float32, 7)            # MLX weight (7, 5)
        @test Array(MX.matmul(x, MX.transpose(MLXArray(W)))) ≈ reshape(W' * reshape(a, 5, :), 7, 4, 3)
        @test Array(MX.addmm(MLXArray(bias), x, MX.transpose(MLXArray(W)))) ≈
              reshape(W' * reshape(a, 5, :) .+ bias, 7, 4, 3)
        E = randn(Float32, 6, 10); ids = Int32[3 0; 9 2]
        @test Array(MX.embedding(MLXArray(E), MLXArray(ids))) == reshape(E[:, vec(ids) .+ 1], 6, 2, 2)
        @test Array(MX.take(MLXArray(E), MLXArray(Int32[1, 4]), 0)) == E[:, [2, 5]]

        w, b = randn(Float32, 5), randn(Float32, 5)
        @test Array(MX.layer_norm(x, MLXArray(w), MLXArray(b), 1e-5)) ≈ jl_layernorm(a, w, b, 1f-5) atol = 1e-5
        @test Array(MX.layer_norm(x, MLXArray(w), nothing, 1e-5)) ≈ jl_layernorm(a, w, 0, 1f-5) atol = 1e-5

        q, k, v = (randn(Float32, 8, 6, 2, 3) for _ in 1:3)           # MLX (b=3, H=2, L=6, hd=8)
        @test Array(MX.rope(MLXArray(q), 8; base=10000.0)) ≈ jl_rope(q, 10000.0) atol = 1e-5
        mask = rand(Bool, 6, 6, 1, 3); mask[1, :, :, :] .= true
        @test Array(MX.sdpa(MLXArray(q), MLXArray(k), MLXArray(v); scale=8^-0.5, mask=MLXArray(mask))) ≈
              jl_sdpa(q, k, v, 8^-0.5f0, mask) atol = 1e-5
    end
end

@testset "tiny checkpoint vs reference trace ($dev)" for dev in (:gpu, :cpu)
    dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
    ref = R.load(dir; dtype="float32", device=string(dev))
    model = LayaMLX.load(dir; dtype=Float32, device=dev)
    for state in ("hello", join(fill("hello world", 30), " "))   # long state exercises sliding masks
        batch = R.collate(ref, R.prepare(ref, state, QUESTIONS))
        _, _, _, errs = compare(ref, model, batch)
        @test all(e -> e <= 1e-5, values(errs))
        @info "tiny ($dev), L=$(size(batch["input_ids"], 1)): relative errors" stages = join(["$k=$v" for (k, v) in summarize(errs)], " ") worst = maximum(values(errs))
    end
    # strict parameter check: a config asking for more head layers must be rejected
    bad = cp(dir, joinpath(mktempdir(), "bad"))
    cfgfile = joinpath(bad, "rl_agent_config.json")
    write(cfgfile, replace(read(cfgfile, String), r"\"head_layers\": *1" => "\"head_layers\": 2"))
    @test_throws ArgumentError LayaMLX.load(bad)
    ref = model = nothing
    free_models()
end

@testset "$repo ($dt)" for repo in TEST_REPOS, dt in TEST_DTYPES
    dir = cached_snapshot(repo)
    if dir === nothing
        @info "Skipping: $repo is not in the Hugging Face cache"
    else
        withenv(() -> real_checkpoint(repo, dir, dt), "HF_HUB_OFFLINE" => "1")
    end
end

end
