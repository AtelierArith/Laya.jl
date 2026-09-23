using Laya
using LayaMLXReference
using Test

# LAYA_TEST_BLAS=accelerate runs the suite with BLAS forwarded to Apple's Accelerate
# (Laya's optional fast path on Apple silicon); the default is the bundled OpenBLAS.
if get(ENV, "LAYA_TEST_BLAS", "openblas") == "accelerate"
    using AppleAccelerate
end
using LinearAlgebra: BLAS
@info "BLAS: $(BLAS.get_config())"

const R = LayaMLXReference

maxerr(a, b) = maximum(abs.(Float32.(a) .- Float32.(b)))

"""Snapshot directory of `repo` in the local Hugging Face cache, or `nothing`."""
function cached_snapshot(repo)
    root = joinpath(homedir(), ".cache", "huggingface", "hub", "models--" * replace(repo, "/" => "--"), "snapshots")
    isdir(root) || return nothing
    snaps = readdir(root; join=true)
    isempty(snaps) ? nothing : first(snaps)
end

# Real checkpoints are large (~1.7 GB per Float32 copy, held by both Julia and the Python
# reference). Select them with LAYA_TEST_REPOS (comma-separated, "" for none) and prefer
# one repository per process on memory-constrained machines.
const TEST_REPOS = filter(!isempty, split(get(ENV, "LAYA_TEST_REPOS", "aac6fef/laya-mlx"), ","))

# Test groups to run: LAYA_TEST_GROUPS=aqua,math,model,tokenizer,agent,backends (default: all).
# LAYA_TEST_METAL=1 adds the Metal.jl backend to "backends" (needs an Apple GPU).
const TEST_GROUPS = split(get(ENV, "LAYA_TEST_GROUPS", "aqua,math,model,tokenizer,agent,backends"), ",")

"""Release checkpoints held by finished tests on both the Julia and the Python side."""
function free_models()
    GC.gc(true)
    R.PythonCall.pyimport("gc").collect()
    mx = R.PythonCall.pyimport("mlx.core")
    mx.clear_cache()
    nothing
end

const QUESTIONS = Dict(
    "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
    "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
    "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
)

"""
Compare every traced activation; returns Dict(key => relative error), i.e. the max abs
error over valid (unpadded) tokens divided by the reference's max magnitude. ModernBERT's
residual stream reaches ~2.5e4, so absolute tolerances are meaningless deep in the encoder.
"""
function compare_trace(model, ref_agent, batch)
    expected = R.trace(ref_agent, batch)
    actual = Dict{String,Any}()
    model(batch; trace=actual)
    errs = Dict{String,Float32}()
    valid = Bool.(batch["attention_mask"])
    for (k, v) in expected
        startswith(k, "mask") && continue
        @test haskey(actual, k)
        actual[k] = Laya.to_host(actual[k])     # device backends trace device arrays
        @test size(actual[k]) == size(v)
        a, e = ndims(v) == 3 ? (actual[k][:, valid], v[:, valid]) : (actual[k], v)
        errs[k] = maxerr(a, e) / maximum(abs, e)
    end
    # Masks: reference (L_k, 1, 1, B) / (L_k, L_q, 1, B); ours (L_k, L_q, B).
    @test all(Laya.to_host(actual["mask_full"]) .== dropdims(expected["mask_full"]; dims=3))
    @test Laya.to_host(actual["mask_sliding"]) == dropdims(expected["mask_sliding"]; dims=3)
    errs
end

@testset "Laya" verbose=true begin
    for group in TEST_GROUPS
        include("test_$(group).jl")
    end
end
