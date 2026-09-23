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

# Test groups to run: LAYA_TEST_GROUPS=math,model,tokenizer,agent (default: all).
const TEST_GROUPS = split(get(ENV, "LAYA_TEST_GROUPS", "math,model,tokenizer,agent"), ",")

"""Release checkpoints held by finished tests on both the Julia and the Python side."""
function free_models()
    GC.gc(true)
    R.PythonCall.pyimport("gc").collect()
    mx = R.PythonCall.pyimport("mlx.core")
    mx.clear_cache()
    nothing
end

@testset "Laya" verbose=true begin
    for group in TEST_GROUPS
        include("test_$(group).jl")
    end
end
