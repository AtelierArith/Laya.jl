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
        @test size(actual[k]) == size(v)
        a, e = ndims(v) == 3 ? (actual[k][:, valid], v[:, valid]) : (actual[k], v)
        errs[k] = maxerr(a, e) / maximum(abs, e)
    end
    # Masks: reference (L_k, 1, 1, B) / (L_k, L_q, 1, B); ours (L_k, L_q, B).
    @test all(actual["mask_full"] .== dropdims(expected["mask_full"]; dims=3))
    @test actual["mask_sliding"] == dropdims(expected["mask_sliding"]; dims=3)
    errs
end

@testset "tiny checkpoint, float32" begin
    dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
    ref = R.load(dir; dtype="float32", device="gpu")
    model, cfg, _ = load_model(dir)
    @test cfg.layer_types == [:full_attention, :sliding_attention, :sliding_attention]
    # A long, partly padded batch exercises the sliding window and padded queries.
    states = ["hello", join(fill("hello", 100), " ")]
    for state in states
        batch = R.collate(ref, R.prepare(ref, state, QUESTIONS))
        errs = compare_trace(model, ref, batch)
        @info "tiny: $(repr(first(state, 20)))" errs
        @test all(e -> e < 1e-5, values(errs))
    end
end

@testset "$repo, float32" for repo in TEST_REPOS
    dir = cached_snapshot(repo)
    if dir === nothing
        @info "Skipping: $repo is not in the Hugging Face cache"
    else
        ref = R.load(dir; dtype="float32", device="gpu")
        model, _, _ = load_model(dir)
        questions = Dict(
            "department" => Dict("type" => "choice", "instructions" => "Which team should handle this request?",
                "criteria" => Dict("billing" => "invoices, payments, refunds", "technical" => "bugs and outages", "sales" => "new purchases")),
            "urgency" => Dict("type" => "score", "instructions" => "How urgent is this request?",
                "criteria" => ["not urgent", "soon", "critical"]),
            "refund" => Dict("type" => "noul", "instructions" => "Does the customer ask for money back?"),
        )
        state = "I was billed twice. Please refund the duplicate today."
        batch = R.collate(ref, R.prepare(ref, state, questions))
        errs = compare_trace(model, ref, batch)
        @info "$repo float32" errs
        # MLX's own GPU-vs-CPU float32 difference is ~1.1e-5 relative at layer 20 of the
        # English checkpoint; the Julia port stays below it at every stage.
        @test all(e -> e < 3e-5, values(errs))
        @test errs["logits"] < 1e-6
    end
    free_models()
end
