# Backend dispatch (`load(...; backend)`) and the Metal.jl extension. The Metal tests need an
# Apple GPU: enable them with LAYA_TEST_METAL=1.

"""Selected answers: the choice, the argmax score level and the side of 0.5 for noul."""
selected(r) = Dict(k => a["type"] == "choice" ? a["choice"] :
                        a["type"] == "score" ? argmax(p -> a["probabilities"][p], collect(keys(a["probabilities"]))) :
                        a["noul"] > 0.5 for (k, a) in r["answers"])

@testset "backend dispatch" begin
    dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
    @test Laya.load(dir; backend=CPUBackend()).model isa DecisionModel
    @test_throws ArgumentError Laya.load(dir; backend=:nonexistent)
    if Base.get_extension(Laya, :LayaAppleAccelerateExt) === nothing
        @test_throws ArgumentError Laya.load(dir; backend=AccelerateBackend())
    else
        agent = Laya.load(dir; backend=AccelerateBackend())
        @test agent.model isa DecisionModel
        @test Laya.uses_accelerate()
    end
end

if get(ENV, "LAYA_TEST_METAL", "0") == "1"
    using Metal, Random
    @testset "Metal: tiny checkpoint, $T" for (T, tol) in ((Float32, 1e-5), (Float16, 1e-2))
        dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
        ref = R.load(dir; dtype=T == Float32 ? "float32" : "float16", device="gpu")
        agent = Laya.load(dir; dtype=T, backend=MetalBackend())
        @test agent.model.type_emb isa MtlArray{T}
        for state in ["hello", join(fill("hello", 100), " ")]   # the long one is padded and windowed
            batch = R.collate(ref, R.prepare(ref, state, QUESTIONS))
            errs = compare_trace(agent.model, ref, batch)
            @info "Metal tiny $T: $(repr(first(state, 20)))" errs
            @test all(e -> e < tol, values(errs))
        end
        @test selected(predict(agent, "hello", QUESTIONS)) == selected(R.predict(ref, "hello", QUESTIONS))
    end

    @testset "Metal: attention vs CPU, $T" for T in (Float32, Float16)
        tol = T == Float32 ? 1e-5 : 2e-2
        for (H, L, B) in ((16, 93, 1), (16, 93, 10), (16, 45, 2), (16, 200, 1), (16, 7, 1), (4, 93, 2))   # H = 4: head_dim 32, the unfused path
            d = 64H ÷ (H == 4 ? 2 : 1)
            qkv = randn(T, 3d, L, B)
            valid = trues(L, B)
            B > 1 && (valid[L÷2+1:end, 2] .= false)                      # padded sequence
            masks = Laya.attention_masks(valid, 128)
            for (base, mask) in ((10000.0, masks.full), (10000.0, masks.sliding), (nothing, reshape(valid, L, 1, B)), (10000.0, nothing))
                B > 1 && mask === nothing && continue
                ref = Laya.qkv_attention(qkv, H, base, mask, T(d ÷ H)^T(-0.5))
                out = Laya.qkv_attention(MtlArray(qkv), H, base, mask === nothing ? nothing : MtlArray(mask), T(d ÷ H)^T(-0.5))
                err = maxerr(Array(out)[:, valid], ref[:, valid]) / maximum(abs, Float32.(ref[:, valid]))
                @test err < tol
            end
        end
    end

    # The fused kernel reads `valid` and `window` of an AttentionMask to skip key tiles outside
    # the local window; padded queries (left, right, holes) must still see every valid key.
    @testset "Metal: attention with AttentionMask, $T" for T in (Float32, Float16)
        tol = T == Float32 ? 1e-5 : 2e-2
        H = 16
        rng = Random.Xoshiro(1)     # unseeded, the Float16 error reached 0.016 at L=512
        for (L, B) in ((93, 4), (200, 4), (300, 4), (512, 2)), window in (128, 16)
            qkv = randn(rng, T, 3 * 64H, L, B)
            valid = trues(L, B)
            valid[L÷2+1:end, 2] .= false                                  # right padding
            B > 2 && (valid[[3, 10, L], 3] .= false)                      # holes
            B > 3 && (valid[1:40, 4] .= false)                            # left padding
            dense = Laya.attention_masks(valid, window)
            for (mask, w) in ((dense.full, nothing), (dense.sliding, window ÷ 2))
                ref = Laya.qkv_attention(qkv, H, 10000.0, Laya.AttentionMask(mask, valid, w), T(0.125))
                out = Laya.qkv_attention(MtlArray(qkv), H, 10000.0, Laya.AttentionMask(MtlArray(mask), MtlArray(valid), w), T(0.125))
                err = maxerr(Array(out)[:, valid], ref[:, valid]) / maximum(abs, Float32.(ref[:, valid]))
                @test err < tol
            end
        end
    end

    @testset "Metal: attention ignores garbage in pooled buffers" begin
        E = Base.get_extension(Laya, :LayaMetalExt)
        for H in (16, 4)                                # the fused and the unfused path
            hd, L, B = 64, 45, 1
            d = H == 4 ? 128 : 64H
            qkv = MtlArray(randn(Float32, 3d, L, B))
            # Poison the pool: every buffer size the attention can take out of it holds NaN
            # (a matmul must not read its uninitialized output operand).
            for dims in ((d, L, B), (d ÷ H, L, H * B), (L, L, H * B))
                for _ in 1:3
                    x = E.pooled(Float32, dims)
                    fill!(x, NaN32)
                    Laya.release!(x)
                end
            end
            y = Laya.qkv_attention(qkv, H, 10000.0, nothing, 0.125)
            @test !any(isnan, Array(y))
            @test Array(y) ≈ Laya.qkv_attention(Array(qkv), H, 10000.0, nothing, 0.125) atol=1e-4
        end
    end

    @testset "Metal: warm forwards reuse device buffers" begin
        E = Base.get_extension(Laya, :LayaMetalExt)
        dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
        ref = R.load(dir; dtype="float32", device="gpu")
        agent = Laya.load(dir; dtype=Float32, backend=MetalBackend())
        batch = R.collate(ref, R.prepare(ref, join(fill("hello", 100), " "), QUESTIONS))
        expected = agent.model(batch)
        agent.model(batch)
        GC.gc(true)                           # dropped intermediates return to the pool
        misses = E.POOL_MISSES[]
        @test all(agent.model(batch) .≈ expected)
        @test E.POOL_MISSES[] == misses       # every intermediate came from the pool
    end

    @testset "Metal: $repo, $T" for repo in TEST_REPOS, T in (Float32, Float16)
        dir = cached_snapshot(repo)
        if dir === nothing
            @info "Skipping: $repo is not in the Hugging Face cache"
        else
            ref = R.load(dir; dtype=T == Float32 ? "float32" : "float16", device="gpu")
            agent = Laya.load(dir; dtype=T, backend=MetalBackend())
            state = "I was billed twice. Please refund the duplicate today."
            questions = Dict(
                "department" => Dict("type" => "choice", "instructions" => "Which team should handle this request?",
                    "criteria" => Dict("billing" => "invoices, payments, refunds", "technical" => "bugs and outages", "sales" => "new purchases")),
                "urgency" => Dict("type" => "score", "instructions" => "How urgent is this request?",
                    "criteria" => ["not urgent", "soon", "critical"]),
                "refund" => Dict("type" => "noul", "instructions" => "Does the customer ask for money back?"),
            )
            errs = compare_trace(agent.model, ref, R.collate(ref, R.prepare(ref, state, questions)))
            @info "Metal $repo $T" errs
            @test errs["logits"] < (T == Float32 ? 1e-4 : 3e-2)
            @test selected(predict(agent, state, questions)) == selected(R.predict(ref, state, questions))
            agent = ref = nothing
            free_models()
        end
    end
end
