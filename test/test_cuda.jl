# The CUDA.jl extension against the CPU backend (no Python needed). Runs only when CUDA is
# functional (an NVIDIA GPU and driver); select the GPU with CUDA_VISIBLE_DEVICES. The real
# checkpoints in LAYA_TEST_REPOS are compared when they are cached (Hugging Face cache or
# Laya's scratch space).
using CUDA, Random

if !CUDA.functional()
    @info "Skipping the CUDA tests: CUDA is not functional on this machine"
else
    CUDA.allowscalar(false)
    cuda_maxerr(a, b) = maximum(abs.(Float32.(a) .- Float32.(b)))
    cuda_selected(r) = Dict(k => a["type"] == "choice" ? a["choice"] :
                                 a["type"] == "score" ? argmax(p -> a["probabilities"][p], collect(keys(a["probabilities"]))) :
                                 a["noul"] > 0.5 for (k, a) in r["answers"])
    const CUDA_QUESTIONS = Dict(
        "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
        "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
        "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
    )
    CE = Base.get_extension(Laya, :LayaCUDAExt)

    @testset "CUDA: tiny checkpoint, $T" for (T, tol) in ((Float32, 1e-5), (Float16, 2e-2))
        # head_dim 64, as the shipped checkpoints, so the fused attention kernel runs (random
        # weights from a seeded RNG: unseeded, near-tied options could flip in Float16)
        Random.seed!(1)
        dir = Laya.write_tiny_checkpoint(joinpath(mktempdir(), "cuda"); hidden_size=128, num_attention_heads=2,
            intermediate_size=96, num_hidden_layers=3, head_layers=2)
        cpu = Laya.load(dir; dtype=Float32)
        agent = Laya.load(dir; dtype=T, backend=CUDABackend())
        @test agent.model.type_emb isa CuArray{T}
        @test_throws ArgumentError Laya.load(dir; dtype=Float64, backend=CUDABackend())
        for state in ["hello", join(fill("hello", 150), " ")]   # the long one is padded and windowed
            items, _ = Laya.prepare(cpu, state, CUDA_QUESTIONS)
            batch = Laya.collate(items, cpu.tok.pad_token_id)
            expected = Dict{String,Any}()
            cpu.model(batch; trace=expected)
            actual = Dict{String,Any}()
            agent.model(batch; trace=actual)
            valid = Bool.(batch["attention_mask"])
            errs = Dict{String,Float32}()
            for (k, v) in expected
                startswith(k, "mask") && continue
                a = Laya.to_host(actual[k])
                @test size(a) == size(v)
                a, e = ndims(v) == 3 ? (a[:, valid], v[:, valid]) : (a, v)
                errs[k] = cuda_maxerr(a, e) / maximum(abs, e)
            end
            @test all(e -> e < tol, values(errs))
        end
        # Float16 answers are compared on the real checkpoint below (the tiny random model's
        # options are nearly tied)
        T == Float32 &&
            @test cuda_selected(predict(agent, "hello", CUDA_QUESTIONS)) == cuda_selected(predict(cpu, "hello", CUDA_QUESTIONS))
    end

    @testset "CUDA: attention vs CPU, $T" for T in (Float32, Float16)
        tol = T == Float32 ? 1e-5 : 2e-2
        rng = Random.Xoshiro(1)
        for (H, L, B) in ((16, 93, 1), (16, 93, 10), (16, 200, 4), (16, 512, 2), (16, 7, 1), (4, 93, 2))  # H = 4: head_dim 32, unfused
            d = H == 4 ? 128 : 64H
            qkv = randn(rng, T, 3d, L, B)
            valid = trues(L, B)
            B > 1 && (valid[L÷2+1:end, 2] .= false)                      # right padding
            B > 2 && (valid[[3, 10, L], 3] .= false)                      # holes
            B > 3 && (valid[1:40, 4] .= false)                            # left padding
            for window in (128, 16)
                dense = Laya.attention_masks(valid, window)
                for (mask, w, base) in ((dense.full, nothing, 10000.0), (dense.sliding, window ÷ 2, 10000.0),
                                        (reshape(valid, L, 1, B), nothing, nothing))
                    scale = T(d ÷ H)^T(-0.5)
                    ref = Laya.qkv_attention(qkv, H, base, Laya.AttentionMask(mask, valid, w), scale)
                    # AttentionMask: fused kernel (head_dim 64); plain dense mask: unfused path
                    for am in (Laya.AttentionMask(CuArray(mask), CuArray(valid), w), CuArray(mask))
                        out = Laya.qkv_attention(CuArray(qkv), H, base, am, scale)
                        err = cuda_maxerr(Array(out)[:, valid], ref[:, valid]) / maximum(abs, Float32.(ref[:, valid]))
                        @test err < tol
                    end
                end
            end
        end
    end

    @testset "CUDA: attention ignores garbage in pooled buffers" begin
        for H in (16, 4)                                # the fused and the unfused path
            L, B = 45, 1
            d = H == 4 ? 128 : 64H
            qkv = CuArray(randn(Float32, 3d, L, B))
            for dims in ((d, L, B), (d ÷ H, L, H * B), (L, L, H * B)), _ in 1:3
                x = CE.pooled(Float32, dims)
                fill!(x, NaN32)
                Laya.release!(x)
            end
            valid = trues(L, B)
            am = Laya.AttentionMask(CuArray(Laya.attention_masks(valid, 128).full), CuArray(valid), nothing)
            y = Array(Laya.qkv_attention(qkv, H, 10000.0, am, 0.125))
            @test !any(isnan, y)
            @test y ≈ Laya.qkv_attention(Array(qkv), H, 10000.0, nothing, 0.125) atol = 1e-4
        end
    end

    @testset "CUDA: warm forwards reuse device buffers and are deterministic" begin
        dir = Laya.write_tiny_checkpoint(joinpath(mktempdir(), "cuda"); hidden_size=128, num_attention_heads=2,
            num_hidden_layers=3, head_layers=2)
        agent = Laya.load(dir; dtype=Float32, backend=CUDABackend())
        items, _ = Laya.prepare(agent, join(fill("hello", 150), " "), CUDA_QUESTIONS)
        batch = Laya.collate(items, agent.tok.pad_token_id)
        expected = agent.model(batch)
        agent.model(batch)
        GC.gc(true)
        misses = CE.POOL_MISSES[]
        @test all(_ -> agent.model(batch) == expected, 1:50)   # bitwise
        @test CE.POOL_MISSES[] == misses                      # every intermediate came from the pool
    end

    @testset "CUDA: $repo, $T" for repo in TEST_REPOS, T in (Float32, Float16)
        dir = try
            withenv("HF_HUB_OFFLINE" => "1") do
                Laya.resolve_model(repo)
            end
        catch
            nothing
        end
        if dir === nothing
            @info "Skipping: $repo is not cached"
        else
            cpu = Laya.load(dir; dtype=Float32)
            agent = Laya.load(dir; dtype=T, backend=CUDABackend())
            state = "I was billed twice. Please refund the duplicate today."
            questions = Dict(
                "department" => Dict("type" => "choice", "instructions" => "Which team should handle this request?",
                    "criteria" => Dict("billing" => "invoices, payments, refunds", "technical" => "bugs and outages", "sales" => "new purchases")),
                "urgency" => Dict("type" => "score", "instructions" => "How urgent is this request?",
                    "criteria" => ["not urgent", "soon", "critical"]),
                "refund" => Dict("type" => "noul", "instructions" => "Does the customer ask for money back?"),
            )
            items, _ = Laya.prepare(cpu, state, questions)
            batch = Laya.collate(items, cpu.tok.pad_token_id)
            lc, ac = cpu.model(batch)
            lg, ag = agent.model(batch)
            mm = Bool.(batch["marker_mask"])
            err = cuda_maxerr(lg[mm], lc[mm]) / maximum(abs, lc[mm])
            @info "CUDA $repo $T: logits relative error $err"
            @test err < (T == Float32 ? 1e-5 : 3e-2)
            @test cuda_selected(predict(agent, state, questions)) == cuda_selected(predict(cpu, state, questions))
            agent = cpu = nothing
            GC.gc(true)
        end
    end
end
