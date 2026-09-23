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
    using Metal
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
