# End-to-end checks without Python or downloads: the tiny random checkpoint that the
# precompile workload writes (byte-level and Metaspace BPE tokenizers), run through the
# public API on the CPU.

@testset "smoke: tiny checkpoint, $kind" for kind in (:bytelevel, :metaspace)
    dir = Laya.write_tiny_checkpoint(joinpath(mktempdir(), string(kind)); metaspace=kind === :metaspace)
    @test Laya.resolve_model(dir) == dir
    agent = Laya.load(dir; batch_size=2)
    @test agent.model isa DecisionModel{Float32}
    questions = Dict(
        "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
        "team" => Dict("type" => "choice", "instructions" => "Route", "criteria" => Dict("x" => "first", "y" => "second")),
        "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
        "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
    )
    for state in ("hello world ünïcode", Dict("body" => "hello", "items" => [1, 2.5, nothing, true]))
        r = predict(agent, state, questions)
        @test Set(keys(r["answers"])) == Set(keys(questions))
        @test r["usage"]["output_tokens"] == 0 && r["usage"]["input_tokens"] > 0
        a = r["answers"]
        @test a["topic"]["choice"] in ("a", "b", "c")
        @test sum(values(a["topic"]["probabilities"])) ≈ 1 atol = 1e-3
        @test a["team"]["choice"] in ("x", "y")
        @test 0 <= a["level"]["score"] <= 1
        @test 0 <= a["yes"]["noul"] <= 1
        @test all(0 <= q["action"]["act_probability"] <= 1 for q in values(a))
    end
    # Batching must not change the answers.
    r1 = predict(Laya.load(dir; batch_size=1), "hello", questions)
    r4 = predict(Laya.load(dir; batch_size=4), "hello", questions)
    @test r1["answers"] == r4["answers"]
end

@testset "smoke: errors" begin
    @test_throws ArgumentError Laya.load(joinpath(mktempdir(), "missing"))
    @test_throws ArgumentError Laya.load("./does/not/exist")
    withenv("HF_HUB_OFFLINE" => "1", "HF_HUB_CACHE" => mktempdir()) do
        @test_throws ArgumentError Laya.resolve_model("nobody/not-a-repo")
    end
    dir = Laya.write_tiny_checkpoint(joinpath(mktempdir(), "ckpt"))
    @test_throws ArgumentError Laya.load(dir; backend=:nonexistent)
    @test_throws ArgumentError predict(Laya.load(dir), "x", Dict("q" => Dict("type" => "bogus", "instructions" => "?")))
end

@testset "smoke: Hub download filter" begin
    listing = ["README.md", "assets/logo.png", "model.safetensors", "rl_agent_config.json", "encoder/config.json",
               "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "multilingual/model.safetensors",
               "multilingual/rl_agent_config.json", "multilingual/encoder/config.json", "multilingual/tokenizer/tokenizer.json"]
    @test filter(f -> Laya.checkpoint_file(f, Laya.subfolder_prefix(nothing)), listing) ==
          ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    @test filter(f -> Laya.checkpoint_file(f, Laya.subfolder_prefix("multilingual")), listing) ==
          ["multilingual/model.safetensors", "multilingual/rl_agent_config.json", "multilingual/encoder/config.json",
           "multilingual/tokenizer/tokenizer.json"]
    @test Laya.subfolder_prefix("multilingual/") == "multilingual/"
end
