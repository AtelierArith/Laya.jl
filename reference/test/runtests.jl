using LayaMLXReference
using LayaMLXReference: ReferenceAgent
using Test

const R = LayaMLXReference

const QUESTIONS = Dict(
    "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
    "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
    "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
)

@testset "tiny checkpoint" begin
    dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
    agent = R.load(dir; dtype="float32", device="cpu")
    @test agent isa ReferenceAgent

    sp = R.special_tokens(agent)
    @test sp["cls_token"] == (token="[CLS]", id=2)
    @test sp["mask_token"].id == 4
    @test R.tokenize(agent, "hello world") == [5, 1]

    items = R.prepare(agent, "hello", QUESTIONS)
    @test length(items) == 3
    @test all(it.ids[1] == 2 && it.ids[end] == 3 for it in items)
    @test all(all(it.ids[m+1] == 4 for m in it.markers) for it in items)  # 0-based markers hit [MASK]

    batch = R.collate(agent, items)
    n, L = length(items), maximum(length(it.ids) for it in items)
    @test size(batch["input_ids"]) == (L, n)
    @test batch["attention_mask"] isa Matrix{Bool}
    @test [batch["input_ids"][1:length(it.ids), j] for (j, it) in enumerate(items)] == [it.ids for it in items]

    tr = R.trace(agent, batch)
    @test size(tr["embeddings"]) == (64, L, n)
    @test size(tr["logits"]) == (3, n)          # max(2, most markers) × batch
    @test size(tr["action"]) == (2, n)
    @test haskey(tr, "layer_2") && haskey(tr, "head_0")
    @test all(isfinite, tr["encoder"])

    result = R.predict(agent, "hello", QUESTIONS)
    @test result["usage"]["input_tokens"] == sum(length(it.ids) for it in items)
    @test Set(keys(result["answers"])) == Set(keys(QUESTIONS))
end

# Real checkpoint: uses the local Hugging Face cache only; skipped when absent.
@testset "aac6fef/laya-mlx" begin
    withenv("HF_HUB_OFFLINE" => "1") do
        agent = try
            R.load("aac6fef/laya-mlx"; dtype="float32")
        catch err
            @info "Skipping real-checkpoint test" exception = err
            nothing
        end
        agent === nothing && return
        q = Dict("department" => Dict(
            "type" => "choice",
            "instructions" => "Who should handle this?",
            "criteria" => ["billing", "technical", "sales"],
        ))
        state = "I was billed twice. Please refund the duplicate."
        result = R.predict(agent, state, q)
        @test result["answers"]["department"]["choice"] == "billing"
        tr = R.trace(agent, R.collate(agent, R.prepare(agent, state, q)))
        @test size(tr["logits"], 1) == 3
    end
end
