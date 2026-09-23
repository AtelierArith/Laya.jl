# Benchmark the mlx-c backend (LayaMLX, MLX GPU) end to end; one checkpoint per process.
#
#     julia --project=LayaMLX/bench LayaMLX/bench.jl --model aac6fef/laya-mlx --output out.json
#
# Same options, workloads, timing boundaries and JSON schema as benchmark/bench_julia.jl
# (backend "julia-mlxc-gpu"), so benchmark/compare.jl can summarize the result. Tokenization,
# prompt building (`Laya.prepare`), batching (`Laya.collate`) and calibration come from
# the pure-Julia Laya package; only the model forward runs on MLX through mlx-c.
# `forward` is one collated batch including host->device upload, evaluation and the copy of
# logits/action back to Julia arrays.

using Dates
using JSON
using Laya
using LayaMLX
using Statistics

const HERE = @__DIR__
const ROOT = joinpath(HERE, "..")
const EXAMPLES = joinpath(ROOT, "extern", "laya-mlx", "examples")

# ---------------------------------------------------------------------------- agent glue

"""A `Laya.Agent` without CPU weights (so `Laya.prepare` can be reused as is)."""
function tokenizer_agent(dir, model_id, ::Type{T}, batch_size) where {T}
    cfg = Dict{String,Any}(JSON.parsefile(joinpath(dir, "rl_agent_config.json")))
    enc = Laya.EncoderConfig(JSON.parsefile(joinpath(dir, "encoder", "config.json")))
    ln = Laya.LayerNorm(T[], nothing, 1.0f-5)
    lin = Laya.Linear(Matrix{T}(undef, 0, 0), nothing)
    stub = Laya.DecisionModel{T}(
        Laya.ModernBert{T}(enc, Matrix{T}(undef, 0, 0), ln, Laya.EncoderLayer{T}[], ln),
        Laya.HeadLayer[], Matrix{T}(undef, 0, 0), ln, lin, lin, lin, lin)
    traw = Float64.(get(cfg, "temperature", [1.0, 1.0, 1.0]))
    braw = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in get(cfg, "temperature_by_options", Dict()))
    Laya.Agent{T}(model_id, dir, cfg, enc, Laya.Tokenizer(joinpath(dir, "tokenizer")), stub, batch_size,
        traw, braw, Laya.clamp_temperature.(traw), Dict(k => Laya.clamp_temperature(v) for (k, v) in braw))
end

struct MLXAgent
    agent::Laya.Agent      # tokenizer, config and calibration
    model::LayaMLX.LayaModel # weights on the MLX device
end

function load_agent(model_id; dtype, batch_size)
    dir = Laya.resolve_model(model_id)
    MLXAgent(tokenizer_agent(dir, model_id, dtype, batch_size), LayaMLX.load(dir; dtype, device=:gpu))
end

"""`Laya.predict` with the model forward on MLX (post-processing copied from src/agent.jl)."""
function predict(a::MLXAgent, state, questions::AbstractDict)
    agent = a.agent
    items, internal = Laya.prepare(agent, state, questions)
    qids = collect(keys(questions))
    answers = JSON.Object{String,Any}()
    r4 = Laya.r4
    for start in 1:agent.batch_size:length(items)
        stop = min(start + agent.batch_size - 1, length(items))
        chunk = items[start:stop]
        logits, act = a.model(Laya.collate(chunk, agent.tok.pad_token_id))
        (all(isfinite, logits) && all(isfinite, act)) || throw(DomainError(logits, "Non-finite model outputs"))
        act = exp.(act .- maximum(act; dims=1))
        act ./= sum(act; dims=1)
        for (row, item) in enumerate(chunk)
            q = internal[start+row-1]
            k, qt = length(item.markers), item.qtype
            scale = get(agent.temperature_by_options, Laya.temp_bucket(qt, k), agent.temperature[qt+1])
            z = logits[1:k, row] ./ Float32(scale)
            p = exp.(z .- maximum(z))
            p ./= sum(p)
            answer = JSON.Object{String,Any}(
                "type" => q.t,
                "confidence" => r4(Laya.confidence_from_probs(p, k)),
                "action" => JSON.Object{String,Any}("act_probability" => r4(act[1, row])),
            )
            if q.t == "choice"
                labels = first.(q.crit)
                answer["choice"] = labels[argmax(p)]
                answer["probabilities"] = JSON.Object{String,Any}(l => r4(v) for (l, v) in zip(labels, p))
            elseif q.t == "score"
                answer["score"] = r4(sum((0:k-1) .* Float64.(p)))
                answer["legend"] = JSON.Object{String,Any}(string(i - 1) => v for (i, v) in enumerate(q.crit))
                answer["probabilities"] = JSON.Object{String,Any}(string(i - 1) => r4(v) for (i, v) in enumerate(p))
            else
                p1 = Float64(p[2])
                answer["noul"] = r4(p1)
                answer["confidence"] = r4(max(p1, 1.0 - p1))
            end
            answers[string(qids[start+row-1])] = answer
        end
    end
    JSON.Object{String,Any}(
        "model" => "laya-rl-agent",
        "answers" => answers,
        "usage" => JSON.Object{String,Any}("input_tokens" => sum(it -> length(it.ids), items), "output_tokens" => 0),
    )
end

# ---------------------------------------------------------------------------- benchmark
# (option parsing, workloads and measurement identical to benchmark/bench_julia.jl)

function parse_args(args)
    opts = Dict{String,String}("model" => "aac6fef/laya-mlx", "dtype" => "float32", "batch-size" => "64",
                               "warmup" => "2", "iterations" => "10", "workloads" => "", "output" => "")
    i = 1
    while i <= length(args)
        key = replace(args[i], r"^--" => "")
        haskey(opts, key) || error("Unknown option $(args[i]); expected one of $(join("--" .* keys(opts), ", "))")
        opts[key] = args[i+1]
        i += 2
    end
    isempty(opts["output"]) && error("--output is required")
    opts
end

function workload(spec, kind, count)
    state = JSON.parsefile(joinpath(EXAMPLES, "state.json"))
    definitions = collect(values(JSON.parsefile(joinpath(EXAMPLES, "questions.json"))))
    kind == "long" && (state["body"] = spec["long_body"]^spec["long_repeat"])
    questions = JSON.Object{String,Any}("q$(i)" => definitions[i%length(definitions)+1] for i in 0:count-1)
    state, questions
end

function measure(f, warmup, iterations)
    for _ in 1:warmup
        f()
    end
    samples = Float64[]
    for _ in 1:iterations
        start = time_ns()
        f()
        push!(samples, (time_ns() - start) / 1e6)
    end
    Dict("samples_ms" => samples, "p50_ms" => median(samples), "p95_ms" => quantile(samples, 0.95),
         "mean_ms" => mean(samples), "min_ms" => minimum(samples), "max_ms" => maximum(samples))
end

function main(args)
    opts = parse_args(args)
    spec = JSON.parsefile(joinpath(ROOT, "benchmark", "workload.json"))
    workloads = isempty(opts["workloads"]) ? String.(spec["default_workloads"]) : split(opts["workloads"], ",")
    dtype = Dict("float32" => Float32, "float16" => Float16)[opts["dtype"]]
    batch_size = parse(Int, opts["batch-size"])
    warmup, iterations = parse(Int, opts["warmup"]), parse(Int, opts["iterations"])

    load_seconds = @elapsed agent = load_agent(opts["model"]; dtype, batch_size)
    report = Dict{String,Any}(
        "backend" => "julia-mlxc-gpu",
        "created_at" => string(now(UTC)),
        "environment" => Dict(
            "julia" => string(VERSION),
            "threads" => Threads.nthreads(),
            "mlx" => LayaMLX.MX.version(),
            "libmlxc" => basename(LayaMLX.LibMLX.libmlxc),
            "cpu" => Sys.cpu_info()[1].model,
            "machine" => string(Sys.ARCH),
        ),
        "model" => opts["model"],
        "dtype" => opts["dtype"],
        "batch_size" => batch_size,
        "warmup" => warmup,
        "iterations" => iterations,
        "load_seconds" => load_seconds,
        "timing" => "wall clock; model loading excluded; forward includes upload, MLX eval and copy back",
        "results" => Any[],
    )
    for name in workloads
        kind, count = split(name, ":")
        count = parse(Int, count)
        state, questions = workload(spec, kind, count)
        result = predict(agent, state, questions)
        items, _ = Laya.prepare(agent.agent, state, questions)
        batch = Laya.collate(items[1:min(end, batch_size)], agent.agent.tok.pad_token_id)
        forward = measure(() -> agent.model(batch), warmup, iterations)
        prep = measure(() -> Laya.prepare(agent.agent, state, questions), warmup, iterations)
        e2e = measure(() -> predict(agent, state, questions), warmup, iterations)
        e2e["questions_per_second"] = count * 1000 / e2e["mean_ms"]
        answers = Dict(k => get(v, "choice", get(v, "score", get(v, "noul", nothing))) for (k, v) in result["answers"])
        push!(report["results"], Dict(
            "workload" => name, "questions" => count, "sequence_length" => size(batch["input_ids"], 1),
            "input_tokens" => result["usage"]["input_tokens"], "prepare" => prep, "forward" => forward, "e2e" => e2e,
            "answers" => answers,
        ))
        println(rpad(name, 10), " e2e p50 ", lpad(round(e2e["p50_ms"]; digits=2), 9), " ms  forward p50 ",
                lpad(round(forward["p50_ms"]; digits=2), 9), " ms")
        flush(stdout)
    end
    report["max_rss_bytes"] = Sys.maxrss()
    report["mlx_peak_memory_bytes"] = LayaMLX.MX.peak_memory()
    mkpath(dirname(abspath(opts["output"])))
    open(io -> JSON.json(io, report; pretty=true), opts["output"], "w")
end

main(ARGS)
