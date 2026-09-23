# Benchmark the mlx-c backend (LayaMLX, MLX GPU) end to end; one checkpoint per process.
#
#     julia --project=LayaMLX/bench LayaMLX/bench.jl --model aac6fef/laya-mlx --output out.json
#
# Same options, workloads, timing boundaries and JSON schema as benchmark/bench_julia.jl
# (backend "julia-mlxc-gpu"), so benchmark/compare.jl can summarize the result. The agent is
# `Laya.load(...; backend=MLXBackend(:gpu))`: tokenization, prompts and calibration run in Laya,
# only the model forward runs on MLX through mlx-c.
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

    load_seconds = @elapsed agent = Laya.load(opts["model"]; dtype, batch_size, backend=MLXBackend(:gpu))
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
        result = Laya.predict(agent, state, questions)
        items, _ = Laya.prepare(agent, state, questions)
        batch = Laya.collate(items[1:min(end, batch_size)], agent.tok.pad_token_id)
        forward = measure(() -> agent.model(batch), warmup, iterations)
        prep = measure(() -> Laya.prepare(agent, state, questions), warmup, iterations)
        e2e = measure(() -> Laya.predict(agent, state, questions), warmup, iterations)
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
