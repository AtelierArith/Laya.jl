# Benchmark the pure-Julia Laya runtime (one checkpoint per process).
#
#     julia --project=benchmark benchmark/bench_julia.jl --model aac6fef/laya-mlx --output results/x.json
#
# Same workloads and timing boundaries as bench_python.py: wall clock, model loading
# excluded. `prepare` is tokenization + prompt building; `forward` is one collated batch.

using Dates
using JSON
using Laya
using Laya: prepare, collate
using LinearAlgebra
using Statistics

# `--blas accelerate` (default on macOS) forwards BLAS to Apple's Accelerate framework.
if Sys.isapple() && !("--blas" in ARGS && ARGS[findfirst(==("--blas"), ARGS)+1] == "openblas")
    using AppleAccelerate
end

const HERE = @__DIR__
const EXAMPLES = joinpath(HERE, "..", "extern", "laya-mlx", "examples")

function parse_args(args)
    opts = Dict{String,String}("model" => "aac6fef/laya-mlx", "dtype" => "float32", "batch-size" => "64",
                               "warmup" => "2", "iterations" => "10", "workloads" => "", "output" => "",
                               "blas" => Sys.isapple() ? "accelerate" : "openblas")
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
    spec = JSON.parsefile(joinpath(HERE, "workload.json"))
    workloads = isempty(opts["workloads"]) ? String.(spec["default_workloads"]) : split(opts["workloads"], ",")
    dtype = Dict("float32" => Float32, "float16" => Float16)[opts["dtype"]]
    batch_size = parse(Int, opts["batch-size"])
    warmup, iterations = parse(Int, opts["warmup"]), parse(Int, opts["iterations"])

    load_seconds = @elapsed agent = Laya.load(opts["model"]; dtype, batch_size)
    report = Dict{String,Any}(
        "backend" => "julia-cpu-" * opts["blas"],
        "created_at" => string(now(UTC)),
        "environment" => Dict(
            "julia" => string(VERSION),
            "threads" => Threads.nthreads(),
            "blas" => string(BLAS.get_config()),
            "blas_threads" => BLAS.get_num_threads(),
            "cpu" => Sys.cpu_info()[1].model,
            "machine" => string(Sys.ARCH),
        ),
        "model" => opts["model"],
        "dtype" => opts["dtype"],
        "batch_size" => batch_size,
        "warmup" => warmup,
        "iterations" => iterations,
        "load_seconds" => load_seconds,
        "timing" => "wall clock; model loading excluded",
        "results" => Any[],
    )
    for name in workloads
        kind, count = split(name, ":")
        count = parse(Int, count)
        state, questions = workload(spec, kind, count)
        result = predict(agent, state, questions)
        items, _ = prepare(agent, state, questions)
        batch = collate(items[1:min(end, batch_size)], agent.tok.pad_token_id)
        forward = measure(() -> agent.model(batch), warmup, iterations)
        prep = measure(() -> prepare(agent, state, questions), warmup, iterations)
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
    mkpath(dirname(abspath(opts["output"])))
    open(io -> JSON.json(io, report; pretty=true), opts["output"], "w")
end

main(ARGS)
