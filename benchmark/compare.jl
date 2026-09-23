# Summarize benchmark/results/*.json as a Markdown table.
#
#     julia --project=benchmark benchmark/compare.jl [results_dir]
#
# Rows are grouped by model, dtype and workload; `vs python` is the Julia e2e p50 divided by
# the python-mlx-gpu e2e p50 of the same model/dtype/workload (>1 means Julia is slower).
# `answers` checks that the selected answers agree with the Python run.

using JSON

dir = isempty(ARGS) ? joinpath(@__DIR__, "results") : ARGS[1]
reports = [JSON.parsefile(f) for f in readdir(dir; join=true) if endswith(f, ".json")]
isempty(reports) && error("No results in $dir")

rows = [(r, x) for r in reports for x in r["results"]]
key(r, x) = (r["model"], r["dtype"], x["workload"])
baseline = Dict(key(r, x) => x for (r, x) in rows if r["backend"] == "python-mlx-gpu")

fmt(v) = v === nothing ? "–" : string(round(v; sigdigits=4))
println("| model | dtype | backend | workload | seq len | prepare p50 ms | forward p50 ms | e2e p50 ms | e2e p95 ms | q/s | vs python | answers |")
println("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for (r, x) in sort(rows; by=((r, x),) -> (r["model"], r["dtype"], x["workload"], r["backend"]))
    base = get(baseline, key(r, x), nothing)
    ratio = base === nothing || r["backend"] == "python-mlx-gpu" ? nothing : x["e2e"]["p50_ms"] / base["e2e"]["p50_ms"]
    same = base === nothing || r["backend"] == "python-mlx-gpu" ? "–" : (x["answers"] == base["answers"] ? "same" : "DIFFER")
    println("| ", join([r["model"], r["dtype"], r["backend"], x["workload"], x["sequence_length"],
        fmt(x["prepare"]["p50_ms"]), fmt(x["forward"]["p50_ms"]), fmt(x["e2e"]["p50_ms"]), fmt(x["e2e"]["p95_ms"]),
        fmt(x["e2e"]["questions_per_second"]), ratio === nothing ? "–" : fmt(ratio) * "×", same], " | "), " |")
end
println()
for r in sort(reports; by=r -> (r["model"], r["dtype"], r["backend"]))
    println("- ", r["backend"], " / ", r["model"], " / ", r["dtype"], ": load ", fmt(r["load_seconds"]), " s, max RSS ",
            fmt(r["max_rss_bytes"] / 2^30), " GiB, ", r["iterations"], " iterations after ", r["warmup"], " warmup")
end
