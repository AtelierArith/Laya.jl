# Public inference runtime; prompt and result formats follow upstream Laya via
# `laya_mlx/agent.py`.

"""
    resolve_model(model_id_or_path; subfolder=nothing, revision=nothing) -> String

A local checkpoint directory, or a Hugging Face repository. A repository is looked up in the
Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`, shared with
Python), then in Laya's scratch space, and otherwise downloaded into the scratch space
(disabled when `HF_HUB_OFFLINE` is set). `HF_TOKEN` is sent for private repositories.
"""
function resolve_model(model_id_or_path::AbstractString; subfolder=nothing, revision=nothing)
    if subfolder !== nothing
        parts = splitpath(subfolder)
        (isabspath(subfolder) || ".." in parts) &&
            throw(ArgumentError("subfolder must be a relative path inside the model repository"))
    end
    path = expanduser(model_id_or_path)
    if !isdir(path)
        startswith(model_id_or_path, r"/|\./|\.\./|~") &&
            throw(ArgumentError("Local model directory does not exist: $model_id_or_path"))
        path = hub_snapshot(model_id_or_path, something(revision, "main"))
    end
    subfolder === nothing || (path = joinpath(path, subfolder))
    for name in ("model.safetensors", "rl_agent_config.json", joinpath("encoder", "config.json"))
        isfile(joinpath(path, name)) || throw(ArgumentError("Not a complete Laya checkpoint: $(joinpath(path, name)) is missing"))
    end
    path
end

function hub_cache()
    haskey(ENV, "HF_HUB_CACHE") && return ENV["HF_HUB_CACHE"]
    haskey(ENV, "HF_HOME") && return joinpath(ENV["HF_HOME"], "hub")
    joinpath(homedir(), ".cache", "huggingface", "hub")
end

# Both caches use the Hub layout `models--org--name/{refs/<revision>,snapshots/<commit>}`.
scratch_hub() = @get_scratch!("hub")

function cached_snapshot(cache::AbstractString, repo::AbstractString, revision::AbstractString)
    root = joinpath(cache, "models--" * replace(repo, "/" => "--"))
    ref = joinpath(root, "refs", revision)
    commit = isfile(ref) ? strip(read(ref, String)) : revision
    dir = joinpath(root, "snapshots", commit)
    isdir(dir) ? dir : nothing
end

function hub_snapshot(repo::AbstractString, revision::AbstractString)
    for cache in (hub_cache(), scratch_hub())
        dir = cached_snapshot(cache, repo, revision)
        dir === nothing || return dir
    end
    offline = lowercase(get(ENV, "HF_HUB_OFFLINE", "0")) in ("1", "true", "yes", "on")
    offline && throw(ArgumentError("$repo@$revision is not cached and HF_HUB_OFFLINE is set"))
    hub_download(repo, revision)
end

hub_endpoint() = rstrip(get(ENV, "HF_ENDPOINT", "https://huggingface.co"), '/')
hub_headers() = haskey(ENV, "HF_TOKEN") ? ["Authorization" => "Bearer $(ENV["HF_TOKEN"])"] : Pair{String,String}[]

"""
    hub_download(repo, revision="main") -> String

Download every file of the Hugging Face model repository `repo` at `revision` into Laya's
scratch space and return the snapshot directory. The snapshot appears only once complete.
"""
function hub_download(repo::AbstractString, revision::AbstractString="main")
    occursin(r"^[\w.-]+/[\w.-]+$", repo) || throw(ArgumentError("Invalid Hugging Face repository id: $repo"))
    headers = hub_headers()
    info = JSON.parse(String(take!(Downloads.download(
        "$(hub_endpoint())/api/models/$repo/revision/$(escape_path(revision))", IOBuffer(); headers))))
    commit = String(info["sha"])
    files = String[s["rfilename"] for s in info["siblings"]]
    root = joinpath(scratch_hub(), "models--" * replace(repo, "/" => "--"))
    dir = joinpath(root, "snapshots", commit)
    if !isdir(dir)
        tmp = mktempdir(mkpath(root); prefix="download-")
        try
            for file in files
                any(==(".."), splitpath(file)) && throw(ArgumentError("Unsafe file name in $repo: $file"))
                dest = joinpath(tmp, file)
                mkpath(dirname(dest))
                @info "Downloading $repo/$file"
                Downloads.download("$(hub_endpoint())/$repo/resolve/$commit/$(escape_path(file))", dest; headers)
            end
            mkpath(dirname(dir))
            mv(tmp, dir)
        finally
            ispath(tmp) && rm(tmp; recursive=true, force=true)
        end
    end
    ref = joinpath(root, "refs", revision)
    mkpath(dirname(ref))
    write(ref, commit)
    dir
end

escape_path(path::AbstractString) = join(map(p -> replace(p, r"[^\w.~-]" => c -> join("%" * uppercase(string(b; base=16, pad=2)) for b in codeunits(c))), split(path, '/')), '/')

"""
    Agent

A loaded Laya checkpoint: tokenizer, model and calibration. Create with [`load`](@ref).
`model` is whatever the backend loaded (a [`DecisionModel`](@ref) on the CPU).
"""
struct Agent{M}
    model_id::String
    model_dir::String
    cfg::Dict{String,Any}
    encoder_cfg::EncoderConfig
    tok::Tokenizer
    model::M
    batch_size::Int
    temperature_raw::Vector{Float64}
    temperature_by_options_raw::Dict{String,Float64}
    temperature::Vector{Float64}
    temperature_by_options::Dict{String,Float64}
end

"""
    load(model_id_or_path="convaiinnovations/laya"; dtype=Float32, batch_size=16, subfolder=nothing,
         revision=nothing, backend=CPUBackend())

Load a Laya checkpoint (local directory or Hugging Face repository, downloaded on first use; see
[`resolve_model`](@ref)). The tokenizer, prompts and calibration always run in Julia; the model
forward runs on `backend` (see [`Backend`](@ref)).
"""
function load(model_id_or_path::AbstractString="convaiinnovations/laya"; dtype::Type{T}=Float32,
              batch_size::Integer=16, subfolder=nothing, revision=nothing, backend=CPUBackend()) where {T}
    batch_size >= 1 || throw(ArgumentError("batch_size must be a positive integer"))
    dir = resolve_model(model_id_or_path; subfolder, revision)
    cfg = Dict{String,Any}(JSON.parsefile(joinpath(dir, "rl_agent_config.json")))
    (haskey(cfg, "encoder") && haskey(cfg, "head_layers")) ||
        throw(ArgumentError("Laya config must specify encoder and head_layers"))
    enc_cfg = EncoderConfig(JSON.parsefile(joinpath(dir, "encoder", "config.json")))
    max_len, head_max_len = get(cfg, "max_len", 512), get(cfg, "head_max_len", 192)
    4 < head_max_len < max_len <= enc_cfg.max_position_embeddings ||
        throw(ArgumentError("Expected 4 < head_max_len < max_len <= max_position_embeddings"))
    traw = Float64.(get(cfg, "temperature", [1.0, 1.0, 1.0]))
    braw = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in get(cfg, "temperature_by_options", Dict()))
    (length(traw) == 3 && all(t -> isfinite(t) && t > 0, [traw; collect(values(braw))])) ||
        throw(ArgumentError("Calibration temperatures must be finite and positive"))
    rejected = [["$k=$(round(v; sigdigits=4))" for (k, v) in braw if clamp_temperature(v) != v];
                ["temperature[$(i-1)]=$(round(t; sigdigits=4))" for (i, t) in enumerate(traw) if clamp_temperature(t) != t]]
    isempty(rejected) || @warn "This checkpoint ships temperatures outside [$TEMP_MIN, $TEMP_MAX] which would distort " *
        "confidence; clamping $(join(rejected, ", ")). Treat confidence from the affected buckets as uncalibrated."
    model = load_backend_model(backend, dir, T)
    Agent(String(model_id_or_path), dir, cfg, enc_cfg, Tokenizer(joinpath(dir, "tokenizer")), model, batch_size,
        traw, braw, clamp_temperature.(traw), Dict(k => clamp_temperature(v) for (k, v) in braw))
end

"""Validate a public question definition and convert it to a [`Question`](@ref)."""
function to_internal(qdef)
    qdef isa AbstractDict || throw(ArgumentError("Each question must be a dictionary"))
    kind = get(qdef, "type", nothing)
    haskey(QTYPES, kind) || throw(ArgumentError("Unknown question type $(repr(kind)); expected choice, score, or noul"))
    haskey(qdef, "instructions") || throw(ArgumentError("Question is missing instructions"))
    criteria = get(qdef, "criteria", nothing)
    if kind == "choice"
        if criteria isa AbstractVector
            all(c -> c isa AbstractString, criteria) || throw(ArgumentError("Choice labels must be strings"))
            allunique(criteria) || throw(ArgumentError("Choice labels must be unique"))
            criteria = [String(c) => nothing for c in criteria]
        elseif criteria isa AbstractDict
            all(k -> k isa AbstractString, keys(criteria)) || throw(ArgumentError("Choice labels must be strings"))
            criteria = [String(k) => v for (k, v) in criteria]
        else
            criteria = nothing
        end
        (criteria === nothing || isempty(criteria)) && throw(ArgumentError("Choice criteria must be a nonempty dictionary or list"))
    elseif kind == "score"
        (criteria isa AbstractVector && !isempty(criteria)) || throw(ArgumentError("Score criteria must be a nonempty list"))
    elseif criteria !== nothing && !(criteria isa AbstractDict)
        throw(ArgumentError("Noul criteria must be a dictionary with false/true descriptions"))
    end
    ins = qdef["instructions"]
    Question(kind, ins isa AbstractString ? String(ins) : py_json(ins; ascii=true), criteria)
end

"""
    prepare(agent, state, questions) -> (items, internal)

Tokenized `(ids, markers, qtype)` per question, in question order, plus the internal questions.
"""
function prepare(agent::Agent, state, questions::AbstractDict)
    items = NamedTuple{(:ids, :markers, :qtype),Tuple{Vector{Int},Vector{Int},Int}}[]
    internal = Question[]
    max_len, head_max_len = get(agent.cfg, "max_len", 512), get(agent.cfg, "head_max_len", 192)
    for (qid, definition) in questions
        q = to_internal(definition)
        ids, markers = build_sequence(agent.tok, state, q; max_len, head_max_len)
        length(markers) == length(render_options(q)) ||
            throw(ArgumentError("Question $(repr(qid)) has too many options for the token budget"))
        push!(items, (ids=ids, markers=markers, qtype=QTYPES[q.t]))
        push!(internal, q)
    end
    items, internal
end

"""
    collate(items, pad_id) -> Dict{String,Array}

Padded batch in the model layout: `input_ids`/`attention_mask` `(L, n)`, `marker_pos`/
`marker_mask` `(K, n)` with `K >= 2`, `qtype` `(n,)`.
"""
function collate(items, pad_id::Integer)
    isempty(items) && throw(ArgumentError("Cannot collate an empty batch"))
    n, L = length(items), maximum(it -> length(it.ids), items)
    K = max(2, maximum(it -> length(it.markers), items))
    batch = Dict{String,Array}(
        "input_ids" => fill(Int32(pad_id), L, n),
        "attention_mask" => falses(L, n) |> Array{Bool},
        "marker_pos" => zeros(Int32, K, n),
        "marker_mask" => falses(K, n) |> Array{Bool},
        "qtype" => Int32[it.qtype for it in items],
    )
    for (j, it) in enumerate(items)
        l, k = length(it.ids), length(it.markers)
        batch["input_ids"][1:l, j] .= it.ids
        batch["attention_mask"][1:l, j] .= true
        batch["marker_pos"][1:k, j] .= it.markers
        batch["marker_mask"][1:k, j] .= true
    end
    batch
end

r4(x) = round(Float64(x); digits=4)

"""
    predict(agent, state, questions) -> JSON.Object

Answer typed questions about `state` (a string, or a dictionary/vector serialized as JSON).
`questions` maps question ids to `Dict("type" => "choice"|"score"|"noul", "instructions" =>
..., "criteria" => ...)`. The result mirrors upstream Laya: `model`, `answers` (per question:
`type`, `confidence`, `action.act_probability`, and `choice`/`score`/`noul` fields) and `usage`.
"""
function predict(agent::Agent, state, questions::AbstractDict)
    items, internal = prepare(agent, state, questions)
    qids = collect(keys(questions))
    answers = JSON.Object{String,Any}()
    for start in 1:agent.batch_size:length(items)
        stop = min(start + agent.batch_size - 1, length(items))
        chunk = items[start:stop]
        logits, act = agent.model(collate(chunk, agent.tok.pad_token_id))
        (all(isfinite, logits) && all(isfinite, act)) || throw(DomainError(logits, "Non-finite model outputs"))
        act = exp.(act .- maximum(act; dims=1))
        act ./= sum(act; dims=1)
        for (row, item) in enumerate(chunk)
            q = internal[start+row-1]
            k, qt = length(item.markers), item.qtype
            scale = get(agent.temperature_by_options, temp_bucket(qt, k), agent.temperature[qt+1])
            z = logits[1:k, row] ./ Float32(scale)
            p = exp.(z .- maximum(z))
            p ./= sum(p)
            answer = JSON.Object{String,Any}(
                "type" => q.t,
                "confidence" => r4(confidence_from_probs(p, k)),
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

"""
    system_one(agent, state, questions) -> JSON.Object

The same as [`predict`](@ref), under upstream Laya's name.
"""
const system_one = predict
