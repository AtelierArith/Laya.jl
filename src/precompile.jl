# Precompile workload: a tiny random checkpoint written to a temporary directory, loaded and
# queried through the public API, so the package image already holds the code for
# `load` (safetensors, config, tokenizer, model) and `predict` (prompts, batching, forward,
# calibration, results). The tokenizer is a small byte-level BPE with NFC, as the English
# checkpoint uses; a Metaspace/byte-fallback BPE, as the multilingual one uses, is also run.

using PrecompileTools: @compile_workload, @setup_workload

"""Write `tensors` (Julia-layout arrays) as a safetensors file (shapes reversed, row-major)."""
function write_safetensors(path, tensors::AbstractDict{String,<:Array{Float16}})
    header, offset = Dict{String,Any}(), 0
    for name in sort!(collect(keys(tensors)))
        n = sizeof(tensors[name])
        header[name] = Dict("dtype" => "F16", "shape" => collect(reverse(size(tensors[name]))),
                            "data_offsets" => [offset, offset + n])
        offset += n
    end
    json = JSON.json(header)
    open(path, "w") do io
        write(io, htol(UInt64(ncodeunits(json))), json)
        for name in sort!(collect(keys(tensors)))
            write(io, htol.(vec(tensors[name])))
        end
    end
end

function write_tiny_checkpoint(dir; metaspace::Bool=false)
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    if metaspace
        pieces = ["▁"; string.(collect("abcdefghijklmnopqrstuvwxyz{}\":,.?")); ["<0x$(uppercase(string(b; base=16, pad=2)))>" for b in 0:255]]
        normalizer = Dict("type" => "Replace", "pattern" => Dict("String" => " "), "content" => "▁")
        pretokenizer = Dict("type" => "Metaspace", "replacement" => "▁", "prepend_scheme" => "always", "split" => true)
        extra = Dict("unk_token" => "[UNK]", "byte_fallback" => true, "fuse_unk" => true)
    else
        pieces = string.(BYTE_TO_CHAR)
        normalizer = Dict("type" => "NFC")
        pretokenizer = Dict("type" => "ByteLevel", "add_prefix_space" => false, "trim_offsets" => true, "use_regex" => true)
        extra = Dict{String,Any}()
    end
    vocab = Dict(t => i - 1 for (i, t) in enumerate([specials; pieces; ["he", "ll"]]))
    model = merge(Dict("type" => "BPE", "vocab" => vocab, "merges" => ["h e", "l l"]), extra)
    added = [Dict("id" => vocab[t], "content" => t, "single_word" => false, "lstrip" => false,
                  "rstrip" => false, "normalized" => false, "special" => true) for t in specials]
    mkpath(joinpath(dir, "tokenizer"))
    write(joinpath(dir, "tokenizer", "tokenizer.json"), JSON.json(Dict(
        "added_tokens" => added, "normalizer" => normalizer, "pre_tokenizer" => pretokenizer, "model" => model)))
    write(joinpath(dir, "tokenizer", "tokenizer_config.json"), JSON.json(Dict(
        "pad_token" => "[PAD]", "cls_token" => "[CLS]", "sep_token" => "[SEP]", "mask_token" => "[MASK]")))

    cfg = Dict("model_type" => "modernbert", "vocab_size" => length(vocab), "hidden_size" => 64,
               "intermediate_size" => 96, "num_hidden_layers" => 2, "num_attention_heads" => 1,
               "local_attention" => 16, "max_position_embeddings" => 256)
    agent_cfg = Dict("encoder" => "precompile/tiny", "head_layers" => 1, "max_len" => 128, "head_max_len" => 32,
                     "act_costs" => Dict("escalate" => 0.5), "temperature" => [1.3, 1.1, 2.0])
    mkpath(joinpath(dir, "encoder"))
    write(joinpath(dir, "encoder", "config.json"), JSON.json(cfg))
    write(joinpath(dir, "rl_agent_config.json"), JSON.json(agent_cfg))
    # Random weights under exactly the names and sizes the model reads.
    tensors = Dict{String,Array{Float16}}()
    function random_weight(name, dims...)
        w = tensors[name] = Float16.(0.02f0 .* randn(Float32, dims...))
        Float32.(w)
    end
    DecisionModel{Float32}(EncoderConfig(cfg), agent_cfg, random_weight)
    write_safetensors(joinpath(dir, "model.safetensors"), tensors)
    dir
end

@setup_workload begin
    questions = Dict(
        "topic" => Dict("type" => "choice", "instructions" => "Choose", "criteria" => ["a", "b", "c"]),
        "team" => Dict("type" => "choice", "instructions" => "Route", "criteria" => Dict("x" => "first", "y" => "second")),
        "level" => Dict("type" => "score", "instructions" => "Level", "criteria" => ["low", "high"]),
        "yes" => Dict("type" => "noul", "instructions" => "Is this true?"),
    )
    @compile_workload begin
        mktempdir() do root
            ACCELERATE_HINT_SHOWN[] = true      # no @info while precompiling
            agent = load(write_tiny_checkpoint(joinpath(root, "bytelevel")); batch_size=2)
            predict(agent, "hello world", questions)
            predict(agent, Dict("body" => "hello", "items" => [1, 2.5, nothing, true]), questions)
            tok = Tokenizer(joinpath(write_tiny_checkpoint(joinpath(root, "metaspace"); metaspace=true), "tokenizer"))
            tok("hello {\"x\": 1} ünïcode")
            ACCELERATE_HINT_SHOWN[] = false
        end
    end
end
