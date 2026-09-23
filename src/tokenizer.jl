# Pure-Julia encoder for Hugging Face `tokenizer.json` files: the subset used by Laya
# checkpoints (ModernBERT byte-level BPE, mmBERT/Gemma Metaspace BPE with byte fallback)
# plus WordLevel/Whitespace for tests. Only token ids are produced; offsets are not tracked.
# Semantics follow the Rust `tokenizers` crate (added-token extraction, normalizers,
# pre-tokenizers and BPE merging).

const unicode_normalize = Base.Unicode.normalize

# ----------------------------------------------------------------------------- added tokens

struct AddedToken
    content::String
    id::Int
    lstrip::Bool
    rstrip::Bool
    single_word::Bool
end

"""Leftmost-longest matcher over added tokens, as the crate's Aho-Corasick trie."""
struct AddedMatcher
    by_first::Dict{Char,Vector{AddedToken}}   # candidates sorted by length, longest first
end

function AddedMatcher(tokens::AbstractVector{AddedToken})
    by_first = Dict{Char,Vector{AddedToken}}()
    for t in tokens
        isempty(t.content) && continue
        push!(get!(by_first, first(t.content), AddedToken[]), t)
    end
    foreach(v -> sort!(v; by=t -> -ncodeunits(t.content)), values(by_first))
    AddedMatcher(by_first)
end

is_word_char(c::Char) = isletter(c) || isnumeric(c) || c == '_'

"""
    split_added(m, text) -> Vector{Union{String,Int}}

Split `text` around added tokens: plain pieces stay `String`, matches become their id.
`lstrip`/`rstrip` tokens also swallow adjacent whitespace, as in the crate.
"""
function split_added(m::AddedMatcher, text::String)
    out = Union{String,Int}[]
    isempty(m.by_first) && return push!(out, text)
    last = 1                                   # start of the pending plain piece
    i = 1
    n = ncodeunits(text)
    while i <= n
        tok = nothing
        candidates = get(m.by_first, text[i], nothing)
        if candidates !== nothing
            for t in candidates
                stop = i + ncodeunits(t.content) - 1
                stop <= n && startswith(SubString(text, i), t.content) || continue
                if t.single_word
                    before = i > 1 && is_word_char(text[prevind(text, i)])
                    after = stop < n && is_word_char(text[nextind(text, stop)])
                    (before || after) && continue
                end
                tok = t
                break
            end
        end
        if tok === nothing
            i = nextind(text, i)
            continue
        end
        start, stop = i, i + ncodeunits(tok.content) - 1
        if tok.lstrip
            while start > last && isspace(text[prevind(text, start)])
                start = prevind(text, start)
            end
        end
        if tok.rstrip
            while stop < n && isspace(text[nextind(text, stop)])
                stop = nextind(text, stop)
            end
        end
        start > last && push!(out, text[last:prevind(text, start)])
        push!(out, tok.id)
        last = i = stop + 1
    end
    last <= n && push!(out, text[last:end])
    out
end

# ----------------------------------------------------------------------------- normalizers

abstract type Normalizer end
struct NoNormalizer <: Normalizer end
struct UnicodeNormalizer <: Normalizer
    form::Symbol
end
struct Lowercase <: Normalizer end
struct Replace <: Normalizer
    pattern::String
    content::String
end
struct Prepend <: Normalizer
    prepend::String
end
struct NormalizerSequence <: Normalizer
    normalizers::Vector{Normalizer}
end

normalize_text(::NoNormalizer, s) = s
normalize_text(n::UnicodeNormalizer, s) = unicode_normalize(s, n.form)
normalize_text(::Lowercase, s) = lowercase(s)
normalize_text(n::Replace, s) = replace(s, n.pattern => n.content)
normalize_text(n::Prepend, s) = isempty(s) ? s : n.prepend * s
normalize_text(n::NormalizerSequence, s) = foldl((acc, x) -> normalize_text(x, acc), n.normalizers; init=s)

function Normalizer(spec)
    spec === nothing && return NoNormalizer()
    kind = spec["type"]
    kind in ("NFC", "NFD", "NFKC", "NFKD") && return UnicodeNormalizer(Symbol(kind))
    kind == "Lowercase" && return Lowercase()
    kind == "Prepend" && return Prepend(spec["prepend"])
    kind == "Sequence" && return NormalizerSequence(Normalizer[Normalizer(s) for s in spec["normalizers"]])
    if kind == "Replace"
        haskey(spec["pattern"], "String") || error("Only string Replace patterns are supported")
        return Replace(spec["pattern"]["String"], spec["content"])
    end
    error("Unsupported normalizer: $kind")
end

# ----------------------------------------------------------------------------- pre-tokenizers

abstract type PreTokenizer end
struct NoPreTokenizer <: PreTokenizer end

const GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""GPT-2 byte-to-unicode table: printable bytes map to themselves, the rest to U+0100...."""
const BYTE_TO_CHAR = let
    keep = [0x21:0x7e; 0xa1:0xac; 0xae:0xff]
    table = Vector{Char}(undef, 256)
    extra = 0
    for b in 0x00:0xff
        if b in keep
            table[b+1] = Char(b)
        else
            table[b+1] = Char(256 + extra)
            extra += 1
        end
    end
    table
end

struct ByteLevel <: PreTokenizer
    add_prefix_space::Bool
    use_regex::Bool
end

struct Metaspace <: PreTokenizer
    replacement::Char
    prepend::Symbol       # :always, :first, :never
    split::Bool
end

struct Whitespace <: PreTokenizer end
struct PreTokenizerSequence <: PreTokenizer
    pretokenizers::Vector{PreTokenizer}
end

# Each pre-tokenizer maps one piece to words. `first` marks the piece at text offset 0.
pretokenize(::NoPreTokenizer, s, first) = [s]

function pretokenize(p::ByteLevel, s, first)
    p.add_prefix_space && !startswith(s, ' ') && (s = " " * s)
    words = p.use_regex ? [m.match for m in eachmatch(GPT2_PATTERN, s)] : [s]
    [String([BYTE_TO_CHAR[b+1] for b in codeunits(w)]) for w in words]
end

function pretokenize(p::Metaspace, s, first)
    r = string(p.replacement)
    s = replace(s, ' ' => r)
    if (p.prepend === :always || (p.prepend === :first && first)) && !startswith(s, p.replacement)
        s = r * s
    end
    p.split || return [s]
    # Split with the delimiter merged into the following word: "▁a▁▁b" -> ["▁a", "▁", "▁b"].
    words = String[]
    start = 1
    for i in eachindex(s)
        if s[i] == p.replacement && i > start
            push!(words, s[start:prevind(s, i)])
            start = i
        end
    end
    start <= ncodeunits(s) && push!(words, s[start:end])
    words
end

pretokenize(::Whitespace, s, first) = [m.match for m in eachmatch(r"\w+|[^\w\s]+", s)]

function pretokenize(p::PreTokenizerSequence, s, first)
    words = [s]
    for q in p.pretokenizers
        words = reduce(vcat, (pretokenize(q, w, first) for w in words); init=String[])
    end
    words
end

function PreTokenizer(spec)
    spec === nothing && return NoPreTokenizer()
    kind = spec["type"]
    kind == "ByteLevel" && return ByteLevel(get(spec, "add_prefix_space", true), get(spec, "use_regex", true))
    kind == "Whitespace" && return Whitespace()
    kind == "Sequence" && return PreTokenizerSequence(PreTokenizer[PreTokenizer(s) for s in spec["pretokenizers"]])
    if kind == "Metaspace"
        scheme = get(spec, "prepend_scheme", get(spec, "add_prefix_space", true) ? "always" : "never")
        return Metaspace(only(spec["replacement"]), Symbol(scheme), get(spec, "split", true))
    end
    error("Unsupported pre-tokenizer: $kind")
end

# ----------------------------------------------------------------------------- models

abstract type TokenModel end

struct WordLevel <: TokenModel
    vocab::Dict{String,Int}
    unk::Union{Nothing,Int}
end

function tokenize_word(m::WordLevel, word)
    id = get(m.vocab, word, m.unk)
    id === nothing && error("Word $(repr(word)) is not in the vocabulary and there is no unk token")
    [id]
end

struct BPE <: TokenModel
    vocab::Dict{String,Int}
    merges::Dict{Tuple{Int,Int},Tuple{Int,Int}}   # (left, right) => (rank, merged id)
    unk::Union{Nothing,Int}
    fuse_unk::Bool
    byte_fallback::Bool
    ignore_merges::Bool
    byte_ids::Vector{Int}                           # id of "<0xXX>" per byte, or -1
    cache::Dict{String,Vector{Int}}
end

function BPE(spec)
    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in spec["vocab"])
    merges = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (rank, m) in enumerate(spec["merges"])
        a, b = m isa AbstractString ? split(m, ' '; limit=2) : (m[1], m[2])
        merges[(vocab[a], vocab[b])] = (rank, vocab[a*b])
    end
    get(spec, "continuing_subword_prefix", nothing) === nothing &&
        get(spec, "end_of_word_suffix", nothing) === nothing ||
        error("BPE subword prefixes/suffixes are not supported")
    unk_token = get(spec, "unk_token", nothing)
    byte_ids = [get(vocab, string("<0x", uppercase(string(b; base=16, pad=2)), ">"), -1) for b in 0x00:0xff]
    BPE(vocab, merges, unk_token === nothing ? nothing : vocab[unk_token],
        something(get(spec, "fuse_unk", false), false), something(get(spec, "byte_fallback", false), false),
        something(get(spec, "ignore_merges", false), false), byte_ids, Dict{String,Vector{Int}}())
end

tokenize_word(m::BPE, word) = get!(() -> bpe_word(m, word), m.cache, word)

function bpe_word(m::BPE, word::String)
    m.ignore_merges && haskey(m.vocab, word) && return [m.vocab[word]]
    symbols = Int[]
    pending_unk = false
    for c in word
        id = get(m.vocab, string(c), nothing)
        if id !== nothing
            pending_unk && (push!(symbols, m.unk); pending_unk = false)
            push!(symbols, id)
            continue
        end
        if m.byte_fallback
            bytes = [m.byte_ids[b+1] for b in codeunits(string(c))]
            if all(>=(0), bytes)
                append!(symbols, bytes)
                continue
            end
        end
        m.unk === nothing && continue
        if pending_unk && !m.fuse_unk
            push!(symbols, m.unk)
        end
        pending_unk = true
    end
    pending_unk && push!(symbols, m.unk)
    # Repeatedly apply the lowest-ranked merge, leftmost first on ties.
    while length(symbols) > 1
        best, pos, new = typemax(Int), 0, 0
        for i in 1:length(symbols)-1
            r = get(m.merges, (symbols[i], symbols[i+1]), nothing)
            r === nothing && continue
            if r[1] < best
                best, pos, new = r[1], i, r[2]
            end
        end
        pos == 0 && break
        symbols[pos] = new
        deleteat!(symbols, pos + 1)
    end
    symbols
end

function TokenModel(spec)
    kind = spec["type"]
    kind == "BPE" && return BPE(spec)
    if kind == "WordLevel"
        vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in spec["vocab"])
        unk = get(spec, "unk_token", nothing)
        return WordLevel(vocab, unk === nothing ? nothing : vocab[unk])
    end
    error("Unsupported tokenizer model: $kind")
end

# ----------------------------------------------------------------------------- tokenizer

"""
    Tokenizer(dir)

Load `tokenizer.json` and `tokenizer_config.json` from a checkpoint's `tokenizer/`
directory. Calling `tok(text)` returns 0-based token ids without special tokens, like
`tokenizers.Tokenizer.encode(text, add_special_tokens=False).ids`.
"""
struct Tokenizer
    normalizer::Normalizer
    pretokenizer::PreTokenizer
    model::TokenModel
    added_raw::AddedMatcher          # matched before normalization
    added_normalized::AddedMatcher   # matched on normalized text
    token_to_id::Dict{String,Int}
    cls_token::String
    sep_token::String
    pad_token::String
    mask_token::String
    cls_token_id::Int
    sep_token_id::Int
    pad_token_id::Int
    mask_token_id::Int
end

function Tokenizer(dir::AbstractString)
    spec = JSON.parsefile(joinpath(dir, "tokenizer.json"))
    config = JSON.parsefile(joinpath(dir, "tokenizer_config.json"))
    model = TokenModel(spec["model"])
    token_to_id = copy(model.vocab)
    raw, normed = AddedToken[], AddedToken[]
    for a in spec["added_tokens"]
        t = AddedToken(a["content"], a["id"], a["lstrip"], a["rstrip"], a["single_word"])
        token_to_id[t.content] = t.id
        push!(a["normalized"] ? normed : raw, t)
    end
    specials = map(("cls_token", "sep_token", "pad_token", "mask_token")) do name
        value = get(config, name, nothing)
        value isa AbstractDict && (value = get(value, "content", nothing))
        value isa AbstractString && haskey(token_to_id, value) || error("Tokenizer is missing a valid $name")
        String(value)
    end
    Tokenizer(Normalizer(get(spec, "normalizer", nothing)), PreTokenizer(get(spec, "pre_tokenizer", nothing)),
        model, AddedMatcher(raw), AddedMatcher(normed), token_to_id,
        specials..., (token_to_id[s] for s in specials)...)
end

function (tok::Tokenizer)(text::AbstractString)
    ids = Int[]
    first = true
    for piece in split_added(tok.added_raw, String(text))
        if piece isa Int
            push!(ids, piece)
        else
            for sub in split_added(tok.added_normalized, normalize_text(tok.normalizer, piece))
                if sub isa Int
                    push!(ids, sub)
                elseif !isempty(sub)
                    for word in pretokenize(tok.pretokenizer, sub, first)
                        append!(ids, tokenize_word(tok.model, String(word)))
                    end
                end
            end
        end
        first = false
    end
    ids
end
