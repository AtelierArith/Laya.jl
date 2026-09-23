# Laya prompt construction, adapted from upstream via `laya_mlx/common.py`.
#
# Dictionaries are rendered in iteration order, which decides option order for `choice`
# questions. Use an insertion-ordered dictionary (e.g. `JSON.Object`, as returned by
# `JSON.parse`) or a vector of labels when order matters; `Dict` order is arbitrary.

const QTYPES = Dict("choice" => 0, "score" => 1, "noul" => 2)
const QTYPE_NAMES = Dict(v => k for (k, v) in QTYPES)

# ----------------------------------------------------------------------------- Python json.dumps

"""Python `repr(float)`: shortest round-trip digits, scientific below 1e-4 or from 1e16."""
function py_float(x::AbstractFloat)
    x = Float64(x)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Infinity" : "-Infinity"
    x == 0 && return signbit(x) ? "-0.0" : "0.0"
    s = string(abs(x))                                   # e.g. "123.45", "1.0e-5"
    mant, e = occursin('e', s) ? (split(s, 'e')[1], parse(Int, split(s, 'e')[2])) : (s, 0)
    ip, fp = occursin('.', mant) ? split(mant, '.') : (mant, "")
    digits = ip * fp
    point = length(ip) + e                               # decimal point after `point` digits
    lead = length(digits) - length(lstrip(digits, '0'))
    digits = rstrip(lstrip(digits, '0'), '0')
    point -= lead
    exp10 = point - 1
    sign = x < 0 ? "-" : ""
    if -4 <= exp10 < 16
        if point <= 0
            return sign * "0." * "0"^(-point) * digits
        elseif point >= length(digits)
            return sign * digits * "0"^(point - length(digits)) * ".0"
        else
            return sign * digits[1:point] * "." * digits[point+1:end]
        end
    end
    m = length(digits) == 1 ? digits : digits[1:1] * "." * digits[2:end]
    sign * m * "e" * (exp10 < 0 ? "-" : "+") * lpad(string(abs(exp10)), 2, '0')
end

function py_json_string(io::IO, s::AbstractString, ascii::Bool)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c == '\b'
            print(io, "\\b")
        elseif c == '\f'
            print(io, "\\f")
        elseif c < ' ' || (ascii && c > '~')
            u = UInt32(c)
            if u > 0xffff                                  # UTF-16 surrogate pair
                u -= 0x10000
                print(io, "\\u", string(0xd800 + (u >> 10); base=16, pad=4), "\\u", string(0xdc00 + (u & 0x3ff); base=16, pad=4))
            else
                print(io, "\\u", string(u; base=16, pad=4))
            end
        else
            print(io, c)
        end
    end
    print(io, '"')
end

py_json_key(k::AbstractString) = k
py_json_key(k::Bool) = k ? "true" : "false"
py_json_key(k::Integer) = string(k)
py_json_key(k::AbstractFloat) = py_float(k)
py_json_key(::Nothing) = "null"
py_json_key(k::Symbol) = String(k)

function py_json(io::IO, x, ascii::Bool)
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        print(io, py_float(x))
    elseif x isa AbstractString || x isa Symbol
        py_json_string(io, string(x), ascii)
    elseif x isa AbstractDict
        print(io, '{')
        for (i, (k, v)) in enumerate(x)
            i > 1 && print(io, ", ")
            py_json_string(io, py_json_key(k), ascii)
            print(io, ": ")
            py_json(io, v, ascii)
        end
        print(io, '}')
    elseif x isa Union{AbstractVector,Tuple}
        print(io, '[')
        for (i, v) in enumerate(x)
            i > 1 && print(io, ", ")
            py_json(io, v, ascii)
        end
        print(io, ']')
    else
        py_json_string(io, string(x), ascii)             # json.dumps(..., default=str)
    end
end

"""`json.dumps(x, ensure_ascii=ascii)` with Python's default `(", ", ": ")` separators."""
py_json(x; ascii::Bool=false) = sprint(io -> py_json(io, x, ascii))

# ----------------------------------------------------------------------------- prompts

serialize_state(state::AbstractString) = String(state)
serialize_state(state) = py_json(state)

render_criterion(value::AbstractString) = String(value)
render_criterion(value) = py_json(value)

is_blank(v) = v === nothing || (v isa AbstractString && isempty(v))

"""Question in internal form: type, instructions and validated criteria."""
struct Question
    t::String
    ins::String
    crit::Any      # choice: ordered label => description pairs; score: vector; noul: dict or nothing
end

"""Option texts in label-index order. Noul is always `[false, true]`."""
function render_options(q::Question)
    if q.t == "choice"
        return [is_blank(v) ? k : "$k: $(render_criterion(v))" for (k, v) in q.crit]
    elseif q.t == "score"
        return ["level $(i-1): $(render_criterion(c))" for (i, c) in enumerate(q.crit)]
    end
    crit = q.crit === nothing ? Dict{String,Any}() : q.crit
    f, t = get(crit, "false", nothing), get(crit, "true", nothing)
    ["false: " * (is_blank(f) ? "no, the statement does not hold" : render_criterion(f)),
     "true: " * (is_blank(t) ? "yes, the statement holds" : render_criterion(t))]
end

"""
    build_prefix(tok, q, head_max_len=192) -> (ids, markers)

Question-only prefix `[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]`,
before state tokens and final truncation. Marker positions are 0-based.
"""
function build_prefix(tok::Tokenizer, q::Question, head_max_len::Integer=192)
    mask = tok.mask_token
    opts = render_options(q)
    ins = replace(q.ins, mask => " ")
    head_ids = tok("$(q.t) question: $ins")
    opt_ids = [[tok.mask_token_id; first(tok(" " * replace(o, mask => " ")), 48)] for o in opts]
    opt_budget = head_max_len - sum(length, opt_ids)
    if opt_budget < 16
        per = max(4, (head_max_len - 16) ÷ max(1, length(opt_ids)))
        opt_ids = [first(o, per) for o in opt_ids]
        opt_budget = head_max_len - sum(length, opt_ids)
    end
    ids = [tok.cls_token_id; first(head_ids, max(8, opt_budget)); tok.sep_token_id]
    markers = Int[]
    for o in opt_ids
        push!(markers, length(ids))
        append!(ids, o)
    end
    push!(ids, tok.sep_token_id)
    ids, markers
end

"""
    build_sequence(tok, state, q; max_len=512, head_max_len=192) -> (ids, markers)

`[CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP] state [SEP]`, truncated to `max_len`.
"""
function build_sequence(tok::Tokenizer, state, q::Question; max_len::Integer=512, head_max_len::Integer=192)
    ids, markers = build_prefix(tok, q, head_max_len)
    room = max(0, max_len - length(ids) - 1)
    st = tok(replace(serialize_state(state), tok.mask_token => " "))
    ids = [ids; first(st, room); tok.sep_token_id]
    first(ids, max_len), filter(<(max_len), markers)
end

"""Normalized Shannon-entropy confidence `1 - H(p) / log(k)`, in Float32 as NumPy."""
function confidence_from_probs(p::AbstractVector{Float32}, k::Integer)
    k < 2 && return 1.0
    p = p[1:k]
    ent = -sum(p .* log.(clamp.(p, 1.0f-12, 1.0f0)))
    Float64(clamp(1.0f0 - ent / Float32(log(k)), 0.0f0, 1.0f0))
end

function temp_bucket(qtype::Integer, k::Integer)
    size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+"
    "$(QTYPE_NAMES[qtype]):$size"
end

# A fitted temperature below 1 sharpens logits instead of softening them; the shipped
# `choice:11+` bucket (0.1006) would report a coin flip as near-certainty. Clamp, as upstream.
const TEMP_MIN = 0.5
const TEMP_MAX = 5.0

function clamp_temperature(t; lo=TEMP_MIN, hi=TEMP_MAX)
    t isa Real || return 1.0
    t = Float64(t)
    isfinite(t) ? clamp(t, lo, hi) : 1.0
end
