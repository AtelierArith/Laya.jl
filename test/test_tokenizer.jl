using Random

const TOKENIZER_CORPUS = [
    "", " ", "  ", "hello", " hello", "hello ", "Hello, world!", "I was billed twice. Please refund the duplicate.",
    "don't won't it's they're I've we'll he'd I'm", "It'S DON'T", "12345 3.14159 1,000,000 ٣٤٥",
    "a  b   c    d", "tabs\tand\nnewlines\r\n\n\n end", "trailing spaces   ", "   leading",
    "日本語のテキストです。", "发票被重复扣款，请退款。", "한국어 텍스트", "العربية نص", "हिन्दी पाठ", "Ελληνικά", "русский текст",
    "emoji 👍🏽 👨‍👩‍👧 🇯🇵", "café naïve é Å Å", "ﬁ ligature ① ＡＢＣ", "　ideographic　space",
    "[CLS] [SEP] [PAD] [MASK] text [MASK]", "x[MASK]y", "  [MASK]  ", "<bos> <eos> <mask> <pad> <unk>", "a <mask> b",
    "|||IP_ADDRESS||| <|endoftext|> <|padding|>", "{\"message\": \"refund\", \"amount\": 12.5, \"tags\": [\"a\", \"b\"]}",
    "choice question: Which team should handle this request?", " billing: invoices, payments, refunds",
    "level 0: not urgent", "false: no, the statement does not hold", " nbsp thin​zero-width",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛", "\x7f\x01control", repeat("long ", 200),
]

const CHAR_POOL = collect("aAbBzZ09 _-.,;:!?'\"()[]{}<>/\\@#\$%^&*+=~`|\t\n  éüßçñ日本語中文한국어ÄÖÜ😀👍🏽́　 АБВ")
random_text(rng) = String(rand(rng, CHAR_POOL, rand(rng, 0:40)))

function check_tokenizer(repo)
    dir = cached_snapshot(repo)
    if dir === nothing
        @info "Skipping: $repo is not in the Hugging Face cache"
        return
    end
    ref = R.load_tokenizer(dir)
    tok = Tokenizer(joinpath(dir, "tokenizer"))
    sp = R.special_tokens(ref)
    for name in ("cls_token", "sep_token", "pad_token", "mask_token")
        @test getfield(tok, Symbol(name)) == sp[name].token
        @test getfield(tok, Symbol(name * "_id")) == sp[name].id
    end
    rng = Xoshiro(20260923)
    texts = [TOKENIZER_CORPUS; [random_text(rng) for _ in 1:2000]]
    mismatches = [t for t in texts if tok(t) != R.tokenize(ref, t)]
    isempty(mismatches) || @info "$repo tokenizer mismatches" first(mismatches, 10)
    @test isempty(mismatches)
end

@testset "tokenizer: $repo" for repo in ("aac6fef/laya-mlx", "aac6fef/laya-multilingual-mlx")
    check_tokenizer(repo)
end

@testset "tokenizer: WordLevel (tiny)" begin
    dir = R.tiny_checkpoint(joinpath(mktempdir(), "checkpoint"))
    ref = R.load_tokenizer(dir)
    tok = Tokenizer(joinpath(dir, "tokenizer"))
    for t in ["hello", "hello world", "hello, [MASK] hello!", ""]
        @test tok(t) == R.tokenize(ref, t)
    end
end
