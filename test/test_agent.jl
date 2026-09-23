using PythonCall
using Laya: JSON, py_float, py_json, prepare, collate

@testset "Python json.dumps compatibility" begin
    pyjson = pyimport("json")
    floats = [0.0, -0.0, 1.0, 3.0, 12.5, 0.1, 1e-4, 1.5e-4, 1e-5, 0.00012345, 123456.789, 1e15, 1e16, 1.5e16,
              2.5e-300, 1.7976931348623157e308, -2.0, 100.0, 0.30000000000000004, 5e-324, 9.999999999999999e15]
    for x in floats
        @test py_float(x) == pyconvert(String, pybuiltins.repr(x))
    end
    values = Any[nothing, true, false, 42, -7, 3.25, "plain", "quote\" back\\ nl\n tab\t ctl\x01 del\x7f é 日本 😀",
                 [1, 2.0, "x", nothing], JSON.parse("{\"b\": 1, \"a\": [true, {\"c\": 1e-7}]}")]
    for v in values, ascii in (false, true)
        expected = pyconvert(String, pyjson.dumps(pyjson.loads(JSON.json(v)); ensure_ascii=ascii))
        @test py_json(v; ascii) == expected
    end
end

const AGENT_STATES = Any[
    "I was billed twice. Please refund the duplicate today.",
    "Our production cluster is down and customers cannot log in!!!",
    JSON.parse("""{"message": "発票被重複扣款，請退款。", "amount": 12.5, "items": [1, 2.0, null], "vip": true}"""),
    [JSON.parse("""{"role": "user", "content": "Can I buy 50 more seats?"}"""), JSON.parse("""{"role": "assistant", "content": "Sure."}""")],
    repeat("This is a very long complaint about a double charge. ", 120),     # truncated to max_len
    "Text containing [MASK] and <mask> tokens.",
]

const AGENT_QUESTIONS = JSON.parse("""
{
  "department": {"type": "choice", "instructions": "Which team should handle this request?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages", "sales": "new purchases"}},
  "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["not urgent", "soon", "critical"]},
  "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
  "refund2": {"type": "noul", "instructions": "Refund requested?", "criteria": {"false": "", "true": {"desc": "asks for money", "w": 0.5}}},
  "labels": {"type": "choice", "instructions": {"task": "pick one", "lang": "日本語"}, "criteria": ["a", "b"]},
  "single": {"type": "choice", "instructions": "Only option", "criteria": ["only"]},
  "rubric": {"type": "score", "instructions": "Quality", "criteria": [{"min": 0}, 1, 2.5, "best"]},
  "many": {"type": "choice", "instructions": "Pick the intent among many options",
           "criteria": [$(join(("\"intent_$i with a fairly long descriptive label number $i\"" for i in 1:14), ", "))]}
}
""")

function check_agent(dir; dtype=Float32)
    ref = R.load(dir; dtype="float32", device="gpu")
    agent = Laya.load(dir; dtype, batch_size=4)
    for state in AGENT_STATES
        items, _ = prepare(agent, state, AGENT_QUESTIONS)
        ritems = R.prepare(ref, state, AGENT_QUESTIONS)
        @test [(i.ids, i.markers, i.qtype) for i in items] == [(i.ids, i.markers, i.qtype) for i in ritems]
        @test collate(items, agent.tok.pad_token_id) == R.collate(ref, ritems)

        actual, expected = predict(agent, state, AGENT_QUESTIONS), R.predict(ref, state, AGENT_QUESTIONS)
        @test actual["usage"] == expected["usage"]
        @test collect(keys(actual["answers"])) == collect(keys(expected["answers"]))
        for (qid, e) in expected["answers"]
            a = actual["answers"][qid]
            @test Set(keys(a)) == Set(keys(e))
            @test a["type"] == e["type"]
            haskey(e, "choice") && @test a["choice"] == e["choice"]
            haskey(e, "legend") && @test JSON.json(a["legend"]) == JSON.json(e["legend"])
            numbers = [(a["confidence"], e["confidence"]), (a["action"]["act_probability"], e["action"]["act_probability"])]
            haskey(e, "score") && push!(numbers, (a["score"], e["score"]))
            haskey(e, "noul") && push!(numbers, (a["noul"], e["noul"]))
            haskey(e, "probabilities") && append!(numbers, [(a["probabilities"][k], v) for (k, v) in e["probabilities"]])
            # Values are rounded to 4 decimals; float32 noise can flip the last digit.
            @test all(((x, y),) -> abs(x - y) <= 1.5e-4, numbers)
        end
    end
end

@testset "Agent: $repo" for repo in TEST_REPOS
    dir = cached_snapshot(repo)
    dir === nothing ? @info("Skipping: $repo not cached") : check_agent(dir)
    free_models()
end
