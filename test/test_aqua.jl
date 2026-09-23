# Code-quality checks (method ambiguities, unbound type parameters, undefined exports,
# stale dependencies, compat bounds, type piracy, persistent tasks). Loaded extensions
# (AppleAccelerate, Metal) are checked along with Laya.
using Aqua

@testset "Aqua" begin
    Aqua.test_all(Laya)
end
