# Regenerate ../src/LibMLX.jl from the installed mlx-c headers:
#   julia --project=. generator.jl
using Clang.Generators

cd(@__DIR__)
const INCLUDE = normpath(joinpath(@__DIR__, "..", "..", "deps", "usr", "include"))
const HEADER = joinpath(INCLUDE, "mlx", "c", "mlx.h")

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()
push!(args, "-I$INCLUDE")
ctx = create_context([HEADER], args, options)
build!(ctx)
