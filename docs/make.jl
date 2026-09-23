using Documenter
using Laya

DocMeta.setdocmeta!(Laya, :DocTestSetup, :(using Laya); recursive=true)

makedocs(;
    modules=[Laya],
    sitename="Laya.jl",
    authors="Satoshi Terasaki and contributors",
    repo=Remotes.GitHub("AtelierArith", "Laya.jl"),
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://atelierarith.github.io/Laya.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Backends" => "backends.md",
        "Checkpoints" => "checkpoints.md",
        "Accuracy and speed" => "performance.md",
        "Development" => "development.md",
        "API reference" => "api.md",
    ],
    checkdocs=:exports,
)

deploydocs(; repo="github.com/AtelierArith/Laya.jl", devbranch="main", push_preview=false)
