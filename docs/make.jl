using FeatherScopePrePro
using Documenter

DocMeta.setdocmeta!(FeatherScopePrePro, :DocTestSetup, :(using FeatherScopePrePro); recursive=true)

makedocs(;
    modules=[FeatherScopePrePro],
    authors="Galen Lynch <galen@galenlynch.com>",
    sitename="FeatherScopePrePro.jl",
    format=Documenter.HTML(;
        canonical="https://galenlynch.github.io/FeatherScopePrePro.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/galenlynch/FeatherScopePrePro.jl",
    devbranch="main",
)
