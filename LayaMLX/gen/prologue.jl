const libmlxc = get(ENV, "LAYAMLX_LIBMLXC",
    normpath(joinpath(@__DIR__, "..", "..", "deps", "usr", "lib", "libmlxc.dylib")))
