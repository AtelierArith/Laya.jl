module LayaAppleAccelerateExt

using AppleAccelerate: AppleAccelerate
using Laya: Laya, AccelerateBackend

# Same model as `CPUBackend`; only the BLAS behind its matrix products changes (process-wide).
function Laya.load_backend_model(::AccelerateBackend, dir::AbstractString, ::Type{T}) where {T}
    Laya.uses_accelerate() || AppleAccelerate.load_accelerate()
    Laya.uses_accelerate() || error("AppleAccelerate could not forward BLAS to Accelerate (needs macOS 13.4 or later)")
    first(Laya.load_model(dir; dtype=T))
end

end
