"""
    IndependentMOKernel(k::Kernel)

Kernel for multiple independent outputs with kernel `k` each.

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel ``\\widetilde{k}``
for independent outputs with kernel ``k`` each is defined as
```math
\\widetilde{k}\\big((x, p_x), (x', p_{x'})\\big) = \\begin{cases}
    k(x, x') & \\text{if } p_x = p_{x'}, \\\\
    0 & \\text{otherwise}.
\\end{cases}
```
Mathematically, it is equivalent to a matrix-valued kernel defined as
```math
\\widetilde{K}(x, x') = \\mathrm{diag}\\big(k(x, x'), \\ldots, k(x, x')\\big) \\in \\mathbb{R}^{m \\times m},
```
where ``m`` is the number of outputs.
"""
struct IndependentMOKernel{Tkernel<:Kernel} <: MOKernel
    kernel::Tkernel
end

function (κ::IndependentMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    return κ.kernel(x, y) * (px == py)
end

function _kernelmatrixkronhelper(::MOInputIsotopicByFeatures, Ktmp, B)
    return kron(Ktmp, B)
end

function _kernelmatrixkronhelper(::MOInputIsotopicByOutputs, Ktmp, B)
    return kron(B, Ktmp)
end

function kernelmatrix(
    k::IndependentMOKernel, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Ktmp)
    return _kernelmatrixkronhelper(x, Ktmp, Eye{mtype}(x.out_dim))
end

if VERSION >= v"1.6"
    _kernelmatrixkronhelper!(K, ::MOInputIsotopicByFeatures, K, B) = kron!(K, K, B)

    _kernelmatrixkronhelper!(K, ::MOInputIsotopicByOutputs, K, B) = kron!(K, B, K)

    function kernelmatrix!(
        K::AbstractMatrix, k::IndependentMOKernel, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}
        @assert x.out_dim == y.out_dim
        Ktmp = kernelmatrix(k.kernel, x.x, y.x)
        mtype = eltype(Ktmp)
        return _kernelmatrixkronhelper!(K, Ktmp, Matrix{mtype}(I, x.out_dim, x.out_dim), x)
    end
end

function Base.show(io::IO, k::IndependentMOKernel)
    return print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
