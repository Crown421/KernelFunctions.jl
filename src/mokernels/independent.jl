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

function _kernelmatrix_kron_helper(::MOInputIsotopicByFeatures, Kfeatures, B)
    return kron(Kfeatures, B)
end

function _kernelmatrix_kron_helper(::MOInputIsotopicByOutputs, Kfeatures, B)
    return kron(B, Kfeatures)
end

function kernelmatrix(
    k::IndependentMOKernel, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    @assert x.out_dim == y.out_dim
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Kfeatures)
    return _kernelmatrix_kron_helper(x, Kfeatures, Eye{mtype}(x.out_dim))
end

if VERSION >= v"1.6"
    function _kernelmatrix_kron_helper!(K, ::MOInputIsotopicByFeatures, Kfeatures, B)
        return kron!(K, Kfeatures, B)
    end

    function _kernelmatrix_kron_helper!(K, ::MOInputIsotopicByOutputs, Kfeatures, B)
        return kron!(K, B, Kfeatures)
    end

    function kernelmatrix!(
        K::AbstractMatrix, k::IndependentMOKernel, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}
        @assert x.out_dim == y.out_dim
        Ktmp = kernelmatrix(k.kernel, x.x, y.x)
        mtype = eltype(Ktmp)
        return _kernelmatrix_kron_helper!(
            K, x, Ktmp, Matrix{mtype}(I, x.out_dim, x.out_dim)
        )
    end
end

function _kronkernelmatrix(
    Ktmp, B, ::MOInputIsotopicByOutputs, ::ExplicitKroneckerKernelMatrix
)
    return kron(B, Ktmp)
end

function kernelmatrix(
    k::IndependentMOKernel, x::MOI, y::MOI; matrixtype::MT=ExplicitKroneckerKernelMatrix()
) where {MOI<:AbstractMOInput,MT<:KroneckerKernelMatrix}
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Ktmp)
    return _kronkernelmatrix(Ktmp, Matrix{mtype}(I, x.out_dim, x.out_dim), x, matrixtype)
end

function kernelmatrix(
    k::IndependentMOKernel, x::MOI; matrixtype::MT=ExplicitKroneckerKernelMatrix()
) where {MOI<:AbstractMOInput,MT<:KroneckerKernelMatrix}
    return kernelmatrix(k, x, x; matrixtype)
end

function _kronkernelmatrix!(K, Ktmp, B, ::MOInputIsotopicByFeatures)
    return kron!(K, Ktmp, B)
end

function _kronkernelmatrix!(K, Ktmp, B, ::MOInputIsotopicByOutputs)
    return kron!(K, B, Ktmp)
end

function kernelmatrix!(
    K::AbstractMatrix,
    k::IndependentMOKernel,
    x::MOInputIsotopicByFeatures,
    y::MOInputIsotopicByFeatures,
)
    @assert x.out_dim == y.out_dim
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    mtype = eltype(Ktmp)
    return _kronkernelmatrix!(K, Ktmp, Matrix{mtype}(I, x.out_dim, x.out_dim), x)
end

function Base.show(io::IO, k::IndependentMOKernel)
    return print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
