module GaussianStatesQPE
using LinearAlgebra

const σ_x = [0.0 1.0; 1.0 0.0im]
const σ_y = [0.0 -1.0im; 1.0im 0.0]
const σ_z = [1.0 0.0im; 0.0 -1.0]
const σs = [σ_x, σ_y, σ_z]

destroy(N) = diagm(1 => [1/sqrt(n) for n in 1:N-1])

bases(dim; T=ComplexF64) = [e for e in I(dim).|>T|>eachrow]

function Williamson_form(A::AbstractMatrix)
    n = size(A)[1]//2 |>Int
    J =  kron([0 1.; -1 0],I(n))
    A_sqrt = sqrt(A)
    B = A_sqrt*J*A_sqrt
    P = one(A)|>x->[x[:,1:2:2n-1] x[:,2:2:2n]]
    T, Q, vals = schur(B)
    c = vals[1:2:2n-1].|>imag
    D = c|>diagm|>x->x^(-0.5)
    S = J*A_sqrt*Q*P*kron([0 1.; -1 0],-D)*transpose(P)|>transpose|> inv
    return S, c
end

# function Cov(Ri::T, Rj::T, ρ::T) where T<:AbstractMatrix 
#     tr(ρ*(Ri*Rj+Rj*Ri))/2 - tr(ρ*Ri)*tr(ρ*Rj)
# end

# Cov(R::AbstractVector, ρ::AbstractMatrix) = [Cov(Ri,Rj,ρ) for Ri in R, Rj in R]

# function filterZeros(x::AbstractVecOrMat{T}) where T<:Number
    # [x+1≈1 ? zero(T) : x for x in x]
# end

const a_Gauss = [im*σ_y,σ_z,σ_x|>one, σ_x]

function A_Gauss(m::Int)
    e = bases(m)
    s = e.*e'
    a_Gauss .|> x -> [kron(s, x)/sqrt(2) for s in s]
end

function G_Gauss(S::M, dC::VM, c::V) where {M<:AbstractMatrix, V,VM<:AbstractVector}
    para_num = length(dC)
    m = size(S)[1]//2 |>Int
    As = A_Gauss(m)
    gs =  [[[inv(S)*∂ₓC*inv(transpose(S))*a'|>tr for a in A] for A in As] for ∂ₓC in dC]
    #[[inv(S)*∂ₓC*inv(transpose(S))*As[l][j,k]|>tr for j in 1:m, k in 1:m] for l in 1:4]
    G = [zero(S) for _ in 1:para_num]
    
    for i in 1:para_num
        for j in 1:m
            for k in 1:m 
                for l in 1:4
                    G[i]+=gs[i][l][j,k]/(4*c[j]c[k]+(-1)^l)*inv(transpose(S))*As[l][j,k]*inv(S)
                end
            end 
        end
    end
    G
end

# function R_quad(m, num_cutoff)
#     a = destroy(num_cutoff)
#     repeat([a+a',-im*(a-a')], m)./sqrt(2)
# end 

# function SLD_Gauss(R̄::V, ∂ₓR̄::V, C::T, ∂ₓC::T, num_cutoff::Int=1000) where {T<:AbstractMatrix, V<:AbstractVector}
#     m = length(R̄)//2 |>Int
#     R = R_quad(m, num_cutoff)
#     S, cs = Williamson_form(C)
#     G = G_Gauss(S, ∂ₓC, cs)
#     L1 = inv(C)*∂ₓR̄ - 2*G*R̄
#     L0 = transpose(R̄)*G*R̄ - transpose(∂ₓR̄)*inv(C)*R̄ - tr(G*C)
#     L = L0*I(num_cutoff) + transpose(L1)*R + transpose(R)*G*R
# end


function QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM <:AbstractVecOrMat}
    para_num = length(dR̄)
    quad_num = length(R̄)
    C = [(D[i,j] + D[j,i])/2 - R̄[i]R̄[j] for i in 1:quad_num, j in 1:quad_num]
    dC = [[(dD[k][i,j] + dD[k][j,i])/2 - dR̄[k][i]R̄[j] - R̄[i]dR̄[k][j] for i in 1:quad_num, j in 1:quad_num] for k in 1:para_num]
    S, cs = Williamson_form(C)
    Gs = G_Gauss(S, dC, cs)    
    F = [tr(Gs[i]*dC[j])+transpose(dR̄[i])*inv(C)*dR̄[j] for i in 1:para_num, j in 1:para_num]
    F|>real
end

function QFIM_Gauss(dR̄::VV, C::M, dC::VM) where {V,VV,M,VM <:AbstractVecOrMat}
    para_num = length(dR̄)
    S, cs = Williamson_form(C)
    Gs = G_Gauss(S, dC, cs)
    F = [tr(Gs[i]*dC[j])+transpose(dR̄[i])*inv(C)*dR̄[j] for i in 1:para_num, j in 1:para_num]
    F|>real
end

function QFIM_Gauss_single_mode(R̄::V, dR̄::VV, D::M, dD::VM) where 
                                                    {V,VV,M,VM <:AbstractVecOrMat}
    Ω = im*σ_y
    para_num = length(dR̄)
    C = [(D[i,j] + D[j,i])/2 - R̄[i]R̄[j] for i in 1:2, j in 1:2]
    dC = [[(dD[k][i,j] + dD[k][j,i])/2 - dR̄[k][i]R̄[j] - R̄[i]dR̄[k][j] for i in 1:2, j in 1:2] for k in 1:para_num]
    c = sqrt(det(C))
    dJ = [dC/(4c^2-1)-4*c^2*tr(inv(C)*dC)*C/(4c^2-1)^2 for dC in dC]
    Gs = [(4c^2-1)/(4c^2+1)*Ω*dJ*Ω for dJ in dJ]
    F = [tr(Gs[i]*dC[j])+transpose(dR̄[i])*inv(C)*dR̄[j] for i in 1:para_num, j in 1:para_num]
    F|>real
end 

export Williamson_form
export QFIM_Gauss, QFIM_Gauss_single_mode

end
