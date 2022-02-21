using GaussianStatesQPE
using Test
using LinearAlgebra

@testset "GaussianStatesQPE.jl" begin
    n = 5
    a = rand(2n,2n)
    A = a'a
    S, c=Williamson_form(A)
    Cd = repeat(c',2)[:]|>diagm
    @test S*Cd*transpose(S) ≈  A
    
    # test QFIM_Gauss for single mode 
    para_num = 3

    for i in 1:10000
        R= rand(2)
        dR = [rand(2) for _ in 1:para_num]
        C = 5*rand(2,2)|>x->x'x
        dC =  [5*rand(2,2)|>x->x'x for _ in 1:para_num]
        D = (C+R*R')
        dD = [dC[i] + dR[i]*R' + R*dR[i]' for i in 1:para_num]
        C = [(D[i,j] + D[j,i])/2 - R[i]R[j] for i in 1:2, j in 1:2]
        dC = [[(dD[k][i,j] + dD[k][j,i])/2 - dR[k][i]R[j] - R[i]dR[k][j] for i in 1:2, j in 1:2] for k in 1:para_num]

        if isposdef(C) && all(isposdef.(dC))
            @test QFIM_Gauss(R,dR,D,dD) ≈ QFIM_Gauss_single_mode(R,dR,D,dD)
        end
    end
end
