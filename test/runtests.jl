using GaussianStatesQPE
using Test
using LinearAlgebra
using PyCall

@pyimport importlib.machinery as machinery

loader = machinery.SourceFileLoader("CramerRao","D:\\workspace\\QuanEstimation\\quanestimation\\AsymptoticBound\\CramerRao.py")
CR = loader[:load_module]("CramerRao")

@testset "GaussianStatesQPE.jl" begin
    n = 5
    a = rand(2n,2n)
    A = a'a
    S, c=Williamson_form(A)
    Cd = repeat(c',2)[:]|>diagm
    @test S*Cd*transpose(S) ≈  A
    
    # test QFIM_Gauss for single mode 
    para_num = 3
    m = 5

    for i in 1:1000
        R= rand(2*m)
        dR = [rand(2*m) for _ in 1:para_num]
        C = 5*rand(2*m,2*m)|>x->x'x
        dC =  [5*rand(2*m,2*m) |>x->x'x for _ in 1:para_num]
        D = (C+R*R')
        dD = [dC[i] + dR[i]*R' + R*dR[i]' for i in 1:para_num]
        C = [(D[i,j] + D[j,i])/2 - R[i]R[j] for i in 1:2m, j in 1:2m]
        dC = [[(dD[k][i,j] + dD[k][j,i])/2 - dR[k][i]R[j] - R[i]dR[k][j] for i in 1:2m, j in 1:2m] for k in 1:para_num]

        if isposdef(C) && all(isposdef.(dC))
            # @test QFIM_Gauss(R,dR,D,dD) ≈ QFIM_Gauss_single_mode(R,dR,D,dD)
            # @test QFIM_Gauss(R,dR,D,dD) ≈ CR[:QFIM_Gauss](R,dR,D,dD)
            @test (QFIM_Gauss(R,dR,D,dD)./CR[:QFIM_Gauss](R,dR,D,dD) - ones(para_num,para_num))/para_num^2|>norm <= 1
        end
    end

end
