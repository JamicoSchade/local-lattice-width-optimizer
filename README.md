This repository contains [Julia](https://github.com/JuliaLang/julia) code to verify that the 4- and 5-dimensional lattice-free simplices given in the paper

[Lattice-free simplices with lattice width 2d − o(d)](https://arxiv.org/abs/2111.08483)

are local maximizers of the lattice width, see Theorem 2 in the paper. All claims are verified symbolically using the [Julia](https://github.com/JuliaLang/julia) packages [DynamicPolynomials](https://github.com/JuliaAlgebra/DynamicPolynomials.jl) and [AlgebraicNumbers](https://github.com/anj1/AlgebraicNumbers.jl). The code can be run via:

```julia
include("main.jl")
verify_local_optimum(4) # ~ 2 minutes
verify_local_optimum(5) # ~ 6 minutes
```
