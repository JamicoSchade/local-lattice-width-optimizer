using LinearAlgebra
using DynamicPolynomials
using AlgebraicNumbers

#Fix compatibility issues and add base functionalities to algebraic numbers

function Base.isreal(x::AlgebraicNumber)
    return x == real(x)
end

function Base.isless(x::AlgebraicNumber, y::AlgebraicNumber)
    @assert isreal(x) && isreal(y)

    return sign(y - x) == 1
end

AlgebraicNumbers.AlgebraicNumber{BigInt, BigFloat}(x::Bool) = AlgebraicNumber(x)

function Base.promote_op(::typeof(LinearAlgebra.matprod), ::Type{T}, ::Type{T}) where T <: AlgebraicNumber
    return T
end

function Base.promote_op(
        ::typeof(LinearAlgebra.matprod),
        ::Type{Polynomial{true, AlgebraicNumber{BigInt, BigFloat}}},
        ::Type{Polynomial{true, AlgebraicNumber{BigInt, BigFloat}}}
    )
    return Polynomial{true, AlgebraicNumber{BigInt, BigFloat}}
end

sym(x) = AlgebraicNumber(x)

#Calculates the adjugate of A, which equals det(A) * inv(A)

function adjugate(A)
    @assert size(A, 1) == size(A, 2)

    n = size(A, 1)
    result = zeros(eltype(A), n, n)

    for i in 1:n, j in 1:n
        result[i,j] = (-1)^(i+j)*det(A[[(1:i-1)...,(i+1:n)...], [(1:j-1)...,(j+1:n)...]])
    end

    return collect(transpose(result))
end

#Calculates reduced row echelon form of A, only used for nullspace
#Modified from https://github.com/blegat/RowEchelon.jl 

function rref_with_pivots!(A::Matrix{T}) where T
    nr, nc = size(A)
    pivots = Vector{Int64}()
    i = j = 1
    while i <= nr && j <= nc
        (m, mi) = findmax(abs.(A[i:nr,j]))
        mi = mi+i - 1
        if m == 0
            j += 1
        else
            for k=j:nc
                A[i, k], A[mi, k] = A[mi, k], A[i, k]
            end
            d = A[i,j]
            for k = j:nc
                A[i,k] /= d
            end
            for k = 1:nr
                if k != i
                    d = A[k,j]
                    for l = j:nc
                        A[k,l] -= d*A[i,l]
                    end
                end
            end
            append!(pivots,j)
            i += 1
            j += 1
        end
    end
    return A, pivots
end

rref_with_pivots(A::Matrix{T}) where {T} = rref_with_pivots!(copy(A))

#Calculates the right nullspace of A

function nullspace(A::Matrix)
    R,pivot_cols = rref_with_pivots(A)
    m,n = size(A)
    r = length(pivot_cols)
    nopiv = collect(1:n)
    deleteat!(nopiv, pivot_cols)
    Z = zeros(eltype(R), n, n-r)
    if n > r
        Z[nopiv,:] = diagm(0 => ones(eltype(R), n-r))
        if r > 0
            Z[pivot_cols,:] = -R[1:r,nopiv]
        end
    end
    return Z
end

#Calculates the Hessian matrix of the function f with respect to the variables x

function hessian(f, x)
    result = zeros(typeof(f), length(x), length(x))
    for i in 1:length(x), j in 1:length(x)
        result[i, j] = differentiate(differentiate(f, x[i]), x[j])
    end
    return result
end

#Evaluates the Hessian matrix at x = 0

function hessian_at_zero(f, x)
    result = zeros(eltype(f.a), length(x), length(x))
    H = hessian(f, x)
    for i in 1:length(x), j in 1:length(x)
        if length(H[i, j].a) == 0
            result[i, j] = zero(eltype(f.a))
        else
            result[i, j] = H[i, j](x[:] => zeros(Int, length(x)))
        end
    end
    return result
end

#Verifies local optimality for Delta_4 and Delta_5

function verify_local_optimum(d)
    
    println("Verifying that Delta_" * string(d) * " is a local maximizer for lattice width.")
    print("Setting entries of vertices... ")

    #Set the entries of the vertices, depending on d
    if d == 4
        q = [
            sym(1)/5 * (7 - 2 * sqrt(sym(5)) + 2 * sqrt(10 + 2 * sqrt(sym(5)))),
            sym(1)/5 * (-3 + 4 * sqrt(sym(5)) - 4 * sqrt(5 - 2 * sqrt(sym(5)))),
            sym(1)/5 * (7 - 4 * sqrt(sym(5)) + 6 * sqrt(5 - 2 * sqrt(sym(5)))),
            sym(1)/5 * (-3 + 4 * sqrt(sym(5)) - 4 * sqrt(5 - 2 * sqrt(sym(5)))),
            sym(1)/5 * (-3 - 2 * sqrt(sym(5)) - 2 * sqrt(5 + 2 * sqrt(sym(5)))),
        ]

        lattice_width = 2 + 2 * sqrt(1 + sym(2)/sqrt(sym(5)))
    elseif d == 5
        q = [
            sym(1)/18 * (57 - 7 * sqrt(sym(3))),
            sym(1)/3 * (4 * sqrt(sym(3)) - 5),
            sym(1)/18 * (27 - 11 * sqrt(sym(3))),
            sym(1)/18 * (27 - 11 * sqrt(sym(3))),
            sym(1)/3 * (4 * sqrt(sym(3)) - 5),
            sym(1)/18 * (-33 - 19 * sqrt(sym(3))),
        ]

        lattice_width = 5 + 2 * sqrt(sym(3))/3
    end

    #Fill v with values to create the vertices of Delta_d
    v = zeros(eltype(q), d+1, d)

    for i = 0:d
        v_temp = circshift(q, i)
        v[i+1, :] = v_temp[1:d]
    end

    #Set the lattice width-attaining lattice directions and corresponding
    #maximizing and minimizing vertices, depending on d
    if d == 4
        a = [
            v[1, :] - v[2, :],
            v[2, :] - v[3, :],
            v[3, :] - v[4, :],
            v[4, :] - v[5, :],
            v[1, :] - v[3, :],
            v[2, :] - v[4, :],
            v[3, :] - v[5, :],
            v[1, :] - v[4, :],
            v[2, :] - v[5, :],
            v[1, :] - v[5, :],
        ]

        u = [
            1 0 0 0;
            0 1 0 0;
            0 0 1 0;
            0 0 0 1;
            1 1 0 0;
            0 1 1 0;
            0 0 1 1;
            1 1 1 0;
            0 1 1 1;
            1 1 1 1;
        ]
    elseif d == 5
        a = [
            v[1, :] - v[2, :],
            v[2, :] - v[3, :],
            v[3, :] - v[4, :],
            v[4, :] - v[5, :],
            v[5, :] - v[6, :],
            v[1, :] - v[3, :],
            v[2, :] - v[4, :],
            v[3, :] - v[5, :],
            v[4, :] - v[6, :],
            v[1, :] - v[4, :],
            v[2, :] - v[5, :],
            v[3, :] - v[6, :],
            v[1, :] - v[5, :],
            v[2, :] - v[6, :],
            v[1, :] - v[6, :],
        ]

        u = [
            1 0 0 0 0;
            0 1 0 0 0;
            0 0 1 0 0;
            0 0 0 1 0;
            0 0 0 0 1;
            1 1 0 0 0;
            0 1 1 0 0;
            0 0 1 1 0;
            0 0 0 1 1;
            1 1 1 0 0;
            0 1 1 1 0;
            0 0 1 1 1;
            1 1 1 1 0;
            0 1 1 1 1;
            1 1 1 1 1;
        ]
    end
    println("Done")
    print("Creating perturbation vectors and matrix M... ")

    #Define t as matrix of variables
    @polyvar t[1:(d + 1), 1:(d - 1)]

    #Multiplication with poly_one will be used throughout the function to fix compatibility issues
    number_type = AlgebraicNumber{BigInt, BigFloat}
    poly_type = Polynomial{true, number_type}
    poly_one = one(poly_type)
    poly_zero = zero(poly_type)

    #Set T depending on t such that p_i + T_i still lies in the same facet of Delta_d as p_i
    T = zeros(poly_type, d + 1, d)
    shifted_vertices = zeros(eltype(v), d - 1, d)
    for i = 1:d + 1
        if i != 1
            for j = 2:i - 1
                shifted_vertices[j - 1, :] = v[j, :] - v[1, :]
            end
            for j = i + 1:d + 1
                shifted_vertices[j - 2, :] = v[j, :] - v[1, :]
            end
        else
            for j = 3:d + 1
                shifted_vertices[j - 2, :] = v[j, :] - v[2, :]
            end
        end
        shifted_vertices_transposed = transpose(shifted_vertices)
        linear_combination = inv(shifted_vertices_transposed[[1, 3:d...], :])

        L = shifted_vertices_transposed * linear_combination
        T[i, :] = (poly_one*L) * (poly_one*t[i, :])
    end

    #The identity matrix
    eye = [i == j ? poly_one : poly_zero for i in 1:d+1, j in 1:d+1]

    p = eye[:, 2:end] + T

    M = zeros(poly_type, d, d)
    for i = 1:d
        M[i, :] = p[i + 1, :] - p[1, :]
    end

    println("Done")
    print("Calculating the adjugate of M... ")
    M_adj = adjugate(M)
    println("Done")

    print("Calculating the determinant of M... ")
    M_det = det(M)
    println("Done")

    print("Calculating the functions h_i and their gradients at zero... ")
    h = [(((poly_one*u[i:i, :]) * collect(transpose(M_adj))) * (poly_one*a[i]))[1] - lattice_width * M_det for i in 1:(d * (d + 1))÷2]

    gradients = zeros(poly_type, length(h), length(t))
    for i = 1:length(h), j in 1:length(t)
        gradients[i, j] = differentiate(h[i], t[j])
    end

    #Evaluate the gradients of the h_i in t = 0
    gradients_eval = zeros(number_type, size(gradients))
    for i = 1:length(h), j in 1:length(t)
        gradients_eval[i, j] = gradients[i,j](t[:] => zeros(Int, length(t)))
    end
    println("Done")

    print("Calculating a linear dependency of these gradients... ")
    lambda = nullspace(collect(transpose(gradients_eval)))
    println("Done")

    print("Calculating sum_(i = 1)^(d * (d + 1)/2) lambda_i * h_i and isolating its quadratic part Q... ")
    g = sum(lambda[i] * h[i] for i = 1:length(h)) 

    #Calculating the quadratic part Q of g using the Hessian of g
    Q = (poly_one/2) * collect(transpose(t[:])) * (poly_one * hessian_at_zero(g, t)) * (poly_one * t[:])
    println("Done")

    print("Calculating a basis of the right nullspace W of the gradients at zero... ")
    W_basis = nullspace(gradients_eval)
    println("Done")

    print("Expressing Q in terms of a basis of W... ")
    @polyvar w[1:length(t) - length(h) + 1]

    W_representation = (poly_one * W_basis) * (poly_one * w)

    #Expressing Q in terms of a basis of W
    Q = subs(Q, t[:] => W_representation)[1]
    println("Done")

    print("Calculating the hessian matrix of Q restricted to W in zero... ")
    hessian_matrix_evaluated = hessian_at_zero(Q, w)
    println("Done")

    println("If the following values are all positive, it is confirmed that this matrix is negative definite... ")
    #A matrix is negative definite if its negative is positive definite
    negative_hessian_matrix_evaluated = - hessian_matrix_evaluated

    #Using Sylvester's Criterion:
    #A matrix is positive definite, if the determinants of its upper left square submatrices are positive
    dets = [det(negative_hessian_matrix_evaluated[1:i, 1:i]) for i = 1:length(w)]
    for i = 1:length(dets)
        print(dets[i])
        print(" is greater than zero: ")
        println(dets[i] > AlgebraicNumber(0))
    end
    println("Done.")

    return dets
end
