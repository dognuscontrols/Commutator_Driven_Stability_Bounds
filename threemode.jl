using LinearAlgebra
using JuMP
using COSMO

const MOI = JuMP.MOI

"""
    Options()

Numerical settings for the three-mode example in the paper.

The matrices are fixed as

    A1 = [-1 4 0; 0 -1 0; 0 0 -1]
    A2 = [-1 0 0; -4 -1 0; 0 0 -1]
    A3(r) = [-1 1.6 0; 0 -1 r; 0 0 -1]

Only `r` is swept. The Euclidean commutator edge
`||[A3(r), A1]||_2 = 4r` is the activated edge reported in Table I.
"""
Base.@kwdef struct Options
    gridN::Int = 31
    alpha_min::Float64 = 0.10
    r_values::Vector{Float64} = collect(0.0:0.2:2.0)
    table_r_values::Vector{Float64} = collect(0.0:0.4:2.0)

    epsP::Float64 = 1e-3
    verify_tol::Float64 = 1e-6

    eta_lo::Float64 = 0.0
    eta_hi_init::Float64 = 1.0
    eta_cap::Float64 = 1e6
    eta_tol::Float64 = 1e-4

    tau_lo::Float64 = 0.0
    tau_hi_init::Float64 = 0.1
    tau_cap::Float64 = 2.0
    tau_tol::Float64 = 1e-5

    floquet_tau_hi_init::Float64 = 0.1
    floquet_tau_cap::Float64 = 10.0
    floquet_tol::Float64 = 1e-5

    run_constant_metric::Bool = false
    run_affine_metric::Bool = true
    write_csv::Bool = true
    summary_csv::String = "three_mode_sweep_summary.csv"
    table_csv::String = "three_mode_table1.csv"
    verbosity::Int = 1
end

function run_three_mode_sweep(; opts::Options = Options())
    grid = simplex_grid_3(opts.gridN; alpha_min = opts.alpha_min)

    logmsg(opts, 1, "Restricted Delta_3 grid points: $(size(grid, 1))")
    logmsg(opts, 1, "Restriction: alpha_i >= $(opts.alpha_min)")
    logmsg(opts, 1, "r values: $(opts.r_values)")

    A0 = mode_matrices(0.0)
    floquet0 = sampled_floquet_threshold(A0, grid; opts = opts)

    logmsg(opts, 1, "\n==================== Baseline Floquet threshold ====================")
    logmsg(opts, 1, "Sampled Floquet threshold at r = 0: $(fmt_tau(floquet0.tau, opts.floquet_tau_cap))")
    logmsg(opts, 1, "Worst sampled duty: $(fmt_alpha(floquet0.alpha))")

    results = NamedTuple[]

    for r in opts.r_values
        push!(results, run_case(r, floquet0, grid, opts))
    end

    sort!(results, by = x -> x.r)
    warn_if_floquet_changes(results, floquet0, opts)
    print_summary(results, size(grid, 1), opts)

    if opts.write_csv
        write_summary_csv(opts.summary_csv, results, opts)
        write_paper_table_csv(opts.table_csv, results, opts)
    end

    return (
        grid = grid,
        opts = opts,
        results = results,
    )
end

function run_case(r::Float64, floquet0, grid::AbstractMatrix, opts::Options)
    A = mode_matrices(r)
    edges = commutator_edges(A)
    exact_edges = analytic_edges(r)

    logmsg(opts, 1, "\n-------------------- r = $(r) --------------------")
    logmsg(opts, 1, "edge norms, computed: $(edges)")
    logmsg(opts, 1, "edge norms, analytic: $(exact_edges)")

    floquet = iszero(r) ? floquet0 : sampled_floquet_threshold(A, grid; opts = opts)

    constant = empty_certificate(:constant, size(grid, 1))
    affine = empty_certificate(:affine, size(grid, 1))

    if opts.run_constant_metric
        logmsg(opts, 1, "  constant metric...")
        constant = certify(A, grid, :constant, opts)
    end

    if opts.run_affine_metric
        logmsg(opts, 1, "  affine metric...")
        affine = certify(A, grid, :affine, opts)
    end

    return (
        r = r,
        edge12 = edges.edge12,
        edge13 = edges.edge13,
        edge23 = edges.edge23,
        floquet_tau = floquet0.tau,
        floquet = floquet,
        floquet_delta = floquet.tau - floquet0.tau,
        constant = constant,
        affine = affine,
    )
end

function warn_if_floquet_changes(results, floquet0, opts::Options)
    deviations = [
        abs(res.floquet.tau - floquet0.tau)
        for res in results
        if isfinite(res.floquet.tau)
    ]

    isempty(deviations) && return

    max_dev = maximum(deviations)

    if max_dev > 5 * opts.floquet_tol
        logmsg(opts, 1, "WARNING: sampled Floquet threshold varies beyond bisection tolerance; max deviation = $(max_dev)")
    end
end

function mode_matrices(r::Float64)
    A1 = [
        -1.0   4.0   0.0
         0.0  -1.0   0.0
         0.0   0.0  -1.0
    ]

    A2 = [
        -1.0   0.0   0.0
        -4.0  -1.0   0.0
         0.0   0.0  -1.0
    ]

    A3 = [
        -1.0   1.6   0.0
         0.0  -1.0   r
         0.0   0.0  -1.0
    ]

    return [A1, A2, A3]
end

comm(A, i, j) = A[i] * A[j] - A[j] * A[i]

function commutator_edges(A)
    return (
        edge12 = opnorm(comm(A, 2, 1), 2),
        edge13 = opnorm(comm(A, 3, 1), 2),
        edge23 = opnorm(comm(A, 3, 2), 2),
    )
end

function analytic_edges(r::Float64)
    return (
        edge12 = 16.0,
        edge13 = 4.0 * abs(r),
        edge23 = 6.4,
    )
end

function certify(A, grid::AbstractMatrix, mode::Symbol, opts::Options)
    raw = if mode == :constant
        synthesize_constant_metric(A, grid, opts)
    elseif mode == :affine
        synthesize_affine_metric(A, grid, opts)
    else
        error("Unknown metric mode $(mode).")
    end

    raw.success || return empty_certificate(mode, size(grid, 1); info = raw.info)

    verified = verify_certificate(A, grid, raw, mode, opts)
    tau_star = sampled_tau_star(A, grid, verified, mode, opts)

    return merge(verified, (uniform_tau_star = tau_star,))
end

function empty_certificate(mode::Symbol, Ng::Int; info::String = "No certificate.")
    common = (
        success = false,
        eta = NaN,
        feasible_alpha = falses(Ng),
        uniform_tau_star = NaN,
        info = info,
    )

    if mode == :constant
        return merge(common, (P = zeros(0, 0),))
    elseif mode == :affine
        return merge(common, (Pcell = Matrix{Float64}[],))
    else
        error("Unknown metric mode $(mode).")
    end
end

function synthesize_constant_metric(A, grid::AbstractMatrix, opts::Options)
    eta, sol = maximize_eta(eta -> constant_metric_feasible(A, grid, eta, opts), opts)

    return (
        success = sol.success,
        eta = eta,
        P = sol.P,
        info = sol.info,
    )
end

function constant_metric_feasible(A, grid::AbstractMatrix, eta::Float64, opts::Options)
    n = size(A[1], 1)

    model = Model(COSMO.Optimizer)
    set_silent(model)

    @variable(model, P[1:n, 1:n], Symmetric)
    @constraint(model, P - opts.epsP * I(n) in PSDCone())
    @constraint(model, tr(P) == 1.0)

    for alpha in eachrow(grid)
        S = averaged_matrix(A, alpha)
        L = S' * P + P * S + 2eta * P
        @constraint(model, -sym(L) in PSDCone())
    end

    optimize!(model)

    term = termination_status(model)
    success = term == MOI.OPTIMAL

    return (
        success = success,
        info = string(term),
        P = success ? Matrix(value.(P)) : zeros(0, 0),
    )
end

function synthesize_affine_metric(A, grid::AbstractMatrix, opts::Options)
    eta, sol = maximize_eta(eta -> affine_metric_feasible(A, grid, eta, opts), opts)

    return (
        success = sol.success,
        eta = eta,
        Pcell = sol.Pcell,
        info = sol.info,
    )
end

function affine_metric_feasible(A, grid::AbstractMatrix, eta::Float64, opts::Options)
    n = size(A[1], 1)
    m = length(A)

    model = Model(COSMO.Optimizer)
    set_silent(model)

    Pvars = [
        @variable(model, [1:n, 1:n], Symmetric, base_name = "P_$i")
        for i in 1:m
    ]

    for i in 1:m
        @constraint(model, Pvars[i] - opts.epsP * I(n) in PSDCone())
    end

    @constraint(model, sum(Pvars[i][j, j] for i in 1:m, j in 1:n) == m)

    for alpha in eachrow(grid)
        S = averaged_matrix(A, alpha)
        Palpha = affine_metric_expr(alpha, Pvars)

        @constraint(model, Palpha - opts.epsP * I(n) in PSDCone())

        L = S' * Palpha + Palpha * S + 2eta * Palpha
        @constraint(model, -sym(L) in PSDCone())
    end

    optimize!(model)

    term = termination_status(model)
    success = term == MOI.OPTIMAL

    return (
        success = success,
        info = string(term),
        Pcell = success ? [Matrix(value.(Pvars[i])) for i in 1:m] : Matrix{Float64}[],
    )
end

function maximize_eta(feasibility, opts::Options)
    lo = max(opts.eta_lo, 0.0)
    sol_lo = feasibility(lo)

    sol_lo.success || return lo, sol_lo

    hi = max(opts.eta_hi_init, lo + 1.0)
    sol_hi = feasibility(hi)

    while sol_hi.success && hi < opts.eta_cap
        hi = 2hi + 1
        sol_hi = feasibility(hi)
    end

    sol_hi.success && return lo, sol_lo

    best = sol_lo

    while hi - lo > opts.eta_tol
        mid = 0.5 * (lo + hi)
        sol_mid = feasibility(mid)

        if sol_mid.success
            lo = mid
            best = sol_mid
        else
            hi = mid
        end
    end

    return lo, best
end

function verify_certificate(A, grid::AbstractMatrix, cert, mode::Symbol, opts::Options)
    Ng = size(grid, 1)
    feasible = falses(Ng)

    for (j, alpha_row) in enumerate(eachrow(grid))
        alpha = collect(alpha_row)
        P = metric_matrix(alpha, cert, mode)

        if minimum(eigvals(Symmetric(sym(P)))) <= opts.epsP / 10
            feasible[j] = false
            continue
        end

        S = averaged_matrix(A, alpha)
        L = sym(S' * P + P * S + 2cert.eta * P)
        feasible[j] = eigmax(Symmetric(L)) <= opts.verify_tol
    end

    return merge(cert, (success = any(feasible), feasible_alpha = feasible))
end

function sampled_tau_star(A, grid::AbstractMatrix, cert, mode::Symbol, opts::Options)
    cert.success || return NaN
    cert.eta > 0 || return NaN

    idx = findall(cert.feasible_alpha)
    isempty(idx) && return NaN

    qstar = tau -> maximum(
        q_bound(
            A,
            collect(grid[j, :]),
            metric_matrix(collect(grid[j, :]), cert, mode),
            cert.eta,
            tau,
        )
        for j in idx
    )

    qstar(max(opts.tau_tol, 1e-10)) < 1 || return NaN

    lo = opts.tau_lo
    hi = opts.tau_hi_init

    while hi < opts.tau_cap && prefix_condition_holds(qstar, hi; samples = 25)
        lo = hi
        hi *= 2
    end

    hi = min(hi, opts.tau_cap)

    if prefix_condition_holds(qstar, hi; samples = 25)
        return hi
    end

    while hi - lo > opts.tau_tol
        mid = 0.5 * (lo + hi)

        if prefix_condition_holds(qstar, mid; samples = 25)
            lo = mid
        else
            hi = mid
        end
    end

    return lo
end

function prefix_condition_holds(qstar, tau::Float64; samples::Int = 25)
    tau <= 0 && return true

    for s in range(tau / samples, tau, length = samples)
        qstar(s) < 1 || return false
    end

    return true
end

function q_bound(A, alpha::AbstractVector, P::AbstractMatrix, eta::Float64, tau::Float64)
    B = [alpha[i] * A[i] for i in eachindex(A)]
    M = [induced_P_norm(B[i], P) for i in eachindex(B)]

    split = 0.0

    for k in 2:length(B)
        inner = 0.0

        for i in 1:(k - 1)
            inner += induced_P_norm(B[k] * B[i] - B[i] * B[k], P)
        end

        split += gfun(tau * M[k]) * inner
    end

    return exp(-eta * tau) + tau^2 * exp(tau * sum(M)) * split
end

gfun(z::Float64) =
    abs(z) < 1e-10 ? 0.5 : (exp(2z) - 1 - 2z) / (4z^2)

function sampled_floquet_threshold(A, grid::AbstractMatrix; opts::Options)
    best_tau = Inf
    best_alpha = nothing

    for alpha_row in eachrow(grid)
        alpha = collect(alpha_row)
        tau = first_floquet_crossing(A, alpha, opts)

        if tau < best_tau
            best_tau = tau
            best_alpha = alpha
        end
    end

    return (tau = best_tau, alpha = best_alpha)
end

function first_floquet_crossing(A, alpha::AbstractVector, opts::Options)
    rho(tau) = spectral_radius(monodromy(A, alpha, tau))

    lo = 0.0
    hi = opts.floquet_tau_hi_init

    while hi < opts.floquet_tau_cap && rho(hi) < 1.0
        lo = hi
        hi *= 1.5
    end

    hi = min(hi, opts.floquet_tau_cap)

    if rho(hi) < 1.0
        return Inf
    end

    while hi - lo > opts.floquet_tol
        mid = 0.5 * (lo + hi)

        if rho(mid) < 1.0
            lo = mid
        else
            hi = mid
        end
    end

    return lo
end

function monodromy(A, alpha::AbstractVector, tau::Float64)
    n = size(A[1], 1)
    Phi = Matrix{Float64}(I, n, n)

    for i in eachindex(A)
        Phi = exp(tau * alpha[i] * A[i]) * Phi
    end

    return Phi
end

spectral_radius(M::AbstractMatrix) = maximum(abs.(eigvals(M)))

function averaged_matrix(A, alpha::AbstractVector)
    S = zeros(Float64, size(A[1])...)

    for i in eachindex(A)
        S .+= alpha[i] .* A[i]
    end

    return S
end

sym(M) = 0.5 .* (M .+ M')

function affine_metric_expr(alpha::AbstractVector, Pvars)
    n = size(Pvars[1], 1)
    m = length(Pvars)

    return sym([
        sum(alpha[i] * Pvars[i][r, c] for i in 1:m)
        for r in 1:n, c in 1:n
    ])
end

function affine_metric_matrix(alpha::AbstractVector, Pcell::Vector{<:AbstractMatrix})
    P = zeros(Float64, size(Pcell[1])...)

    for i in eachindex(Pcell)
        P .+= alpha[i] .* Pcell[i]
    end

    return sym(P)
end

function metric_matrix(alpha::AbstractVector, cert, mode::Symbol)
    if mode == :constant
        return sym(cert.P)
    elseif mode == :affine
        return affine_metric_matrix(alpha, cert.Pcell)
    else
        error("Unknown metric mode $(mode).")
    end
end

function induced_P_norm(M::AbstractMatrix, P::AbstractMatrix)
    R = cholesky(Symmetric(sym(P))).U
    return opnorm(R * M / R, 2)
end

function simplex_grid_3(N::Int; alpha_min::Float64 = 0.0)
    N >= 2 || error("gridN must be at least 2.")
    0.0 <= alpha_min < 1 / 3 || error("Need 0 <= alpha_min < 1/3.")

    pts = Vector{NTuple{3, Float64}}()

    for i in 0:N
        for j in 0:(N - i)
            a1 = i / N
            a2 = j / N
            a3 = 1.0 - a1 - a2

            if minimum((a1, a2, a3)) >= alpha_min - 1e-12
                push!(pts, (a1, a2, a3))
            end
        end
    end

    isempty(pts) && error("No grid points. Increase gridN or decrease alpha_min.")

    grid = zeros(Float64, length(pts), 3)

    for (k, p) in enumerate(pts)
        grid[k, :] .= collect(p)
    end

    return grid
end

function print_summary(results, Ng::Int, opts::Options)
    println("\n==================== SUMMARY ====================")

    for res in results
        println("\nr=$(res.r)")
        println("  unscaled edges: [A2,A1]=$(round(res.edge12, digits=6)), [A3,A1]=$(round(res.edge13, digits=6)), [A3,A2]=$(round(res.edge23, digits=6))")
        println("  sampled Floquet tau: $(fmt_tau(res.floquet.tau, Inf)); worst alpha=$(fmt_alpha(res.floquet.alpha)); Delta=$(res.floquet_delta)")
        opts.run_constant_metric && print_certificate("constant", res.constant, Ng, res.floquet_tau)
        print_certificate("affine", res.affine, Ng, res.floquet_tau)
    end
end

function print_certificate(name::String, cert, Ng::Int, floquet_tau)
    feasible_count = count(cert.feasible_alpha)
    ratio = conservatism_ratio(floquet_tau, cert.uniform_tau_star)

    println("  $(name): success=$(cert.success), feasible=$(feasible_count)/$(Ng), eta=$(round_or_nan(cert.eta)), tau*=$(round_or_nan(cert.uniform_tau_star)), ratio=$(round_or_nan(ratio))")
end

function write_summary_csv(path::String, results, opts::Options)
    open(path, "w") do io
        println(io, join([
            "r",
            "edge12_unscaled",
            "edge13_unscaled",
            "edge23_unscaled",
            "metric",
            "success",
            "eta_star",
            "tau_cert",
            "feasible_count",
            "total_count",
            "floquet_tau",
            "floquet_tau_diag",
            "floquet_delta",
            "floquet_alpha1",
            "floquet_alpha2",
            "floquet_alpha3",
            "conservatism_ratio",
        ], ","))

        for res in results
            alpha = res.floquet.alpha
            a = alpha === nothing ? (NaN, NaN, NaN) : Tuple(alpha)

            metric_rows = opts.run_constant_metric ?
                (("constant", res.constant), ("affine", res.affine)) :
                (("affine", res.affine),)

            for (metric_name, cert) in metric_rows
                ratio = conservatism_ratio(res.floquet_tau, cert.uniform_tau_star)

                println(io, join([
                    res.r,
                    res.edge12,
                    res.edge13,
                    res.edge23,
                    metric_name,
                    cert.success,
                    cert.eta,
                    cert.uniform_tau_star,
                    count(cert.feasible_alpha),
                    length(cert.feasible_alpha),
                    res.floquet_tau,
                    res.floquet.tau,
                    res.floquet_delta,
                    a[1],
                    a[2],
                    a[3],
                    ratio,
                ], ","))
            end
        end
    end
end

function write_paper_table_csv(path::String, results, opts::Options)
    rows = NamedTuple[]

    for r in opts.table_r_values
        idx = findfirst(res -> isapprox(res.r, r; atol = 1e-12), results)
        idx === nothing && error("Missing table row for r = $(r)")
        push!(rows, results[idx])
    end

    open(path, "w") do io
        println(io, join([
            "r",
            "edge13_unscaled",
            "eta_star",
            "tau_cert",
            "floquet_over_tau_cert",
        ], ","))

        for res in rows
            cert = res.affine
            ratio = conservatism_ratio(res.floquet_tau, cert.uniform_tau_star)

            println(io, join([
                res.r,
                res.edge13,
                cert.eta,
                cert.uniform_tau_star,
                ratio,
            ], ","))
        end
    end
end

conservatism_ratio(floquet_tau, cert_tau) =
    isfinite(floquet_tau) && isfinite(cert_tau) && cert_tau > 0 ?
    floquet_tau / cert_tau :
    NaN

round_or_nan(x; digits = 6) = isfinite(x) ? round(x, digits = digits) : x
fmt_alpha(alpha) = alpha === nothing ? "none" : string(round.(alpha, digits = 4))
fmt_tau(tau, cap) = isfinite(tau) ? string(round(tau, digits = 6)) : "> $(cap)"
logmsg(opts::Options, level::Int, msg) = opts.verbosity >= level && println(msg)

if abspath(PROGRAM_FILE) == @__FILE__
    run_three_mode_sweep()
end
