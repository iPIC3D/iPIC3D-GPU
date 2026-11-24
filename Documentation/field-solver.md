# Field solver

[Documentation index](../README.md#documentation)

## Scheme

Each cycle solves implicitly for E(n+θ), with δ = cθΔt:

(I+μ)E − δ²(∇²E + ∇∇·(μE)) = Eⁿ + δ(∇×Bⁿ − (4π/c)Ĵ) − 4πδ²∇ρ̂

μ is the implicit susceptibility, and Ĵ and ρ̂ are the implicit (hat) moments. The solved field is then used as follows, with S the smoothing:

| Use | Field |
|---|---|
| Particle mover | S(E^{n+θ}) |
| Faraday's law, Bⁿ⁺¹ = Bⁿ − cΔt ∇×E^{n+θ} | The unsmoothed E^{n+θ} |
| Advance, Eⁿ⁺¹ = (S(E^{n+θ}) − (1−θ)Eⁿ)/θ | S(E^{n+θ}) |

Boundary rows of the solve are described in [Boundary conditions](boundary-conditions.md#boundary-rows-of-the-implicit-solve).

## Parameters

| Key | Default | Description |
|---|---|---|
| `SolverType` | `GMRES` | `GMRES`: GMRES(20). `FGMRESBlockJacobi`: FGMRES(20) with a communication-free point-block Jacobi preconditioner. `Chebyshev`: Chebyshev semi-iteration, without either [run guard](#run-guards). The last two are experimental, and the run warns. Any other value stops the run at startup |
| `GMREStol` | `1e-3` | Relative tolerance: stop when ‖r‖ < `GMREStol` ‖r₀‖, r₀ being the residual of the initial guess Eⁿ. A warm start with ‖r₀‖ ≤ `GMREStol` ‖b‖ is kept without iterating |
| `BlockJacobiSweeps`, `BlockJacobiOmega` | `1`, `1.0` | Richardson sweeps and relaxation of the block-Jacobi preconditioner |
| `ChebyshevMaxIter` | `20` | Iterations of `SolverType = Chebyshev` |
| `ChebyshevEigMin`, `ChebyshevEigMax` | `0.0` | Its eigenvalue bounds. At 0 or below, the lower bound is 1 and the upper bound is estimated |
| `Smooth`, `SmoothNiter` | `1.0`, `6` | [Smoothing](#smoothing) coefficient and passes. `Smooth = 1` or `SmoothNiter = 0` disables it |
| `HaloOverlap` | `0` | `1`: [overlap the halo exchanges](#overlapped-halo-exchange) with computation |
| `PoissonCorrection`, `PoissonCorrectionCycle` | `no`, `10` | ∇·E correction (`yes`) and its interval, at least 1 when on |
| `divBCorrection`, `divBCorrectionCycle` | `no`, `10` | ∇·B cleaning (`yes`) and its interval, at least 1 when on. Needs at least 2 cells along every non-periodic direction |
| `DivCleaningTol` | `1e-6` | Relative tolerance of both corrections, in (0, 1) when one is on |
| `PoissonChebMaxIter` | `20` | Operator applications of the corrections' Chebyshev preconditioner (residual polynomial of degree `PoissonChebMaxIter` + 1) |
| `PoissonChebEigRatio` | `300` | Floor of its lower eigenvalue bound, λ_max / ratio. `0` or below uses the exact bounds |
| `PoissonChebRescaleEigMin`, `PoissonChebRescaleEigMax` | `1.0`, `1.0` | Factors on its lower and upper bounds. An inverted interval stops the run |
| `SolverSourceGrowthLimit` | `100` | [Guard](#run-guards) on the growth of ‖b‖; `0` disables it |
| `SolverStopOnNonConvergence` | `1` | [Guard](#run-guards) on non-converged solves; `0` only warns |

## Smoothing

`Smooth` = a applies a f + (1−a)/6 (sum of the 6 neighbours), `SmoothNiter` times, to E after each solve and to ρ_c and Ĵ before ρ̂ is formed. `a` must lie in [0, 1].

Below (2d − 3)/(2d), with d the number of directions that are not a single periodic cell (`0.5` in 3D, `0.25` in 2D), the response is negative at grid scale and those modes flip sign every pass. With an odd `SmoothNiter` they come out inverted and the run warns, so use an even value there.

On domain faces, the E smoothing keeps the values the solve imposes (E_t = 0 on conductors, and E0 on [field inflow faces](boundary-conditions.md#boundary-codes) with `ApplyInflowBcsEImage = 1`) and continues the other components evenly about the face node.

## Overlapped halo exchange

`HaloOverlap = 1` overlaps the exchanges of the Maxwell image, the ∇·E and ∇·B images, smoothing, and B and hat-moment interpolation with the computation that does not read the exchanged values. Results are the same as with `0`.

- It is meant for multi-rank runs. On one rank there is no MPI to hide, and the extra boundary-layer launches can make it slower.
- A ∇·B cleaning cycle runs the B update blocking.
- The Maxwell image overlaps on ranks with at least 3 cells along every axis that is not flat. With a flat axis (the y of an x-z run), only the image's node stage overlaps. Other ranks use the blocking exchange.

## Run guards

| Guard | Stops the run when |
|---|---|
| `SolverSourceGrowthLimit` | ‖b‖ of the Maxwell right-hand side exceeds the limit times its value in the previous cycle. Normally ‖b‖ changes by tens of percent between cycles, and up to a few times in the first cycles and on ∇·E correction cycles; a corrupted moment or field makes it jump by orders of magnitude. The first solve of a run, also after a restart, is not checked |
| `SolverStopOnNonConvergence` | A Krylov solve reaches its restart limit (200 for Maxwell, 66 for a correction) without meeting its tolerance |

Either guard prints the reason on rank 0 and calls `MPI_Abort` before the failed result is used. No further output or checkpoint is written, including the final one, so the restart slots keep the checkpoints written before the failure. A non-finite residual stops any Krylov solve at once, whatever these keys.

## ∇·B cleaning

`divBCorrection = yes` cleans the cell B every `divBCorrectionCycle` cycles, in the B update after Faraday's law and the face treatments, before B is interpolated to the nodes. Faraday's law does not change divC2N(B) (divC2N∘curlN2C = 0), so the cleaning removes the divergence of the initial or restarted field and what the face treatments create.

1. It solves divC2N(gradN2C ψ) = divC2N(B) for ψ on the nodes. Nodes of a domain face are not unknowns. On a conductor, ψ equals the next inner node (zero normal gradient), so the wall B_n stays as Faraday's law made it. On other faces ψ = 0, so the correction has no tangential part there.
2. It subtracts gradN2C ψ from the cell B. This is the projection of B onto fields without divergence at the interior nodes, to `DivCleaningTol`. A divergence-free field is left alone and nothing filters B.

Because it keeps B_n on a conductor, the projection is orthogonal only where B does not cross the conductor. The B energy changes by −‖gradN2C ψ‖² − 2∮ψ B_n, so where the initial B crosses a wall it can rise (by a few percent in a cold test with y walls).

Use the cleaning with the [absorbing layer](boundary-conditions.md#absorbing-layer-on-open-faces) and `sal_tangential_B = 1`. With `sal_tangential_B = 0`, or without the layer on an open X-left face or a field inflow face, those faces hold B_n at B0_n, the cleaning carries that flux into the whole box, and the code warns.

## ∇·E correction

`PoissonCorrection = yes` corrects Eⁿ at the start of every `PoissonCorrectionCycle`-th field step, before the Maxwell solve: it solves divN2C(gradC2N φ) = divN2C(E) − 4πρ_c for φ on the cells, with the smoothed ρ_c, and subtracts gradC2N φ from Eⁿ on the nodes.

Ghost cells of φ:

- On a conductor, the odd image of φ: φ = 0 on the wall, which stays an equipotential, and the correction has no tangential part there.
- On other domain faces, 0.

With these rules gradC2N is minus the transpose of divN2C in the lumped inner product with E_t = 0 on the walls, so the step is the orthogonal projection onto the E of the discrete Gauss law.

After the correction, the values the solve imposes are restored (see [Smoothing](#smoothing)), and Eⁿ gets the boundary of a solved E: ghost nodes are exchanged and filled, then the open-face layers are set. So for θ < 1 the E advance keeps the imposed face values.

## Solves of the ∇·E and ∇·B corrections

**Operator.** Both corrections solve with the operator of their correction, the 27-point Σ_a δ_a² μ_b² μ_c² (δ² the (1, −2, 1)/h² difference, μ² the (¼, ½, ¼) average).

**Solver.** FGMRES(60), at most 66 restarts (about 4000 iterations), to `DivCleaningTol`. A correction leaves the fraction `DivCleaningTol` of the divergence it found, and a 10⁻⁶ solve costs 2 to 11 times a 10⁻³ one.

**Preconditioner.** On each rank, a Chebyshev polynomial in the rank's local operator ([`PoissonChebMaxIter`](#parameters) applications): block Jacobi, zero beyond the rank's rows, and for ∇·B also the face rows and the conductor copy. Its exact eigenvalue bounds [λ_min, λ_max] come from the 1D spectra of the averages.

**Eigenvalue bounds.** With the exact λ_min of a large block (λ_max/λ_min ≈ 1.5·10⁵ for 960 × 768 cells in 2D) the polynomial barely reduces the residual. `PoissonChebEigRatio` raises the lower bound to λ_max/ratio. The polynomial then clusters the upper spectrum (residual factor ≈ 0.18 at 300, 0.03 at 100), and FGMRES resolves the modes below, including the near-null checkerboard modes of the 3D operators (about 10⁻⁶ λ_max for 63 nodes per axis). Keep `PoissonChebRescaleEigMax` at 1: at 0.95 the iterations rise about tenfold. A larger ratio, or `PoissonChebRescaleEigMin` below 1, saves iterations at tight tolerances and costs them at loose ones.

**Null spaces.** The operators are singular in two cases, and the right-hand side loses its null-space components before the solve (the gradient does not depend on them):

- *Constants:* for ∇·E with all three directions periodic (removing the mean charge, a uniform neutralising background), and for ∇·B when every face is periodic or a conductor. Between conductors, the mean of divC2N(B) is the net flux of the wall cells' B_n, which no correction can change, so a uniform ∇·B of that size remains.
- *Checkerboards* (−1)^(i_a+i_b) f(i_c), for any profile f along the third direction, of two directions that each carry the checkerboard. A direction carries it when it is periodic with an even cell count, or, for ∇·E, between two conductors with any cell count (the odd ghosts of φ). The charge density can hold them, but no E produces them through divN2C, so that part of the charge stays in the Gauss residual. divC2N(B) holds them only at round-off.

**Skipped solves.** A right-hand side at the round-off of its fields leaves nothing to clean, and the solve is skipped: ‖b‖ ≤ 10³ ε ‖B‖ (Σ_a h_a⁻²)^½ for ∇·B, and ‖b‖ ≤ 10³ ε (‖E‖² Σ_a h_a⁻² + 16π² Σ_s ‖ρ_s‖²)^½ for ∇·E.
