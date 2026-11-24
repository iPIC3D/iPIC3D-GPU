#!/usr/bin/env python3
"""Print iPIC3D input-derived startup checks without running MPI/CUDA."""


import math
import os
import re
import sys
from pathlib import Path


HEAT_FLUX_COMPONENTS = 10
# the default of the CMake option of the same name (initial capacity of the
# device particle arrays, times the particle count); a build with another
# value needs proportionally more particle memory than estimated here
PARTICLE_ARRAY_HEADROOM = 1.2
# the numbers that the input lists of Collective::ReadInput read (a C++
# stream): no nan, inf, hexadecimal or digit separators
INT_TOKEN = re.compile(r"[+-]?[0-9]+")
REAL_TOKEN = re.compile(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?")
INT_RANGE = (-2**31, 2**31 - 1)
# the whitespace of a C++ stream in the C locale
STREAM_SPACE = re.compile(r"[ \t\n\v\f\r]+")


def cxx(x):
    """Approximate std::cout default numeric formatting."""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return f"{x:.6g}"
    return str(x)


def parse_config(path):
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    entries = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].split("#", 1)[0]
        if "EndConfigFile" in line:
            break
        if "=" not in line:
            i += 1
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        i += 1

        # ConfigFile supports continuation lines until the next key line.
        extra = []
        while i < len(lines):
            if not lines[i].strip():
                break  # a blank line ends the value, a comment line does not
            nxt = lines[i].split("#", 1)[0]
            stripped = nxt.strip()
            if not stripped:
                i += 1
                continue
            if "EndConfigFile" in nxt or "=" in nxt:
                break
            extra.append(stripped)
            i += 1
        if extra:
            value = "\n".join([value] + extra if value else extra)
        entries[key] = value
    return entries


def as_bool(value: str) -> bool:
    return value.strip().upper() not in {"FALSE", "F", "NO", "N", "0", "NONE"}


class Input:
    def __init__(self, values):
        self.values = values

    def raw(self, key: str, default=None):
        if key in self.values:
            return self.values[key]
        if default is not None:
            return default
        raise SystemExit(f"key not found: {key}")

    def string(self, key: str, default=None) -> str:
        return str(self.raw(key, default))

    def integer(self, key: str, default=None) -> int:
        return int(float(self.raw(key, default)))

    def real(self, key: str, default=None) -> float:
        return float(self.raw(key, default))

    def boolean(self, key: str, default=None) -> bool:
        raw = self.raw(key, default)
        if isinstance(raw, bool):
            return raw
        return as_bool(str(raw))

    def species_list(self, key: str, n: int, whose: str, kind=float):
        """The n values of a per-species list, one per species in order (of
        whose); a missing key, another number of values, or a value that is
        not a number of the kind stops, as in Collective::ReadInput."""
        raw = str(self.values.get(key, ""))
        token = INT_TOKEN if kind is int else REAL_TOKEN
        tokens = [v for v in STREAM_SPACE.split(raw) if v]
        ok = len(tokens) == n and all(token.fullmatch(v) for v in tokens)
        vals = [kind(v) for v in tokens] if ok else []
        # out of the range of the type (3000000000, 1e400): the stream fails
        if kind is int:
            ok = ok and all(INT_RANGE[0] <= v <= INT_RANGE[1] for v in vals)
        else:
            ok = ok and all(math.isfinite(v) for v in vals)
        if not ok:
            raise SystemExit(f'{key} needs {n} value{"" if n == 1 else "s"}, '
                             f'one per {whose}, not "{raw}"')
        return vals


def bc_em_name(code: int) -> str:
    return {
        0: "perfect conductor",
        1: "perfect magnetic conductor (first order)",
        2: "open",
    }.get(code, "unknown")


def bc_p_name(code: int) -> str:
    return {
        0: "exit",
        1: "perfect mirror",
        2: "open inflow",
        3: "open outflow",
    }.get(code, "unknown")


def track_particle_id_mask(cfg: Input, ns: int):
    """TrackParticleID per species, as Collective::parseTrackParticleIDMask:
    one value applies to all species; 0/false/f/no/n/none (any case) is off."""
    values = [tok.lower() not in {"0", "false", "f", "no", "n", "none"}
              for tok in str(cfg.raw("TrackParticleID", "")).split()]
    if len(values) == 1:
        return values * ns
    return (values + [False] * ns)[:ns]


# Heat-flux tokens of MomentsOutputTag (OutputTagConfig.h): Q or a component,
# for all species, one species (digit suffix) or the sum (_tot)
HEAT_FLUX_TOKEN = re.compile(
    r"^Q(xxx|xxy|xxz|xyy|xyz|xzz|yyy|yyz|yzz|zzz)?(\d+|_tot)?$")


def heat_flux_requested(cfg: Input) -> bool:
    """Field output writes heat flux, as in c_Solver::initCUDA."""
    tokens = [t.strip() for t in str(cfg.raw("MomentsOutputTag", "")).split("+")]
    return (cfg.integer("FieldOutputCycle", 100) > 0
            and any(HEAT_FLUX_TOKEN.match(t) for t in tokens if t))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def print_topology(xlen: int, ylen: int, zlen: int,
                   periodic_x: bool, periodic_y: bool, periodic_z: bool) -> None:
    nprocs = xlen * ylen * zlen
    threads = os.environ.get("OMP_NUM_THREADS", "1")
    print()
    print(f"Number of processes = {nprocs}")
    print("-------------------------")
    print(f"Number of threads = {threads}")
    print("-------------------------")
    print()
    print("Virtual Cartesian Processors Topology")
    print("-------------------------------------")
    print(f"Processors grid: {xlen}x{ylen}x{zlen}")
    print(f"Periodicity X: {1 if periodic_x else 0}")
    print(f"Periodicity Y: {1 if periodic_y else 0}")
    print(f"Periodicity Z: {1 if periodic_z else 0}")
    print()


def print_report(cfg: Input) -> None:
    """Print the startup report of Collective::Print. The input checks shared with
    Collective::ReadInput run in its order and with its messages, so a deck with
    several errors stops at the same first one."""
    dt = cfg.real("dt")
    ncycles = cfg.integer("ncycles")
    th = cfg.real("th", 1.0)
    if not th >= 0.5:
        raise SystemExit(f"th must be at least 0.5, not {cxx(th)}")
    smooth = cfg.real("Smooth", 1.0)
    smooth_niter = cfg.integer("SmoothNiter", 6)
    halo_overlap = cfg.integer("HaloOverlap", 0) != 0
    save_dir = cfg.string("SaveDirName", "data")
    ns = cfg.integer("ns")
    ns_test = cfg.integer("nsTestPart", 0)
    if ns < 1 or ns_test < 0:
        raise SystemExit(f"ns must be at least 1 and nsTestPart at least 0, "
                         f"not {ns} and {ns_test}")

    b0x = cfg.real("B0x", 0.0)
    b0y = cfg.real("B0y", 0.0)
    b0z = cfg.real("B0z", 0.0)
    delta = cfg.real("delta", 0.5)

    case = cfg.string("Case")
    # the Harris pressure law on the perturbed flux function; the GEM
    # perturbation current, which fills the box, cannot be carried by the
    # sheet electrons
    ampere = cfg.integer("currentFromAmpere", 0)
    if cfg.integer("spatiallyVaryingThermal", 0) and not (
            case == "GEMHarris" and ampere and b0x != 0.0 and b0y == 0.0):
        raise SystemExit("spatiallyVaryingThermal = 1 needs Case = GEMHarris, "
                         "currentFromAmpere = 1, B0x != 0 and B0y = 0")
    if case == "GEMHarris" and ampere and cfg.real("pertGEM", 0.0) != 0.0:
        raise SystemExit("pertGEM needs currentFromAmpere = 0: its current fills "
                         "the box, where the sheet electrons that would carry it "
                         "have no density")
    # the HDF5 methods also need a build with them, which this script cannot tell
    field_method = cfg.string("FieldWriteMethod")
    if field_method not in ("shdf5", "phdf5", "pvtk", "nbcvtk"):
        raise SystemExit(f"FieldWriteMethod must be shdf5, phdf5, pvtk or nbcvtk, "
                         f"not {field_method}")
    sim_name = cfg.string("SimulationName")
    poisson = cfg.string("PoissonCorrection", "no")
    poisson_cycle = cfg.integer("PoissonCorrectionCycle", 10)
    divb = cfg.string("divBCorrection", "no")
    divb_cycle = cfg.integer("divBCorrectionCycle", 10)
    for name, on, every in (("PoissonCorrectionCycle", poisson == "yes", poisson_cycle),
                            ("divBCorrectionCycle", divb == "yes", divb_cycle)):
        if on and every < 1:
            raise SystemExit(f"{name} must be at least 1, not {every}")
    # one value per species
    rho_init = cfg.species_list("rhoINIT", ns, "species")
    rho_inject = cfg.species_list("rhoINJECT", ns, "species")
    div_cleaning_tol = cfg.real("DivCleaningTol", 1e-6)
    if (poisson == "yes" or divb == "yes") and not 0.0 < div_cleaning_tol < 1.0:
        raise SystemExit(f"DivCleaningTol must be in (0, 1), not {cxx(div_cleaning_tol)}")
    niter_mover = cfg.integer("NiterMover", 3)
    if niter_mover < 1:
        raise SystemExit(f"NiterMover must be at least 1, not {niter_mover}")
    solver_type = cfg.string("SolverType", "GMRES")
    if solver_type not in ("GMRES", "FGMRESBlockJacobi", "Chebyshev"):
        raise SystemExit(f"SolverType must be GMRES, FGMRESBlockJacobi or Chebyshev, not {solver_type}")

    c = cfg.real("c", 1.0)
    if not c > 0.0:
        raise SystemExit(f"the speed of light c must be positive, not {cxx(c)}")
    lx = cfg.real("Lx", 10.0)
    ly = cfg.real("Ly", 10.0)
    lz = cfg.real("Lz", 10.0)
    nxc = cfg.integer("nxc", 64)
    nyc = cfg.integer("nyc", 64)
    nzc = cfg.integer("nzc", 64)
    xlen = cfg.integer("XLEN", 1)
    ylen = cfg.integer("YLEN", 1)
    zlen = cfg.integer("ZLEN", 1)
    # every rank along an axis gets ceil(cells / ranks) cells and the last
    # one the rest, at least one
    for cell_key, cells, rank_key, ranks in (("nxc", nxc, "XLEN", xlen),
                                             ("nyc", nyc, "YLEN", ylen),
                                             ("nzc", nzc, "ZLEN", zlen)):
        if cells < 1 or ranks < 1:
            raise SystemExit(f"{cell_key} and {rank_key} must be positive, not {cells} and {ranks}")
        regular = ceil_div(cells, ranks)
        if cells - regular * (ranks - 1) < 1:
            raise SystemExit(f"{cell_key} = {cells} over {rank_key} = {ranks} ranks of {regular} cells "
                             f"each leaves the last rank no cells; choose values that leave it at "
                             f"least one, for example with {rank_key} dividing {cell_key}")
    periodic_x = cfg.boolean("PERIODICX", True)
    periodic_y = cfg.boolean("PERIODICY", True)
    periodic_z = cfg.boolean("PERIODICZ", True)
    # outside [0, 1] some mode grows
    if not 0.0 <= smooth <= 1.0:
        raise SystemExit(f"Smooth must be in [0, 1], not {cxx(smooth)}")
    # Collective::smoothFlipBound: the filter's response is negative below
    # (2d-3)/(2d), d the directions that are not a single periodic cell
    dims = sum(not (n == 1 and p) for n, p in
               ((nxc, periodic_x), (nyc, periodic_y), (nzc, periodic_z)))
    smooth_flip = (2 * dims - 3) / (2 * dims) if dims > 1 else 0.0
    # the div(B) face rules take the next inner node
    if divb == "yes" and any(n < 2 and not p for n, p in
                             ((nxc, periodic_x), (nyc, periodic_y), (nzc, periodic_z))):
        raise SystemExit("divBCorrection needs at least 2 cells along every non-periodic "
                         "direction (a single-cell direction must be periodic)")

    planet_reflection = cfg.integer("planetReflectionType", 1)
    if planet_reflection not in (0, 1):
        raise SystemExit(f"planetReflectionType must be 0 (specular) or 1 (diffuse), "
                         f"not {planet_reflection}")
    ns_solar = cfg.integer("ns_solar_wind", ns)
    ns_planetary = cfg.integer("ns_planetary", 0)
    exosphere_enabled = cfg.integer("AddExosphereInjection", 0)
    max_injection_radius = cfg.real("RmaxExosphereInjection", 3.0)
    l_square = cfg.real("L_square", 5.0)
    if not (0 <= ns_solar <= ns and ns_planetary >= 0):
        raise SystemExit(f"ns_solar_wind must be from 0 to ns ({ns}) and ns_planetary "
                         f"at least 0, not {ns_solar} and {ns_planetary}")
    if exosphere_enabled and ns_planetary != ns - ns_solar:
        raise SystemExit(f"AddExosphereInjection = 1 needs ns_planetary = ns - "
                         f"ns_solar_wind = {ns - ns_solar}, not {ns_planetary}")
    if ns_planetary > 0:
        planetary = "planetary species"
        neutral_density = cfg.species_list("NeutralSurfaceDensity", ns_planetary, planetary)
        scale_height = cfg.species_list("ExosphericScaleHeight", ns_planetary, planetary)
        photo_freq = cfg.species_list("PhotoionizationFrequency", ns_planetary, planetary)
        weight_ratio = cfg.species_list("MacroParticleWeightRatio", ns_planetary, planetary)

    # one value per species
    uth = cfg.species_list("uth", ns, "species")
    vth = cfg.species_list("vth", ns, "species")
    wth = cfg.species_list("wth", ns, "species")
    for key in ("u0", "v0", "w0"):  # read, not printed
        cfg.species_list(key, ns, "species")
    if ns_test > 0:
        for key in ("pitch_angle", "energy"):
            cfg.species_list(key, ns_test, "test-particle species")
    all_species = "species and test-particle species"
    npcelx = cfg.species_list("npcelx", ns + ns_test, all_species, int)
    npcely = cfg.species_list("npcely", ns + ns_test, all_species, int)
    npcelz = cfg.species_list("npcelz", ns + ns_test, all_species, int)
    qom = cfg.species_list("qom", ns + ns_test, all_species)
    npcel = [npcelx[i] * npcely[i] * npcelz[i] for i in range(ns + ns_test)]
    for i in range(ns + ns_test):
        if min(npcelx[i], npcely[i], npcelz[i]) < 0:
            raise SystemExit(f"npcelx, npcely and npcelz of species {i} must be at "
                             f"least 0, not {npcelx[i]} {npcely[i]} {npcelz[i]}")
    for i in range(ns):
        if rho_init[i] < 0.0 or rho_inject[i] < 0.0:
            raise SystemExit(f"rhoINIT and rhoINJECT of species {i} must be at least "
                             f"0, not {cxx(rho_init[i])} and {cxx(rho_inject[i])}")
        if (rho_init[i] > 0.0 or rho_inject[i] > 0.0) and npcel[i] == 0:
            raise SystemExit(f"species {i} has rhoINIT = {cxx(rho_init[i])} and "
                             f"rhoINJECT = {cxx(rho_inject[i])} but npcelx * npcely * "
                             f"npcelz = 0: it would load or inject no particles")
    if exosphere_enabled and ns_planetary > 0 and npcel[0] == 0:
        raise SystemExit("AddExosphereInjection = 1 sizes the injected particles by "
                         "npcelx * npcely * npcelz of species 0, which is 0: it would "
                         "inject none")
    track_id = track_particle_id_mask(cfg, ns)
    heat_flux = heat_flux_requested(cfg)
    restart_output_cycle = cfg.integer("RestartOutputCycle", 5000)

    def bc_em(face):
        code = cfg.integer(f"bcEMface{face}")
        if not 0 <= code <= 2:
            raise SystemExit(f"bcEMface{face} must be 0 (perfect conductor), 1 (perfect magnetic "
                             f"conductor) or 2 (open), not {code}")
        return code
    bc_em_xr, bc_em_xl, bc_em_yr, bc_em_yl, bc_em_zr, bc_em_zl = (
        bc_em(face) for face in ("Xright", "Xleft", "Yright", "Yleft", "Zright", "Zleft"))
    yes_sal = cfg.integer("yes_sal", 1)
    sal_tangential_b = cfg.integer("sal_tangential_B", 1)
    n_layers_sal = cfg.integer("n_layers_sal", 3)

    def bc_p(face):
        code = cfg.integer(f"bcPface{face}", 1)
        if not 0 <= code <= 3:
            raise SystemExit(f"bcPface{face} must be 0 (exit), 1 (mirror), 2 (open inflow) "
                             f"or 3 (open outflow), not {code}")
        return code
    bc_p_xr, bc_p_xl, bc_p_yr, bc_p_yl, bc_p_zr, bc_p_zl = (
        bc_p(face) for face in ("Xright", "Xleft", "Yright", "Yleft", "Zright", "Zleft"))
    outflow_layer_fields = cfg.string("OutflowLayerFields", "solved")
    if outflow_layer_fields not in ("copy", "solved"):
        raise SystemExit(f"OutflowLayerFields must be copy or solved, not {outflow_layer_fields}")
    apply_inflow = cfg.integer("ApplyInflowBcsEImage", 1)
    sorting_cycle = cfg.integer("SortingCycle", 0)

    dx = lx / nxc
    dy = ly / nyc
    dz = lz / nzc

    print_topology(xlen, ylen, zlen, periodic_x, periodic_y, periodic_z)

    print()
    print("Simulation Parameters")
    print("---------------------")
    print(f"Number of species    = {ns}")
    for i in range(ns):
        print(f"qom[{i}] = {cxx(qom[i])}")
    # as Collective::Print: a qom of 0 is allowed, warned of
    for i in range(ns + ns_test):
        if qom[i] != 0.0:
            continue
        if i < ns:
            print(f"WARNING: qom of species {i} is 0: its charge sign and mass (q/qom) "
                  f"are not defined: its particles, or a charge held fixed for it "
                  f"(a Dipole or Dipole2D planet, the open faces of a TaylorGreen "
                  f"restart), make the field solve non-finite; with neither, its "
                  f"bulk-flow energy in ConservedQuantities.txt is nan")
        else:
            print(f"WARNING: qom of species {i} (a test-particle species) is 0: the "
                  f"charge of its particles is not defined (nan in the serial-HDF5 checkpoints)")
    print(f"x-Length                 = {cxx(lx)}")
    print(f"y-Length                 = {cxx(ly)}")
    print(f"z-Length                 = {cxx(lz)}")
    print(f"Number of cells (x)      = {nxc}")
    print(f"Number of cells (y)      = {nyc}")
    print(f"Number of cells (z)      = {nzc}")
    print(f"Time step                = {cxx(dt)}")
    print(f"Number of cycles         = {ncycles}")
    print(f"Results saved in  : {save_dir}")
    print(f"Case type         : {case}")
    print(f"Simulation name   : {sim_name}")
    print(f"Maxwell solver    : {solver_type}")
    if solver_type != "GMRES":
        print(f"WARNING: SolverType = {solver_type} is experimental; GMRES is the tested Maxwell solver")
    print(f"Smoothing         : {'off' if smooth == 1.0 else 'on'} (alpha={cxx(smooth)}, Niter={smooth_niter})")
    if smooth != 1.0 and smooth_niter % 2 == 1 and smooth < smooth_flip:
        print(f"WARNING: Smooth = {cxx(smooth)} is below {cxx(smooth_flip)} on this grid: the filter's response "
              f"to the shortest grid modes is negative, so every pass flips their sign, and the odd "
              f"SmoothNiter = {smooth_niter} leaves them inverted in the smoothed E and Jhat; use an even SmoothNiter")
    print(f"Halo overlap      : {'on' if halo_overlap else 'off'} (field solver)")
    print("---------------------")
    print("EM Field Boundary Conditions")
    print("---------------------")
    print(f"Xleft  : {bc_em_xl} ({bc_em_name(bc_em_xl)})")
    print(f"Xright : {bc_em_xr} ({bc_em_name(bc_em_xr)})")
    print(f"Yleft  : {bc_em_yl} ({bc_em_name(bc_em_yl)})")
    print(f"Yright : {bc_em_yr} ({bc_em_name(bc_em_yr)})")
    print(f"Zleft  : {bc_em_zl} ({bc_em_name(bc_em_zl)})")
    print(f"Zright : {bc_em_zr} ({bc_em_name(bc_em_zr)})")
    sal = "yes" if yes_sal else "no"
    print(f"SAL (absorbing layer): {sal}" + (f", n_layers={n_layers_sal}" if yes_sal else ""))
    # same warning as Collective::Print: the open-face B treatment skips wall
    # ranks with at most 8 cells along the face normal (meant for flat 2D
    # directions); the last rank along an axis takes the remainder
    for face, bc, periodic, cells, nlen, right, axis in (
            ("Xleft", bc_em_xl, periodic_x, nxc, xlen, False, "x"),
            ("Xright", bc_em_xr, periodic_x, nxc, xlen, True, "x"),
            ("Yleft", bc_em_yl, periodic_y, nyc, ylen, False, "y"),
            ("Yright", bc_em_yr, periodic_y, nyc, ylen, True, "y"),
            ("Zleft", bc_em_zl, periodic_z, nzc, zlen, False, "z"),
            ("Zright", bc_em_zr, periodic_z, nzc, zlen, True, "z")):
        first = ceil_div(cells, nlen)
        local = cells - first * (nlen - 1) if right else first
        if bc == 2 and not periodic and cells > 1 and local <= 8:
            print(f"WARNING: open face {face} has {local} cells per rank along "
                  f"{axis}: the open-face B treatment needs at least 9 and is "
                  f"skipped; use fewer ranks along {axis} (at least 9 cells per rank on that face)")
    print("---------------------")
    print("Particle Boundary Conditions")
    print("---------------------")
    print(f"Xleft  : {bc_p_xl} ({bc_p_name(bc_p_xl)})")
    print(f"Xright : {bc_p_xr} ({bc_p_name(bc_p_xr)})")
    print(f"Yleft  : {bc_p_yl} ({bc_p_name(bc_p_yl)})")
    print(f"Yright : {bc_p_yr} ({bc_p_name(bc_p_yr)})")
    print(f"Zleft  : {bc_p_zl} ({bc_p_name(bc_p_zl)})")
    print(f"Zright : {bc_p_zr} ({bc_p_name(bc_p_zr)})")
    print(f"E inflow BCs in Maxwell image : {'yes' if apply_inflow else 'no'} (on faces with bcEMface = 2 and bcPface = 2, not X-right)")
    force_wall_b = cfg.integer("ForceWallB", 0)
    print(f"B forced on the y walls (ForceWallB) : {'yes (GEM/ForceFree equilibrium layers)' if force_wall_b else 'no'}")
    print("---------------------")
    print("Field Corrections")
    print("---------------------")
    line = f"Poisson div(E) correction  : {poisson}"
    if poisson == "yes":
        line += f", every {poisson_cycle} cycles (in gpuCalculateE)"
    print(line)
    line = f"div(B) cleaning            : {divb}"
    if divb == "yes":
        line += f", every {divb_cycle} cycles (in gpuCalculateB)"
    print(line)
    if divb == "yes" and yes_sal and not sal_tangential_b:
        print("WARNING: full B SAL with the global div(B) cleaning: the cleaning "
              "carries the SAL's B_n = B0_n at the faces into the whole box; use "
              "sal_tangential_B = 1")
    if divb == "yes" and not yes_sal:
        # without the SAL these open faces set the whole B to B0: X-left, and
        # the Y/Z inflow faces (bcPface = 2)
        b0_faces = [name for name, sets_b0 in (
            ("Xleft", not periodic_x and bc_em_xl == 2),
            ("Yleft", not periodic_y and bc_em_yl == 2 and bc_p_yl == 2),
            ("Yright", not periodic_y and bc_em_yr == 2 and bc_p_yr == 2),
            ("Zleft", not periodic_z and bc_em_zl == 2 and bc_p_zl == 2),
            ("Zright", not periodic_z and bc_em_zr == 2 and bc_p_zr == 2),
        ) if sets_b0]
        if b0_faces:
            many = len(b0_faces) > 1
            print(f"WARNING: open {', '.join(b0_faces)} "
                  f"{'faces' if many else 'face'} without SAL and the global "
                  f"div(B) cleaning: {'these faces set' if many else 'the face sets'} "
                  "the whole B, B_n included, to B0 every cycle and the cleaning "
                  "carries that face flux into the whole box; use yes_sal = 1 "
                  "with sal_tangential_B = 1")
    print("---------------------")
    print("Planet Boundary")
    print("---------------------")
    refl = " (specular)" if planet_reflection == 0 else " (diffuse, Lambert cosine law)"
    print(f"Reflection type            : {planet_reflection}{refl}")
    print("---------------------")
    print("Exosphere Ionization")
    print("---------------------")
    if exosphere_enabled and ns_planetary > 0:
        print("Status                     : enabled")
        print(f"Solar wind species         : {ns_solar} (indices 0..{ns_solar - 1})")
        print(f"Planetary species          : {ns_planetary} (indices {ns_solar}..{ns - 1})")
        print(f"Max injection radius       : {cxx(max_injection_radius)} d_i")
        print(f"Planet radius (L_square)   : {cxx(l_square)} d_i")
        for i in range(ns_planetary):
            global_idx = ns_solar + i
            print(f"  Species {global_idx} (neutral {i}):")
            print(f"    NeutralSurfaceDensity    = {cxx(neutral_density[i])} n_sw")
            print(f"    ExosphericScaleHeight    = {cxx(scale_height[i])} d_i")
            print(f"    PhotoionizationFrequency = {cxx(photo_freq[i])} wci")
            print(f"    MacroParticleWeightRatio = {cxx(weight_ratio[i])}")
            print(f"    qom                      = {cxx(qom[global_idx])}")
            print(f"    uth/vth/wth              = {cxx(uth[global_idx])} / {cxx(vth[global_idx])} / {cxx(wth[global_idx])}")
    else:
        print("Status                     : disabled")

    print("---------------------")
    print("Check Simulation Constraints")
    print("---------------------")
    print("Accuracy Constraint:  ")
    for i in range(ns):
        print(f"u_th < dx/dt species {i}.....", end="")
        print("OK" if uth[i] < dx / dt else "NOT SATISFIED. STOP THE SIMULATION.")
        print(f"v_th < dy/dt species {i}......", end="")
        print("OK" if vth[i] < dy / dt else "NOT SATISFIED. STOP THE SIMULATION.")

    print()
    print("Finite Grid Stability Constraint:  ")
    for i in range(ns):
        ux = uth[i] * dt / dx
        vy = vth[i] * dt / dy
        if ux > 0.1:
            print(f"OK u_th*dt/dx (species {i}) = {cxx(ux)} > .1")
        else:
            print(f"WARNING. u_th*dt/dx (species {i}) = {cxx(ux)} < .1")
        if vy > 0.1:
            print(f"OK v_th*dt/dy (species {i}) = {cxx(vy)} > .1")
        else:
            print(f"WARNING. v_th*dt/dy (species {i}) = {cxx(vy)} < .1")

    print()
    print("CFL Condition (c*dt/dx < 1):  ")
    print("---------------------")
    print(f"Speed of light c     = {cxx(c)}")
    print(f"Time step dt         = {cxx(dt)}")
    print(f"Grid spacing dx      = {cxx(dx)}")
    print(f"Grid spacing dy      = {cxx(dy)}")
    print(f"Grid spacing dz      = {cxx(dz)}")
    cfl_x = c * dt / dx
    cfl_y = c * dt / dy
    cfl_z = c * dt / dz
    print(f"c*dt/dx              = {cxx(cfl_x)}" + ("  OK" if cfl_x < 1.0 else "  WARNING: CFL VIOLATED!"))
    print(f"c*dt/dy              = {cxx(cfl_y)}" + ("  OK" if cfl_y < 1.0 else "  WARNING: CFL VIOLATED!"))
    if nzc > 1:
        print(f"c*dt/dz              = {cxx(cfl_z)}" + ("  OK" if cfl_z < 1.0 else "  WARNING: CFL VIOLATED!"))
    cfl_multi = c * dt * math.sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + (1.0 / (dz * dz) if nzc > 1 else 0.0))
    print(f"c*dt*|1/dx|          = {cxx(cfl_multi)}" + ("  OK" if cfl_multi < 1.0 else "  WARNING: multi-dim CFL VIOLATED!"))

    print()
    print("Numerical Resolution Parameters:  ")
    print("---------------------")
    b0 = math.sqrt(b0x * b0x + b0y * b0y + b0z * b0z)
    if b0 > 0.0:
        for i in range(ns):
            omega_c_dt = abs(qom[i]) * b0 / c * dt if c != 0 else math.inf
            r_l = uth[i] * c / (abs(qom[i]) * b0) if qom[i] != 0 and b0 != 0 else math.inf
            print(f"Larmor radius / dx (species {i}) = {cxx(r_l / dx)}  (rL = {cxx(r_l)}, Omega_c*dt = {cxx(omega_c_dt)})")
    for i in range(ns):
        omega_arg = abs(qom[i]) * rho_init[i]
        omega_p = math.sqrt(omega_arg) if omega_arg >= 0.0 else math.nan
        skin_depth = c / omega_p if omega_p != 0.0 else math.inf
        omega_p_dt = omega_p * dt
        suffix = "  OK" if omega_p_dt < 2.0 else "  WARNING: plasma oscillations under-resolved!"
        print(f"Species {i} (qom={cxx(qom[i])}): omega_p = {cxx(omega_p)}, d_s/dx = {cxx(skin_depth / dx)}, omega_p*dt = {cxx(omega_p_dt)}{suffix}")

    print()
    print("Estimated Memory Per Rank")
    print("---------------------")
    nxc_loc = ceil_div(nxc, xlen) + 2
    nyc_loc = ceil_div(nyc, ylen) + 2
    nzc_loc = ceil_div(nzc, zlen) + 2
    nxn_loc = nxc_loc + 1
    nyn_loc = nyc_loc + 1
    nzn_loc = nzc_loc + 1
    grid_n = nxn_loc * nyn_loc * nzn_loc
    grid_c = nxc_loc * nyc_loc * nzc_loc
    field_size = nzn_loc * (nyn_loc - 1) * (nxn_loc - 1)
    mb = 1024.0 * 1024.0

    print(f"Local grid (with ghosts): {nxc_loc}x{nyc_loc}x{nzc_loc} cells, {nxn_loc}x{nyn_loc}x{nzn_loc} nodes")

    nxc_r = nxc_loc - 2
    nyc_r = nyc_loc - 2
    nzc_r = nzc_loc - 2

    int_c = float(nxc_r) * nyc_r * nzc_r

    # Particle SoA arrays (host and device): x, y, z, u, v, w, q, and the ID
    # of the species with TrackParticleID, 8 bytes each
    soa_arrays = [7.0 + (1.0 if track_id[i] else 0.0) for i in range(ns)]

    # EMfields3D (its constructor), as in Collective::Print: 19 node arrays
    # and 4 cell arrays; per species the 10 node moments and the cell density.
    host_emf = (19.0 * grid_n + 4.0 * grid_c + ns * (10.0 * grid_n + grid_c)) * 8.0
    # The heat-flux arrays are always allocated but resident only when used
    host_heat_flux = ns * HEAT_FLUX_COMPONENTS * grid_n * 8.0 if heat_flux else 0.0
    # Restart cell metadata when checkpoints are written: offsets and counts
    # (int) of the guarded and of the active cells, per species
    host_restart = ns * 2.0 * (grid_c + int_c) * 4.0 if restart_output_cycle > 0 else 0.0
    host_pcl = 0.0
    host_comm = 0.0
    for i in range(ns):
        nop_i = npcel[i] * nxc_r * nyc_r * nzc_r
        host_pcl += nop_i * soa_arrays[i] * 8.0
        host_comm += 0.1 * nop_i * 64.0
    host_total = (host_emf + host_heat_flux + host_restart + host_pcl + host_comm) / mb

    line = (f"HOST:   EMfields = {cxx(host_emf / mb)} MB, particles = {cxx(host_pcl / mb)} MB, "
            f"comm = {cxx(host_comm / mb)} MB")
    if host_heat_flux > 0:
        line += f", heat flux = {cxx(host_heat_flux / mb)} MB"
    if host_restart > 0:
        line += f", restart metadata = {cxx(host_restart / mb)} MB"
    print(line)
    print(f"        TOTAL = {cxx(host_total)} MB ({cxx(host_total / 1024.0)} GB)  (+ ~250 MB runtime and libraries)")

    cap_factor = PARTICLE_ARRAY_HEADROOM
    aux_frac = 0.1
    planet_frac = 0.05
    is_dipole = case in {"Dipole", "Dipole2D"}
    dev_field_buf = field_size * 24.0 * 8.0
    dev_moments = ns * grid_n * 10.0 * 8.0
    dev_pcl = 0.0
    dev_buf = 0.0
    dev_sort = 0.0
    dev_planet = 0.0
    for i in range(ns):
        nop_i = npcel[i] * nxc_r * nyc_r * nzc_r
        cap_i = int(nop_i * cap_factor)
        dev_pcl += cap_i * soa_arrays[i] * 8.0
        dev_buf += cap_i * 8.0 + aux_frac * nop_i * (64.0 + 4.0 + 64.0)
        if sorting_cycle > 0:
            dev_sort += 3.0 * grid_c * 4.0 + cap_i * (4.0 + 8.0)
        if is_dipole:
            dev_planet += planet_frac * nop_i * 64.0
    # GPU field solver (EMfields3D::gpuSolverAllocate), as in Collective::Print:
    # 35 node arrays (+1 PSI with the div(B) cleaning, +3 for the smoothed
    # E(n+theta) when E is smoothed), the 10 node moments of each species, 16
    # cell arrays, the Krylov vectors, the GMRES basis (GMRES_M + 1) and w, the
    # lazy FGMRES basis Z and Chebyshev-solver / block-Jacobi vectors, the two
    # Chebyshev iterates of the cleaning preconditioner, and the halo buffers
    # (64 fields per direction).
    gmres_m = 20.0
    halo_max_batch = 64.0
    smooth_e = smooth != 1.0 and smooth_niter > 0  # EMfields3D::smoothingOn()
    int_n = float(nxn_loc - 2) * (nyn_loc - 2) * (nzn_loc - 2)
    n_maxwell = 3.0 * int_n
    n_krylov = n_maxwell  # the longest Krylov vectors
    div_e, div_b = poisson == "yes", divb == "yes"
    cleaning = div_e or div_b
    # the div(B) rows: the interior nodes, one row plane along a flat axis
    int_b = (float(nxc_r if nxc == 1 and periodic_x else nxn_loc - 2)
             * (nyc_r if nyc == 1 and periodic_y else nyn_loc - 2)
             * (nzc_r if nzc == 1 and periodic_z else nzn_loc - 2))
    dev_solver = ((35.0 + (1.0 if div_b else 0.0) + (3.0 if smooth_e else 0.0)
                   + 10.0 * ns) * grid_n
                  + 16.0 * grid_c + 2.0 * n_maxwell
                  + (2.0 * int_c if div_e else 0.0) + (2.0 * int_b if div_b else 0.0)
                  + (gmres_m + 2.0) * n_krylov)
    if cleaning or solver_type == "FGMRESBlockJacobi":
        dev_solver += gmres_m * n_krylov
    if solver_type == "Chebyshev":
        dev_solver += 4.0 * n_maxwell
    if cleaning:  # the two Chebyshev iterates of the cleaning preconditioner
        dev_solver += 2.0 * max(int_c if div_e else 0.0, int_b if div_b else 0.0)
    if solver_type == "FGMRESBlockJacobi":
        dev_solver += 2.0 * n_maxwell
    dev_solver += 2.0 * halo_max_batch * 2.0 * (
        float(nyn_loc) * nzn_loc + float(nxn_loc) * nzn_loc + float(nxn_loc) * nyn_loc)
    dev_solver *= 8.0
    # Heat flux (c_Solver::initCUDA): components and 4 bulk moments per species
    dev_heat_flux = ns * (HEAT_FLUX_COMPONENTS + 4.0) * grid_n * 8.0 if heat_flux else 0.0
    dev_total = (dev_field_buf + dev_moments + dev_pcl + dev_buf + dev_sort + dev_planet
                 + dev_solver + dev_heat_flux) / mb
    line = (f"DEVICE: particles = {cxx(dev_pcl / mb)} MB, buffers = {cxx(dev_buf / mb)} MB, "
            f"moments = {cxx(dev_moments / mb)} MB, field = {cxx(dev_field_buf / mb)} MB, "
            f"field solver = {cxx(dev_solver / mb)} MB")
    if dev_sort > 0:
        line += f", sort = {cxx(dev_sort / mb)} MB"
    if dev_planet > 0:
        line += f", planet = {cxx(dev_planet / mb)} MB"
    if dev_heat_flux > 0:
        line += f", heat flux = {cxx(dev_heat_flux / mb)} MB"
    print(line)
    print(f"        TOTAL = {cxx(dev_total)} MB ({cxx(dev_total / 1024.0)} GB)  (+ 500 MB CUDA context)")
    combined = host_total + dev_total
    print(f"COMBINED = {cxx(combined)} MB ({cxx(combined / 1024.0)} GB)")
    print("---------------------")

    _ = th, delta


def usage() -> None:
    print(f"usage: {Path(sys.argv[0]).name} INPUT.inp", file=sys.stderr)


def main(argv) -> int:
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        usage()
        return 0 if len(argv) == 2 else 2
    path = Path(argv[1])
    print_report(Input(parse_config(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
