#!/usr/bin/env python3
"""Print iPIC3D input-derived startup checks without running MPI/CUDA."""


import math
import os
import re
import sys
from pathlib import Path


HEAT_FLUX_COMPONENTS = 10


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

    def array_real(self, key: str, n: int, default=None):
        raw = self.raw(key, default)
        vals = [float(v) for v in str(raw).replace("\n", " ").split()]
        if len(vals) < n:
            raise SystemExit(f"{key} has {len(vals)} values, needs at least {n}")
        return vals[:n]

    def array_int(self, key: str, n: int, default=None):
        raw = self.raw(key, default)
        vals = [int(float(v)) for v in str(raw).replace("\n", " ").split()]
        if len(vals) < n:
            raise SystemExit(f"{key} has {len(vals)} values, needs at least {n}")
        return vals[:n]


def bc_em_name(code: int) -> str:
    return {
        0: "perfect conductor",
        1: "Dirichlet (first order)",
        2: "open/inflow (Neumann)",
    }.get(code, "unknown")


def bc_p_name(code: int) -> str:
    return {
        0: "exit",
        1: "perfect mirror",
        2: "reemission",
        3: "open BC outflow",
        4: "open BC inflow",
    }.get(code, "unknown")


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
    dt = cfg.real("dt")
    ncycles = cfg.integer("ncycles")
    th = cfg.real("th", 1.0)
    smooth = cfg.real("Smooth", 1.0)
    smooth_niter = cfg.integer("SmoothNiter", 6)
    save_dir = cfg.string("SaveDirName", "data")
    ns = cfg.integer("ns")

    b0x = cfg.real("B0x", 0.0)
    b0y = cfg.real("B0y", 0.0)
    b0z = cfg.real("B0z", 0.0)
    delta = cfg.real("delta", 0.5)

    case = cfg.string("Case")
    sim_name = cfg.string("SimulationName")
    poisson = cfg.string("PoissonCorrection", "no")
    poisson_cycle = cfg.integer("PoissonCorrectionCycle", 10)
    divb = cfg.string("divBCorrection", "no")
    divb_cycle = cfg.integer("divBCorrectionCycle", 10)

    c = cfg.real("c", 1.0)
    lx = cfg.real("Lx", 10.0)
    ly = cfg.real("Ly", 10.0)
    lz = cfg.real("Lz", 10.0)
    nxc = cfg.integer("nxc", 64)
    nyc = cfg.integer("nyc", 64)
    nzc = cfg.integer("nzc", 64)
    xlen = cfg.integer("XLEN", 1)
    ylen = cfg.integer("YLEN", 1)
    zlen = cfg.integer("ZLEN", 1)
    periodic_x = cfg.boolean("PERIODICX", True)
    periodic_y = cfg.boolean("PERIODICY", True)
    periodic_z = cfg.boolean("PERIODICZ", True)

    planet_reflection = cfg.integer("planetReflectionType", 0)
    ns_solar = cfg.integer("ns_solar_wind", ns)
    ns_planetary = cfg.integer("ns_planetary", 0)
    exosphere_enabled = cfg.integer("AddExosphereInjection", 0)
    max_injection_radius = cfg.real("RmaxExosphereInjection", 3.0)
    l_square = cfg.real("L_square", 5.0)

    qom = cfg.array_real("qom", ns)
    rho_init = cfg.array_real("rhoINIT", ns)
    rho_inject = cfg.array_real("rhoINJECT", ns)
    uth = cfg.array_real("uth", ns)
    vth = cfg.array_real("vth", ns)
    wth = cfg.array_real("wth", ns)
    npcelx = cfg.array_int("npcelx", ns)
    npcely = cfg.array_int("npcely", ns)
    npcelz = cfg.array_int("npcelz", ns)
    npcel = [npcelx[i] * npcely[i] * npcelz[i] for i in range(ns)]

    bc_em_xr = cfg.integer("bcEMfaceXright")
    bc_em_xl = cfg.integer("bcEMfaceXleft")
    bc_em_yr = cfg.integer("bcEMfaceYright")
    bc_em_yl = cfg.integer("bcEMfaceYleft")
    bc_em_zr = cfg.integer("bcEMfaceZright")
    bc_em_zl = cfg.integer("bcEMfaceZleft")
    yes_sal = cfg.integer("yes_sal", 0)
    n_layers_sal = cfg.integer("n_layers_sal", 3)

    bc_p_xr = cfg.integer("bcPfaceXright", 1)
    bc_p_xl = cfg.integer("bcPfaceXleft", 1)
    bc_p_yr = cfg.integer("bcPfaceYright", 1)
    bc_p_yl = cfg.integer("bcPfaceYleft", 1)
    bc_p_zr = cfg.integer("bcPfaceZright", 1)
    bc_p_zl = cfg.integer("bcPfaceZleft", 1)
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
    print(f"Smoothing         : {'off' if smooth == 1.0 else 'on'} (alpha={cxx(smooth)}, Niter={smooth_niter})")
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
    print("---------------------")
    print("Particle Boundary Conditions")
    print("---------------------")
    print(f"Xleft  : {bc_p_xl} ({bc_p_name(bc_p_xl)})")
    print(f"Xright : {bc_p_xr} ({bc_p_name(bc_p_xr)})")
    print(f"Yleft  : {bc_p_yl} ({bc_p_name(bc_p_yl)})")
    print(f"Yright : {bc_p_yr} ({bc_p_name(bc_p_yr)})")
    print(f"Zleft  : {bc_p_zl} ({bc_p_name(bc_p_zl)})")
    print(f"Zright : {bc_p_zr} ({bc_p_name(bc_p_zr)})")
    print(f"E inflow BCs in GMRes : {'yes' if apply_inflow else 'no'} (applied per-face where bcPface == 2)")
    print("---------------------")
    print("Field Corrections")
    print("---------------------")
    line = f"Poisson div(E) correction  : {poisson}"
    if poisson == "yes":
        line += f", every {poisson_cycle} cycles (in calculateE)"
    print(line)
    line = f"div(B) cleaning            : {divb}"
    if divb == "yes":
        line += f", every {divb_cycle} cycles (in calculateB)"
    print(line)
    print("---------------------")
    print("Planet Boundary")
    print("---------------------")
    refl = " (specular)" if planet_reflection == 0 else " (diffuse/isotropic)"
    print(f"Reflection type            : {planet_reflection}{refl}")
    print("---------------------")
    print("Exosphere Ionization")
    print("---------------------")
    if exosphere_enabled and ns_planetary > 0:
        neutral_density = cfg.array_real("NeutralSurfaceDensity", ns_planetary)
        scale_height = cfg.array_real("ExosphericScaleHeight", ns_planetary)
        photo_freq = cfg.array_real("PhotoionizationFrequency", ns_planetary)
        weight_ratio = cfg.array_real("MacroParticleWeightRatio", ns_planetary)
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

    host_emf = (50.0 * grid_n + 15.0 * grid_c
                + (10.0 * ns) * grid_n + ns * grid_c
                + grid_n * 8.0 + 6.0 * 3.0 * grid_n) * 8.0
    host_field_buf = field_size * 24.0 * 8.0
    host_pcl = 0.0
    host_comm = 0.0
    for i in range(ns):
        nop_i = npcel[i] * nxc_r * nyc_r * nzc_r
        host_pcl += nop_i * 8.0 * 8.0
        host_comm += 0.1 * nop_i * 64.0
    host_total = (host_emf + host_field_buf + host_pcl + host_comm) / mb

    print(f"HOST:   EMfields = {cxx(host_emf / mb)} MB, particles = {cxx(host_pcl / mb)} MB, comm = {cxx(host_comm / mb)} MB, field buf = {cxx(host_field_buf / mb)} MB")
    print(f"        TOTAL = {cxx(host_total)} MB ({cxx(host_total / 1024.0)} GB)")

    cap_factor = 1.4
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
        dev_pcl += cap_i * 8.0 * 8.0
        dev_buf += cap_i * 8.0 + aux_frac * nop_i * (64.0 + 4.0 + 64.0)
        if sorting_cycle > 0:
            dev_sort += 3.0 * grid_c * 4.0 + cap_i * (4.0 + 8.0)
        if is_dipole:
            dev_planet += planet_frac * nop_i * 64.0
    dev_total = (dev_field_buf + dev_moments + dev_pcl + dev_buf + dev_sort + dev_planet) / mb
    line = (f"DEVICE: particles = {cxx(dev_pcl / mb)} MB, buffers = {cxx(dev_buf / mb)} MB, "
            f"moments = {cxx(dev_moments / mb)} MB, field = {cxx(dev_field_buf / mb)} MB")
    if dev_sort > 0:
        line += f", sort = {cxx(dev_sort / mb)} MB"
    if dev_planet > 0:
        line += f", planet = {cxx(dev_planet / mb)} MB"
    print(line)
    print(f"        TOTAL = {cxx(dev_total)} MB ({cxx(dev_total / 1024.0)} GB)  (+ 500 MB CUDA context)")
    combined = host_total + dev_total
    print(f"COMBINED = {cxx(combined)} MB ({cxx(combined / 1024.0)} GB)")
    print("---------------------")

    _ = th, delta, rho_inject, HEAT_FLUX_COMPONENTS


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
