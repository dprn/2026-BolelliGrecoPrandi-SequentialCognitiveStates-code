"""Write numbers.tex, one LaTeX macro per computed number quoted in the paper.

Reads only the files in results/ and computes nothing:

  results/lv-training.json   periods, return errors, projection errors and
                             display windows (train_lv.py)
  results/lv-verify.json     errors on K and near Gamma, saddles, plane
                             condition, initial conditions, Floquet
                             multipliers (verify_lv.py)
  results/ring-results.json  threshold c* and the numbers of Figure S1.1
                             (ring_cycle.py)

Macro names use letters only.  Section 5 macros start with \\lv and end with
Sym or Asym for the symmetric and asymmetric configurations, the saddles are
numbered One, Two, Three, and the macros of the cyclic-inhibition example
start with \\ring.  Every value is wrapped in \\ensuremath, so a macro works
both in text and in math mode, and is printed the way the paper prints it,
for instance 16.73 or 1.02\\times10^{-13}.

Usage (from the repository root):

  python3 make_numbers.py
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUTPUT = HERE / "numbers.tex"

CONFIGS = {"symmetric": "Sym", "asymmetric": "Asym"}
ORDINALS = ["One", "Two", "Three"]
C_NAMES = {2.1: "TwoPointOne", 5.0: "Five", 10.0: "Ten", 40.0: "Forty"}


# ---------------------------------------------------------------------------
# Formatting

def sci(x, sig=3):
    """x as m\\times10^{e} with sig significant digits."""
    mantissa, exponent = f"{x:.{sig - 1}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def fixed(x, decimals):
    return f"{x:.{decimals}f}"


def sigfig(x, sig=2):
    """sig significant digits, in decimal form between 1e-3 and 1e4."""
    if x == 0:
        return "0"
    if not 1e-3 <= abs(x) < 1e4:
        return sci(x, sig)
    rounded = float(f"{x:.{sig - 1}e}")
    decimals = max(sig - 1 - math.floor(math.log10(abs(rounded))), 0)
    return f"{rounded:.{decimals}f}"


def integer(n):
    return str(int(n))


def complex_number(re, im, fmt):
    if abs(im) < 1e-12 * max(abs(re), 1.0):
        return fmt(re)
    sign = "+" if im > 0 else "-"
    return f"{fmt(re)}{sign}{fmt(abs(im))}\\,i"


# ---------------------------------------------------------------------------
# Macros

def lv_macros(training, verify):
    out = []
    for key, suffix in CONFIGS.items():
        label = key
        period = training[key]["period"]
        out.append((None, f"Section 5, {label} configuration"))
        out += [
            (f"lvPeriod{suffix}", fixed(period["T"], 2),
             "period of the periodic orbit of f_theta"),
            (f"lvPeriodSpread{suffix}", sci(period["relative_period_spread"]),
             "relative spread of the last five return times"),
            (f"lvReturnError{suffix}", sci(period["relative_return_error"]),
             "max relative distance between successive crossing states"),
            (f"lvProjError{suffix}", sci(period["projection_error"]),
             "max |P y(t) - x_theta(t)| on [0, 500]"),
            (f"lvWindowStart{suffix}", fixed(period["display_window"][0], 1),
             "start of the window shown in the population figure"),
            (f"lvWindowEnd{suffix}", f"{period['display_window'][1]:g}",
             "end of that window"),
        ]

        v = verify[key]
        for tag, block in (("Test", "test_errors_K"),
                           ("Gamma", "errors_near_Gamma")):
            e = v[block]
            where = ("on the test points of K" if tag == "Test"
                     else "within the radius of Gamma")
            out += [
                (f"lv{tag}FieldMax{suffix}", sigfig(e["field maximum"]),
                 f"max |f_theta - g| {where}"),
                (f"lv{tag}FieldRms{suffix}", sigfig(e["field RMS"]),
                 f"RMS of |f_theta - g| {where}"),
                (f"lv{tag}JacMax{suffix}", sigfig(e["Jacobian maximum"]),
                 f"max |Df_theta - Dg|_F {where}"),
                (f"lv{tag}JacRms{suffix}", sigfig(e["Jacobian RMS"]),
                 f"RMS of |Df_theta - Dg|_F {where}"),
            ]
        out.append((f"lvGammaPoints{suffix}",
                    integer(v["errors_near_Gamma"]["points_near_Gamma"]),
                    "sample points of K within the radius of Gamma"))

        for i, row in enumerate(v["equilibria"]):
            eqs = row["f_theta_equilibria"]
            if not eqs:
                raise RuntimeError(f"no equilibrium of f_theta near {row}")
            eq = min(eqs, key=lambda q: q["distance_to_e_i"])
            n = ORDINALS[i]
            out.append((f"lvSaddle{n}Count{suffix}", integer(len(eqs)),
                        f"equilibria of f_theta found near e_{i + 1}"))
            out.append((f"lvSaddle{n}Dist{suffix}",
                        sigfig(eq["distance_to_e_i"]),
                        f"distance of that equilibrium to e_{i + 1}"))
            names = ("Radial", "Stable", "Unstable")
            for name, (re, im) in zip(names, eq["eigenvalues"]):
                out.append((f"lvSaddle{n}{name}{suffix}",
                            complex_number(re, im, lambda x: fixed(x, 3)),
                            f"eigenvalue of Df_theta there, next to "
                            f"the {name.lower()} one of Dg(e_{i + 1})"))

        out.append((f"lvSaddleDistMax{suffix}",
                    sigfig(max(min(q["distance_to_e_i"]
                                   for q in row["f_theta_equilibria"])
                               for row in v["equilibria"])),
                    "largest of the three distances above"))

        planes = v["plane_condition"]
        for k, row in enumerate(planes):
            n = ORDINALS[k]
            out += [
                (f"lvPlane{n}Min{suffix}", sigfig(row["min"]),
                 f"min of (f_theta)_{k + 1} on the grid of x_{k + 1} = 0"),
                (f"lvPlane{n}Frac{suffix}",
                 sigfig(row["fraction_nonpositive"]),
                 f"fraction of that grid where (f_theta)_{k + 1} <= 0"),
            ]
        out.append((f"lvPlaneMin{suffix}",
                    sigfig(min(row["min"] for row in planes)),
                    "min over the three faces"))

        s = v["initial_conditions"]
        out += [
            (f"lvSweepPeriodic{suffix}", integer(s["periodic"]),
             "initial conditions converging to a periodic orbit"),
            (f"lvSweepPeriodMin{suffix}", fixed(s["period_min"], 5),
             "smallest period over the initial conditions"),
            (f"lvSweepPeriodMax{suffix}", fixed(s["period_max"], 5),
             "largest period over the initial conditions"),
            (f"lvSweepPeriodRange{suffix}", sci(s["relative_period_range"]),
             "(max - min) / reference period"),
            (f"lvSweepDistance{suffix}", sci(s["max_tail_distance"]),
             "max distance of the tails to the reference orbit"),
        ]

        f = v["floquet"]
        formats = (lambda x: fixed(x, 4), lambda x: sigfig(x, 3),
                   lambda x: sigfig(x, 3))
        for name, (re, im), fmt in zip(("Trivial", "Second", "Third"),
                                       f["multipliers"], formats):
            out.append((f"lvFloquet{name}{suffix}",
                        complex_number(re, im, fmt),
                        "Floquet multiplier, by decreasing modulus"))
        out.append((f"lvFloquetClosure{suffix}", sci(f["closure_error"], 2),
                    "|x(T) - x(0)| for the variational integration"))

    out.append((None, "Section 5, common settings"))
    first = verify[next(iter(CONFIGS))]
    out += [
        ("lvTestPoints", integer(first["test_errors_K"]["points"]),
         "test points in K"),
        ("lvGammaRadius", f"{first['errors_near_Gamma']['radius']:g}",
         "radius of the neighborhood of Gamma"),
        ("lvPlaneGrid", integer(first["plane_condition"][0]["grid"][0]),
         "grid points per side on each face"),
        ("lvSweepCount", integer(first["initial_conditions"]["count"]),
         "initial conditions in the sweep"),
        ("lvSweepTailStart", f"{first['initial_conditions']['tail_start']:g}",
         "start of the tail compared with the reference orbit"),
        ("lvSweepLow", f"{min(min(r['x0']) for r in first['initial_conditions']['runs']):g}",
         "smallest coordinate of the initial conditions"),
        ("lvSweepHigh", f"{max(max(r['x0']) for r in first['initial_conditions']['runs']):g}",
         "largest coordinate of the initial conditions"),
    ]
    return out


def ring_macros(ring):
    out = [(None, "Cyclic inhibition, Figure S1.1"),
           ("ringCstar", fixed(ring["c_star"], 4), "threshold c*")]
    for row in ring["orbit"]:
        n = C_NAMES[row["c"]]
        out += [
            (f"ringRangeAt{n}", fixed(row["peak_to_peak"], 3),
             f"peak-to-peak range of the orbit at c = {row['c']:g}"),
            (f"ringPeriodAt{n}", fixed(row["period"], 2),
             f"period of the orbit at c = {row['c']:g}"),
        ]
    counts = {row["initial_conditions"] for row in ring["spread"]}
    if len(counts) != 1:
        raise RuntimeError("different numbers of initial conditions")
    out += [
        ("ringSweepCount", integer(counts.pop()),
         "initial conditions at each of c = 2.1, 5, 10"),
        ("ringSweepSpread",
         sci(max(row["extreme_value_spread"] for row in ring["spread"]), 2),
         "max spread of the extreme values of x_1 over them"),
    ]
    return out


def main():
    def load(name):
        with open(RESULTS_DIR / name) as fh:
            return json.load(fh)

    entries = (lv_macros(load("lv-training.json"), load("lv-verify.json"))
               + ring_macros(load("ring-results.json")))

    names = [e[0] for e in entries if e[0] is not None]
    assert len(names) == len(set(names)), "duplicate macro name"
    assert all(n.isalpha() for n in names), "macro names must be letters"

    lines = [
        "% numbers.tex: generated by make_numbers.py from results/*.json.",
        "% Do not edit by hand: rerun `make numbers` instead.",
        "% Each macro expands to \\ensuremath{value}, so it works in text and",
        "% in math mode.  Sym and Asym refer to the symmetric and asymmetric",
        "% configurations of Section 5.",
    ]
    for entry in entries:
        if entry[0] is None:
            lines += ["", f"% {entry[1]}"]
            continue
        name, value, comment = entry
        lines.append(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{value}}}}}"
                     f"  % {comment}")
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT.relative_to(HERE)} ({len(names)} macros)")


if __name__ == "__main__":
    main()
