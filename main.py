"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, five
quantiles of its earn rate on Exophase at the case's target, from the case's
history of reads of Exophase's trophy page.

Method (standard library only, deterministic, no network):
  1. Collapse the history into refresh states. Exophase refreshes its counts
     on its own schedule, not with every read, so consecutive identical reads
     are one state; a state is (time, players tracked, rates).
  2. Forecast the players-tracked count at the target with an ensemble of
     three growth models -- a power law in time since release, an
     exponentially decaying inflow, and a damped log-linear trend -- and take
     their median.
  3. For each trophy, forecast its rate at the target. The candidates are:
     V0 persistence (the last rate), V1 logit(rate) linear in log10(players),
     V2 V1 plus a linear time term (ridge), and V3 the house curve baseline
     (logit linear in ln(1 + hours) over the last 24 hours). The winner is
     picked per case by a rolling backtest: fit each model on the states
     before each of the last few states, score the pinball loss on that
     state, keep the lowest. With too few states, persistence is used.
     The idea behind V1/V2: a trophy's rate is earners over players tracked.
     As the tracked base explodes after launch, early trophies rise (players
     progress faster than the base dilutes them) while late ones fall to the
     floor (dilution wins); both directions show up as a slope of the rate
     against log players, which the history's refresh states trace.
     Note: an earners-growth variant and per-trophy selection were tried
     and dropped -- on synthetic worlds with known truth they lost to the
     per-case pick, because a backtest over hours cannot judge an
     extrapolation over days and the per-trophy pick overfits the tail.
  4. Quantile spreads combine a random-walk term growing with the horizon,
     each trophy's fit residual, and an extrapolation penalty for reaching
     beyond the observed player counts.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry ships one file, but keep the
# line so the layout stays standard.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's sigma).
SIGMA_RW = 0.5
# Origin the house curve baseline measures hours from.
CURVE_ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as JavaScript Math.round does it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def lstsq(rows, ys, ridge=0.0):
    """Least squares for a small design matrix. rows: list of feature lists
    (first feature should be 1.0 for an intercept). Returns coefficients or
    None when the system is singular."""
    n = len(rows[0])
    # Normal equations: (X'X + ridge*I) b = X'y (intercept unpenalised).
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, y in zip(rows, ys):
        for i in range(n):
            atb[i] += row[i] * y
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    for i in range(1, n):
        ata[i][i] += ridge
    # Gaussian elimination with partial pivoting.
    m = [ata[i] + [atb[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def pinball(y: float, q: float, tau: float) -> float:
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def mean_pinball(truth, quants):
    """Mean pinball loss over trophies (truth[i] is None where unknown)."""
    total = 0.0
    count = 0
    for y, qs in zip(truth, quants):
        if y is None:
            continue
        for tau, q in zip(LEVELS, qs):
            total += pinball(y, q, tau)
            count += 1
    return total / count if count else float("inf")


def collapse_states(history):
    """Merge consecutive identical reads into refresh states.

    Returns a list of dicts {t, n, rates, earned}, oldest first, where t is
    the first time the state was seen."""
    states = []
    for read in history:
        rates = tuple(read["rates"])
        n = read["players_tracked"]
        earned = read.get("trophies_earned")
        if states and states[-1]["n"] == n and states[-1]["rates"] == rates:
            continue
        states.append(
            {"t": parse_time(read["at"]), "n": n, "rates": rates, "earned": earned}
        )
    return states


def forecast_players(states, target_t, release_t):
    """Forecast players tracked at the target: median of three growth fits."""
    last = states[-1]
    n_last = last["n"]
    t_last = last["t"]
    horizon_h = (target_t - t_last).total_seconds() / 3600.0
    if horizon_h <= 0 or len(states) < 2:
        return float(n_last)
    preds = []
    # M1: power law, log10(N) ~ a + b*log10(hours since release + 1).
    xs, ys = [], []
    for s in states:
        h = (s["t"] - release_t).total_seconds() / 3600.0 + 1.0
        if h > 0 and s["n"] > 0:
            xs.append(math.log10(h))
            ys.append(math.log10(s["n"]))
    if len(xs) >= 2:
        coef = lstsq([[1.0, x] for x in xs], ys)
        if coef:
            h_t = (target_t - release_t).total_seconds() / 3600.0 + 1.0
            if h_t > 0:
                preds.append(10.0 ** (coef[0] + coef[1] * math.log10(h_t)))
    # M2: exponentially decaying inflow, fit log(dN/dt) linear in time.
    trans = []
    for a, b in zip(states, states[1:]):
        dt = (b["t"] - a["t"]).total_seconds() / 3600.0
        dn = b["n"] - a["n"]
        if dt > 0 and dn > 0:
            mid = (a["t"] + (b["t"] - a["t"]) / 2 - states[0]["t"]).total_seconds() / 3600.0
            trans.append((mid, math.log(dn / dt)))
    if len(trans) >= 2:
        coef = lstsq([[1.0, t] for t, _ in trans], [v for _, v in trans])
        if coef and coef[1] < 0:
            lam = -coef[1]
            t0 = (t_last - states[0]["t"]).total_seconds() / 3600.0
            r_last = math.exp(coef[0] + coef[1] * t0)
            preds.append(n_last + (r_last / lam) * (1.0 - math.exp(-lam * horizon_h)))
    # M3: damped log-linear trend from the last few states.
    m = min(5, len(states) - 1)
    a, b = states[-1 - m], states[-1]
    dt = (b["t"] - a["t"]).total_seconds() / 3600.0
    if dt > 0 and a["n"] > 0 and b["n"] > 0:
        g = (math.log(b["n"]) - math.log(a["n"])) / dt
        if g > 0:
            preds.append(n_last * math.exp(0.5 * g * horizon_h))
    if not preds:
        return float(n_last)
    preds.sort()
    med = preds[len(preds) // 2] if len(preds) % 2 else 0.5 * (preds[len(preds) // 2 - 1] + preds[len(preds) // 2])
    return min(max(med, float(n_last)), float(n_last) * 100.0)


def series_for(states, idx):
    """Per-state (log10 N, hours since curve origin, logit rate) for trophy idx."""
    out = []
    for s in states:
        r = s["rates"][idx]
        if r is None or s["n"] <= 0:
            continue
        out.append(
            (
                math.log10(s["n"]),
                (s["t"] - CURVE_ORIGIN).total_seconds() / 3600.0,
                logit(r),
            )
        )
    return out


def fit_median(series, kind, n_target, t_target):
    """Fit one candidate model on a per-trophy series and forecast the median
    rate at (n_target, t_target). Returns (median, resid_std) or None."""
    if len(series) < 3:
        return None
    if kind == "v1":
        xs = [p[0] for p in series]
        if max(xs) - min(xs) < 0.3:
            return None
        coef = lstsq([[1.0, x] for x in xs], [p[2] for p in series])
        if coef is None:
            return None
        logn_t = math.log10(n_target)
        logn_last = xs[-1]
        # Damp the slope for the part beyond the observed player range.
        eff = coef[0] + coef[1] * logn_last + coef[1] * 0.8 * (logn_t - logn_last)
        resid = [p[2] - (coef[0] + coef[1] * p[0]) for p in series]
        return logistic(eff), _std(resid)
    if kind == "v2":
        xs = [p[0] for p in series]
        if max(xs) - min(xs) < 0.3:
            return None
        rows = [[1.0, p[0], p[1] / 24.0] for p in series]
        coef = lstsq(rows, [p[2] for p in series], ridge=1e-6)
        if coef is None:
            return None
        logn_t = math.log10(n_target)
        logn_last = xs[-1]
        days_t = (t_target - CURVE_ORIGIN).total_seconds() / 86400.0
        eff = (
            coef[0]
            + coef[1] * logn_last
            + coef[1] * 0.8 * (logn_t - logn_last)
            + coef[2] * days_t
        )
        resid = [
            p[2] - (coef[0] + coef[1] * p[0] + coef[2] * (p[1] / 24.0))
            for p in series
        ]
        return logistic(eff), _std(resid)
    if kind == "v3":
        # House curve baseline: logit ~ a + b*ln(1 + hours), last 24 h only.
        recent = [p for p in series if p[1] >= series[-1][1] - 24.0]
        if len(recent) < 3:
            recent = series
        if len(recent) < 3:
            return None
        xs = [math.log1p(max(0.0, p[1])) for p in recent]
        if max(xs) - min(xs) < 1e-9:
            return None
        coef = lstsq([[1.0, x] for x in xs], [p[2] for p in recent])
        if coef is None:
            return None
        h_t = max(0.0, (t_target - CURVE_ORIGIN).total_seconds() / 3600.0)
        resid = [p[2] - (coef[0] + coef[1] * math.log1p(max(0.0, p[1]))) for p in recent]
        return logistic(coef[0] + coef[1] * math.log1p(h_t)), _std(resid)
    return None


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def spread_for(kind, resid_std, horizon_days, logn_gap):
    base = SIGMA_RW * math.sqrt(max(0.0, horizon_days))
    if kind == "v0":
        s = base + 0.35
    else:
        s = math.sqrt(base * base + (1.5 * resid_std) ** 2)
        s += 0.15 * max(0.0, logn_gap)
    return min(max(s, 0.15), 3.0)


def quantiles(median, spread):
    lg = logit(median)
    qs = [round4(min(100.0, max(0.0, logistic(lg + z * spread)))) for z in Z]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    target_t = parse_time(case["target"])
    at_t = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    release_raw = (case.get("game") or {}).get("release_date")
    try:
        release_t = parse_time(release_raw) if release_raw else parse_time(history[0]["at"])
    except Exception:
        release_t = parse_time(history[0]["at"])
    states = collapse_states(history)
    last = states[-1]
    n_target = forecast_players(states, target_t, release_t)
    logn_gap = math.log10(n_target) - math.log10(last["n"]) if last["n"] > 0 else 0.0

    # Pick the model variant by rolling backtest on the tail states.
    kinds = ["v0", "v1", "v2", "v3"]
    winner = "v0"
    if len(states) >= 6:
        h = min(3, len(states) // 3)
        val_states = states[-h:]
        fit_states = states[:-h]
        scores = {}
        for kind in kinds:
            total, cnt = 0.0, 0
            for vs in val_states:
                before = [s for s in fit_states if s["t"] < vs["t"]]
                if len(before) < 3:
                    before = fit_states
                n_v = forecast_players(before, vs["t"], release_t) if len(before) >= 2 else vs["n"]
                qs = []
                truth = []
                for i in range(len(trophies)):
                    ser = series_for(before, i)
                    got = fit_median(ser, kind, n_v, vs["t"]) if kind != "v0" else None
                    if got is None:
                        med = before[-1]["rates"][i]
                        med = med if med is not None else 0.0
                        sp = spread_for("v0", 0.0, 0.0, 0.0)
                    else:
                        med, rs = got
                        sp = spread_for(kind, rs, 0.0, 0.0)
                    qs.append(quantiles(med, sp))
                    truth.append(vs["rates"][i])
                lp = mean_pinball(truth, qs)
                if lp != float("inf"):
                    total += lp
                    cnt += 1
            scores[kind] = total / cnt if cnt else float("inf")
        order = {"v0": 0, "v1": 1, "v3": 2, "v2": 3}
        winner = min(kinds, key=lambda k: (scores[k], order[k]))

    rows = []
    for i, trophy in enumerate(trophies):
        ser = series_for(states, i)
        last_rate = last["rates"][i]
        if winner == "v0" or not ser:
            med = last_rate if last_rate is not None else 0.0
            sp = spread_for("v0", 0.0, horizon_days, logn_gap)
        else:
            got = fit_median(ser, winner, n_target, target_t)
            if got is None:
                med = last_rate if last_rate is not None else 0.0
                sp = spread_for("v0", 0.0, horizon_days, logn_gap)
            else:
                med, rs = got
                sp = spread_for(winner, rs, horizon_days, logn_gap)
        rows.append({"id": trophy["id"], "q": quantiles(med, sp)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
