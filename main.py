"""Forecast GTA VI launch-day Twitch viewers, hour by hour (gta6-twitch-ccu-forecast-01).

Method, in one paragraph: every past big launch in the case's `comparables`
traces a release-day curve, but the curves do not share one shape when they are
divided by the first hour's print -- a game whose first hour is already huge
(global unlock the night before) barely grows, while a game whose first hour is
tiny (rolling midnight, most regions still locked) can grow a hundredfold. So
for each forecast hour we fit, across the comparables, a log-linear model of
that hour's viewers on four log features of the launch itself -- its hour-1
viewers, its hour-1 live channels, its biggest pre-release hourly audience,
and its viewers twelve hours before the release day -- weighted by a Gaussian
kernel on log hour-1 viewers, so launches of a similar size count most. The
fitted value is the median; the five quantiles are a lognormal band around it
with a fixed log spread of 0.32, which local validation put near the pinball
optimum. The median is clamped to the comparables' observed range for that
hour, extended by one log unit, so a fit can never extrapolate wildly; if a
case carries no usable comparables the forecast falls back to persistence at
the hour-1 print, and if it carries no hourly history the minute reads are
aggregated into whole-hour means first. Everything is computed from the case
file at run time: no network, no stored curve, standard library only, fully
deterministic.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SPREAD = 0.32          # log spread of the quantile band (tuned on practice)
KERNEL_BW = 1.5        # Gaussian kernel bandwidth on log hour-1 viewers
FEATURES = ("lp1", "lch1", "lpremax", "lv_m12h")


def log_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0 or not math.isfinite(value):
        return None
    return math.log(value)


def parse_ts(text):
    """ISO-8601 UTC timestamp -> epoch seconds (naive, Z only)."""
    from datetime import datetime, timezone

    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def hourly_map(rows):
    """{epoch seconds of `from`: (viewers, channels)} for hourly rows."""
    out = {}
    for row in rows or []:
        ts = parse_ts(row.get("from"))
        if ts is None:
            continue
        out[ts] = (row.get("viewers"), row.get("channels"))
    return out


def nearest_value(table, ts, tol_seconds=5400):
    """Value at ts, or the closest point within tol_seconds, else None."""
    if ts in table:
        return table[ts]
    best = None
    best_gap = None
    for key, value in table.items():
        gap = abs(key - ts)
        if gap <= tol_seconds and (best_gap is None or gap < best_gap):
            best, best_gap = value, gap
    return best


def hourly_from_minutes(points, day_start_ts):
    """Aggregate minute `history` points into whole-hour means, as a fallback
    when the case carries no `history_hourly` rows."""
    from datetime import datetime, timezone

    buckets = {}
    for point in points or []:
        ts = parse_ts(point.get("at"))
        viewers = point.get("viewers")
        if ts is None or viewers is None:
            continue
        hour = int(ts // 3600) * 3600
        buckets.setdefault(hour, []).append(float(viewers))
    rows = []
    for hour in sorted(buckets):
        vals = buckets[hour]
        rows.append({
            "from": datetime.fromtimestamp(hour, tz=timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "viewers": sum(vals) / len(vals),
            "channels": None,
            "reads": len(vals),
        })
    return rows


def extract_features(hourly_rows, day_start_ts, known):
    """The four log features of a launch, from its hourly history + hour 1."""
    table = hourly_map(hourly_rows)
    feats = {}
    p1 = None
    ch1 = None
    for row in known or []:
        if row.get("hour") == 1:
            p1 = row.get("viewers")
            ch1 = row.get("channels")
            break
    feats["lp1"] = log_or_none(p1)
    feats["lch1"] = log_or_none(ch1)
    pre = [v for ts, (v, _c) in table.items()
           if day_start_ts is None or ts < day_start_ts]
    pre_vals = [v for v in pre if v]
    feats["lpremax"] = log_or_none(max(pre_vals)) if pre_vals else None
    if day_start_ts is not None:
        hit = nearest_value(table, day_start_ts - 12 * 3600)
        feats["lv_m12h"] = log_or_none(hit[0]) if hit else None
    else:
        feats["lv_m12h"] = None
    return feats


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gauss-Jordan with pivoting."""
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / pv
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def weighted_fit(rows, targets, weights):
    """Weighted least squares with intercept; rows are feature lists."""
    k = len(rows[0]) + 1
    matrix = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for feats, y, w in zip(rows, targets, weights):
        x = [1.0] + list(feats)
        for i in range(k):
            rhs[i] += w * x[i] * y
            for j in range(k):
                matrix[i][j] += w * x[i] * x[j]
    for i in range(k):
        matrix[i][i] += 1e-9
    return solve_linear(matrix, rhs)


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def forecast_case(case, name):
    day_start_ts = parse_ts(case.get("day_start"))
    known = case.get("known") or []
    hourly_rows = case.get("history_hourly") or []
    if not hourly_rows:
        hourly_rows = hourly_from_minutes(case.get("history"), day_start_ts)
    target = extract_features(hourly_rows, day_start_ts, known)
    p1 = 1.0
    for row in known:
        if row.get("hour") == 1 and row.get("viewers"):
            p1 = max(1.0, float(row["viewers"]))
            break

    # Build the training set: one row per comparable with a complete day.
    train = []
    for comp in case.get("comparables") or []:
        hourly = comp.get("hourly") or []
        table = hourly_map(hourly)
        start = parse_ts(comp.get("day_start"))
        if start is None:
            continue
        day = []
        complete = True
        for hh in range(24):
            hit = table.get(start + hh * 3600)
            if not hit or not hit[0] or float(hit[0]) <= 0:
                complete = False
                break
            day.append(float(hit[0]))
        if not complete:
            continue
        feats = extract_features(hourly, start, [])
        # The comparable's hour-1 row comes from its own hourly table.
        first = table.get(start)
        if first and first[0]:
            feats["lp1"] = log_or_none(first[0])
        if first and first[1]:
            feats["lch1"] = log_or_none(first[1])
        if feats["lp1"] is None:
            continue
        train.append((feats, day))

    # Fallback medians for any feature the target is missing.
    fallback = {}
    for key in FEATURES:
        vals = [f[key] for f, _d in train if f.get(key) is not None]
        fallback[key] = median_of(vals) if vals else 0.0
    for key in FEATURES:
        if target.get(key) is None:
            target[key] = fallback[key]

    hours = []
    for row in case.get("forecast") or []:
        h = int(row["hour"])
        median = None
        if len(train) >= 8:
            xs, ys, ws = [], [], []
            for feats, day in train:
                vec = [feats.get(k) if feats.get(k) is not None else fallback[k]
                       for k in FEATURES]
                xs.append(vec)
                ys.append(math.log(day[h - 1]))
                gap = (feats["lp1"] - target["lp1"]) / KERNEL_BW
                ws.append(math.exp(-0.5 * gap * gap))
            coef = weighted_fit(xs, ys, ws)
            if coef is not None:
                x = [target[k] for k in FEATURES]
                median = math.exp(coef[0] + sum(coef[i + 1] * x[i]
                                                for i in range(len(x))))
                # Keep the forecast inside the range the comparables support
                # (extended by one log unit either way): the target launch
                # may be bigger than any comparable, but a fit extrapolating
                # far past the training range is a bug, not a forecast.
                lo = min(ys) - 1.0
                hi = max(ys) + 1.0
                median = math.exp(min(max(math.log(median), lo), hi))
        if median is None or not math.isfinite(median) or median <= 0:
            # Last resort: plain persistence at the hour-1 print.
            median = p1
        quantiles = []
        prev = 0.0
        for z in Z:
            q = median * math.exp(z * SPREAD)
            q = min(max(q, 1.0), 1e9)
            q = max(q, prev)
            quantiles.append(round2(q))
            prev = q
        hours.append({"hour": h, "q": quantiles})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
