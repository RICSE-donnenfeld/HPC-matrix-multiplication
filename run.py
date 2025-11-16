#!/usr/bin/env python3
import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# ------------------------------
# CONFIG
# ------------------------------
DB_FILE = "mmpar_raw_perf_data.csv"
THREADS = [1, 2, 4, 6, 8, 10, 12]
#THREADS = [10, 12]
DQSZ_VALUES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
#DQSZ_VALUES = [256, 512]
MATRIX_SIZE = 2048

PROGRAM = "./MMpar"

PERF_CMD = [
    "perf", "stat",
    "-e", "task-clock",
    "-e", "cycles",
    "-e", "instructions",
    "-e", "cache-misses:u",
    "-e", "branches",
    "-e", "branch-misses",
    "-e", "duration_time",
    "-x,", "--log-fd", "1",
]

OUTPUT_CSV = "mmpar_perf_results.csv"

### Check exsiting

import os

if os.path.exists(DB_FILE):
    db = pd.read_csv(DB_FILE)
else:
    db = pd.DataFrame(columns=["threads", "dq", "event", "value"])


# ------------------------------
# PARSING
# ------------------------------

# Example lines (your sample):
# 1281,90,msec,task-clock:u,1281901128,100,00,4,CPUs utilized
# 25287216196,,instructions:u,1281901128,100,00,4,insn per cycle
# 5508409769,,cycles:u,1281901128,100,00,4,GHz

def parse_perf_line(line: str):
    """
    Parse 'perf stat -x,' CSV lines:
    value,unit,event,....

    Returns (value_float, event_base_name) or None.
    """
    parts = line.strip().split(",")
    if len(parts) < 4:
        return None

    # Numeric value
    v = parts[0].strip().replace(",", ".")
    try:
        value = float(v)
    except:
        return None

    # event is in field index 2 *OR* 3 depending on unit presence
    # Detect automatically:
    # If parts[2] contains a ':' or is known event → use it
    # Else → use parts[3]
    candidate2 = parts[2].strip()
    candidate3 = parts[3].strip()

    if ":" in candidate2 or candidate2 in ("task-clock", "duration_time",
                                           "cycles", "instructions",
                                           "branches", "branch-misses",
                                           "cache-misses"):
        event_full = candidate2
    else:
        event_full = candidate3

    if not event_full:
        return None

    event = event_full.split(":")[0]
    return value, event


# ------------------------------
# RUN EXPERIMENTS + COLLECT
# ------------------------------

rows = []

for threads in THREADS:
    for dq in DQSZ_VALUES:
        print(f"[ RUN ] threads={threads} N={MATRIX_SIZE} DQSZ={dq}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)

        cmd = PERF_CMD + [PROGRAM, str(MATRIX_SIZE), str(dq)]

        already_done = (
            (db["threads"] == threads)
            & (db["dq"] == dq)
            & (db["event"] == "instructions")
        ).any()

        if already_done:
            print(f"[ SKIP ] already measured: threads={threads}, dq={dq}")
            continue

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            env=env,
        )

        for line in result.stdout.splitlines():
            parsed = parse_perf_line(line)
            if not parsed:
                continue

            value, event = parsed

            db = pd.concat(
                [
                    db,
                    pd.DataFrame(
                        [
                            {
                                "threads": threads,
                                "dq": dq,
                                "event": event,
                                "value": value,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    db.to_csv(DB_FILE, index=False)


# Build long-format DataFrame
df = db.copy()
if df.empty:
    raise RuntimeError("No perf data parsed — check PERF_CMD / output format.")

print("\nRaw parsed data (long format):")
print(df.head())


# ------------------------------
# WIDE FORMAT + DERIVED METRICS
# ------------------------------

# Pivot to one row per (threads, dq), cols = events (task-clock, cycles, ...)
wide = df.pivot_table(
    index=["threads", "dq"],
    columns="event",
    values="value",
    aggfunc="first",
).reset_index()

# Flatten columns (pivot adds a name for columns)
wide.columns.name = None

print("\nWide data (one row per config):")
print(wide.head())

# Ensure required columns exist
required_events = ["task-clock", "cycles", "instructions"]
for ev in required_events:
    if ev not in wide.columns:
        print("Columns:", wide.columns)
        print(wide.head())
        raise RuntimeError(f"Missing required event '{ev}' in perf output.")

# time in seconds: task-clock is in msec
wide["time_s"] = wide["task-clock"] / 1000.0

# IPC = instructions / cycles
wide["ipc"] = wide["instructions"] / wide["cycles"]

# Detect cache-misses column (might be cache-misses:u)
cache_miss_col = None
for col in wide.columns:
    if "cache-misses" in col:
        cache_miss_col = col
        break

if cache_miss_col is None:
    print("WARNING: cache-misses column not found!")
    wide["cache_miss_per_instruction"] = float("nan")
else:
    wide["cache_miss_per_instruction"] = wide[cache_miss_col] / wide["instructions"]

if "duration_time" in wide.columns:
    wide["duration_time_s"] = wide["duration_time"] / 1e9
else:
    print("WARNING: duration_time not found!")
    wide["duration_time_s"] = float("nan")

# Speedup: baseline = time at threads == 1 for the same dq
base_times = wide[wide["threads"] == 1].set_index("dq")["time_s"]

#T_base = float(
#    wide[(wide["threads"] == 1) & (wide["dq"] == MATRIX_SIZE)]["time_s"].iloc[0]
#)
#wide["speedup_absolute"] = T_base / wide["time_s"]
#
#print(f"\nBaseline T(1, {MATRIX_SIZE}) = {T_base:.6f} s")


def compute_speedup(row):
    base_time = base_times.get(row["dq"])
    if base_time is None or row["time_s"] == 0:
        return float("nan")
    return base_time / row["time_s"]


wide["speedup"] = wide.apply(compute_speedup, axis=1)
wide["efficiency"] = wide["speedup"] / wide["threads"]

print("\nWide data + derived metrics:")
print(wide.head())

# Save CSV for later analysis
wide.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved CSV: {OUTPUT_CSV}")


# ------------------------------
# PLOTTING HELPERS
# ------------------------------

def line_plot_vs_threads_log(df_wide, y_col, ylabel=None, fname="plot_log.pdf"):
    """
    Same as line_plot_vs_threads, but Y axis is log scale.
    One curve per DQSZ, X = threads, Y = metric (log scale).
    """
    plt.figure()
    for dq in DQSZ_VALUES:
        sub = df_wide[df_wide["dq"] == dq].sort_values("threads")
        if sub.empty:
            continue
        plt.plot(sub["threads"], sub[y_col], marker="o", label=f"DQSZ={dq}")

    plt.xlabel("OMP_NUM_THREADS")
    plt.ylabel(ylabel or y_col)
    plt.yscale("log")                         # <-- LOG SCALE
    plt.title((ylabel or y_col) + " (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("out/" + fname)
    plt.close()
    print(f"Saved {fname}")

def line_plot_vs_threads(df_wide, y_col, ylabel=None, fname="plot.svg"):
    """
    One curve per DQSZ, X = threads, Y = metric.
    """
    plt.figure()
    for dq in DQSZ_VALUES:
        sub = df_wide[df_wide["dq"] == dq].sort_values("threads")
        if sub.empty:
            continue
        plt.plot(sub["threads"], sub[y_col], marker="o", label=f"DQSZ={dq}")

    plt.xlabel("OMP_NUM_THREADS")
    plt.ylabel(ylabel or y_col)
    plt.title(ylabel or y_col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("out/" + fname)
    plt.close()
    print(f"Saved {fname}")


def heatmap_metric(df_wide, metric, fname="heatmap.svg"):
    """
    Heatmap with threads as rows, DQSZ as columns.
    """
    # Create table threads x dq for the metric
    table = df_wide.pivot_table(
        index="threads",
        columns="dq",
        values=metric,
        aggfunc="first",
    ).reindex(index=THREADS, columns=DQSZ_VALUES)

    plt.figure()
    im = plt.imshow(table, aspect="auto")  # default colormap

    # Ticks / labels
    plt.xticks(range(len(DQSZ_VALUES)), DQSZ_VALUES)
    plt.yticks(range(len(THREADS)), THREADS)
    plt.xlabel("DQSZ")
    plt.ylabel("OMP_NUM_THREADS")
    plt.title(metric)

    # Colorbar
    plt.colorbar(im, label=metric)

    plt.tight_layout()
    plt.savefig("out/" + fname)
    plt.close()
    print(f"Saved {fname}")


def heatmap_metric_log(df_wide, metric, fname="heatmap_log.pdf"):
    """
    Heatmap with threads as rows, DQSZ as columns,
    and LOG-SCALE color mapping.
    """
    table = df_wide.pivot_table(
        index="threads",
        columns="dq",
        values=metric,
        aggfunc="first",
    ).reindex(index=THREADS, columns=DQSZ_VALUES)

    plt.figure()

    # Use LogNorm for color scaling
    im = plt.imshow(
        table,
        aspect="auto",
        norm=LogNorm(),        # <-- LOG SCALE HERE
    )

    plt.xticks(range(len(DQSZ_VALUES)), DQSZ_VALUES)
    plt.yticks(range(len(THREADS)), THREADS)
    plt.xlabel("DQSZ")
    plt.ylabel("OMP_NUM_THREADS")
    plt.title(metric + " (log scale)")

    cbar = plt.colorbar(im)
    cbar.set_label(metric)

    plt.tight_layout()
    plt.savefig("out/" + fname)
    plt.close()
    print(f"Saved {fname}")


# ------------------------------
# LINE PLOTS (SVG)
# ------------------------------

line_plot_vs_threads(wide, "time_s", ylabel="Time (s)", fname="time_vs_threads.svg")
line_plot_vs_threads(wide, "speedup", ylabel="Speedup", fname="speedup_vs_threads.svg")
line_plot_vs_threads(
    wide, "efficiency", ylabel="Efficiency", fname="efficiency_vs_threads.svg"
)
line_plot_vs_threads(wide, "ipc", ylabel="IPC", fname="ipc_vs_threads.svg")
line_plot_vs_threads(
    wide,
    "cache_miss_per_instruction",
    ylabel="Cache-misses per instruction",
    fname="cachemiss_perinst_vs_threads.svg",
)

# ------------------------------
# HEATMAPS (SVG)
# ------------------------------

heatmap_metric(wide, "speedup", fname="speedup_heatmap.svg")
#heatmap_metric(
#    wide,
#    "speedup_absolute",
#    fname="speedup_absolute_heatmap.svg",
#)

heatmap_metric(wide, "ipc", fname="ipc_heatmap.svg")
heatmap_metric(
    wide,
    "cache_miss_per_instruction",
    fname="cachemiss_perinst_heatmap.svg",
)
heatmap_metric(
    wide, 
    "duration_time_s", 
    fname="duration_time_heatmap.pdf"
)
heatmap_metric_log(
    wide,
    "duration_time_s",
    fname="duration_time_heatmap_log_all.pdf"
)
print("\nAll SVGs generated.")
