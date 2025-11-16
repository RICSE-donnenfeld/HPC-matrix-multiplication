from run import *

#!/usr/bin/env python3
import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# ------------------------------
# CONFIG
# ------------------------------
DB_FILE = "mmpar_raw_perf_data.csv"
THREADS = [1, 2, 4, 6, 8, 10, 12]
DQSZ_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
MATRIX_SIZE = 2048

PROGRAM = "./MMpar"

PERF_CMD = [
    "perf", "stat", "--no-big-num", "-x,",
    "-e", "cycles",
    "-e", "instructions",
    "-e", "cache-misses:u",
    "-e", "branches",
    "-e", "branch-misses",
    "--log-fd", "1",
]

# NEW: use robust wall clock timing
TIME_CMD = ["/usr/bin/time", "-f", "%e"]

OUTPUT_CSV = "mmpar_perf_results.csv"

# ------------------------------
# LOAD OR CREATE DB
# ------------------------------
if os.path.exists(DB_FILE):
    db = pd.read_csv(DB_FILE)
else:
    db = pd.DataFrame(columns=["threads", "dq", "event", "value"])


# ------------------------------
# PARSE PERF CSV LINE
# ------------------------------
def parse_perf_line(line: str):
    """
    Parses a single CSV line from perf.

    Returns:
        (value_float, event_name) or None
    """

    parts = line.strip().split(",")
    if len(parts) < 3:
        return None

    try:
        value = float(parts[0])
    except ValueError:
        return None

    event_full = parts[2].strip()
    event_base = event_full.split(":")[0]

    return value, event_base


# ------------------------------
# DATA COLLECTION
# ------------------------------

for threads in THREADS:
    for dq in DQSZ_VALUES:

        print(f"[ RUN ] threads={threads} DQSZ={dq}")

        # Skip if instructions already present
        already_done = (
            (db["threads"] == threads)
            & (db["dq"] == dq)
            & (db["event"] == "instructions")
        ).any()

        if already_done:
            print("[ SKIP ] already in database")
            continue

        # Set OpenMP threads
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)

        # --------------------------
        # 1. WALL CLOCK TIME MEASUREMENT
        # --------------------------
        time_cmd = TIME_CMD + [PROGRAM, str(MATRIX_SIZE), str(dq)]
        proc = subprocess.run(time_cmd, text=True, capture_output=True, env=env)

        try:
            time_seconds = float(proc.stderr.strip())
        except ValueError:
            time_seconds = float("nan")

        # Add execution time as event
        db = pd.concat(
            [
                db,
                pd.DataFrame(
                    [
                        {
                            "threads": threads,
                            "dq": dq,
                            "event": "time_s",
                            "value": time_seconds,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # --------------------------
        # 2. PERF METRICS
        # --------------------------
        perf_cmd = PERF_CMD + [PROGRAM, str(MATRIX_SIZE), str(dq)]
        result = subprocess.run(
            perf_cmd, text=True, capture_output=True, env=env
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


# ------------------------------
# WIDE FORMAT
# ------------------------------
df = db.copy()

wide = df.pivot_table(
    index=["threads", "dq"],
    columns="event",
    values="value",
    aggfunc="first",
).reset_index()

wide.columns.name = None

# Derived metrics
if "instructions" in wide.columns and "cycles" in wide.columns:
    wide["ipc"] = wide["instructions"] / wide["cycles"]
else:
    wide["ipc"] = np.nan

# cache miss rate
cachecol = None
for c in wide.columns:
    if "cache-misses" in c:
        cachecol = c
        break

if cachecol:
    wide["cache_miss_per_instruction"] = wide[cachecol] / wide["instructions"]
else:
    wide["cache_miss_per_instruction"] = np.nan


# ------------------------------
# PLOTTING
# ------------------------------

def heatmap(df, col, fname, log=False):
    table = df.pivot_table(
        index="threads",
        columns="dq",
        values=col,
        aggfunc="first",
    ).reindex(index=THREADS, columns=DQSZ_VALUES)

    plt.figure(figsize=(10, 6))

    if log:
        eps = np.nanmin(table.replace(0, np.nan).values)
        im = plt.imshow(table, aspect="auto",
                        norm=LogNorm(vmin=eps, vmax=np.nanmax(table.values)))
        title = f"{col} (log scale)"
    else:
        im = plt.imshow(table, aspect="auto")
        title = col

    plt.xticks(range(len(DQSZ_VALUES)),DQSZ_VALUES)
    plt.yticks(range(len(THREADS)), THREADS)
    plt.xlabel("DQSZ")
    plt.ylabel("OMP_NUM_THREADS")
    plt.title(title)
    plt.colorbar(im, label=col)

    plt.tight_layout()
    os.makedirs("out", exist_ok=True)
    plt.savefig("out/" + fname)
    plt.close()
    print("Saved:", fname)


# ------------------------------
# HEATMAPS
# ------------------------------

heatmap(wide, "time_s", "time_heatmap.pdf")           # linear scale
heatmap(wide, "time_s", "time_heatmap_log.pdf", log=True)

print("\n[ DONE ] Execution time heatmaps generated.")

