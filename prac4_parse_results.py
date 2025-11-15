import re
import os
import csv
import pandas as pd


def parse_perf_file(path):
    rows = []
    current_threads = None
    current_run = {}

    # Regex patterns
    re_threads = re.compile(r"export OMP_NUM_THREADS\s*=\s*(\d+)")
    re_run = re.compile(r"perf stat\s+(.*)")
    re_cycles = re.compile(r"([\d\.,]+)\s+cpu_core/cycles")
    re_insts = re.compile(r"([\d\.,]+)\s+cpu_core/instructions")
    re_time = re.compile(r"([\d\.,]+)\s+seconds time elapsed")

    def flush_run():
        print(current_run)
        if all(k in current_run for k in ("cycles", "instructions", "time", "script")):
            current_run["IPC"] = current_run["instructions"] / current_run["cycles"]
            rows.append(current_run.copy())

    with open(path, "r") as f:
        for line in f:
            line_strip = line.strip()

            # Match thread declarations
            m = re_threads.search(line_strip)
            if m:
                current_threads = int(m.group(1))
                continue

            # Match start of a new perf stat run
            if "perf stat" in line_strip:
                flush_run()
                current_run = {"threads": current_threads}

                run_part = line_strip.split("$ ")[1]
                m = re_run.match(run_part)
                if m:
                    cmd_parts = m.group(1).split()
                    # Expect: <script> <N> <DQSZ>
                    if len(cmd_parts) >= 3:
                        current_run["script"] = cmd_parts[0]
                        current_run["problem_size"] = int(cmd_parts[1])
                        current_run["dqs"] = int(cmd_parts[2])
                    else:
                        current_run["script"] = cmd_parts[0]
                        current_run["problem_size"] = None
                        current_run["dqs"] = None
                continue

            # Cycles
            m = re_cycles.search(line_strip)
            if m:
                val = int(m.group(1).replace(".", "").replace(",", ""))
                current_run["cycles"] = val
                continue

            # Instructions
            m = re_insts.search(line_strip)
            if m:
                val = int(m.group(1).replace(".", "").replace(",", ""))
                current_run["instructions"] = val
                continue

            # Time elapsed
            m = re_time.search(line_strip)
            if m:
                val = float(m.group(1).replace(",", "."))
                current_run["time"] = val
                continue

    flush_run()
    return rows


def save_csv(rows, csv_path):
    if not rows:
        print("No data parsed.")
        return

    keys = rows[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def show_table(csv_path):
    df = pd.read_csv(csv_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    # directory where THIS script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # expected input .txt in same directory
    txt_file = os.path.join(script_dir, "AHPC.txt")
    csv_file = os.path.join(script_dir, "AHPC_results.csv")

    rows = parse_perf_file(txt_file)
    save_csv(rows, csv_file)
    show_table(csv_file)
