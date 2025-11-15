import subprocess

nb_procs_choices = [2, 3, 6, 12]
nb_DQSZ = [64, 128, 256, 512, 1024]
MATRIX_SIZE = 2048
results = []

for p in nb_procs_choices:
    print(f"*** Procs : {p}")
    for DQSZ in nb_DQSZ:
        result_string = ""
        print(f"\t*** Starting DQSize : {DQSZ}")
        cmd = f"OMP_NUM_THREADS={p} perf stat ./MMpar {MATRIX_SIZE} {DQSZ}"
        res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        result_string += res.stdout
        result_string += res.stderr
        results += [result_string]

with open("out.txt", "w") as f:
    for res in results:
        f.write(res)
