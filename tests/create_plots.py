import matplotlib.pyplot as plt
import re
from cycler import cycler
from pathlib import Path

# Matplotlib settings
colors = [
    '#8c8c8c', '#64b5cd', '#55a868', '#c44e52', '#da8bc3', '#937860',
    '#ccb974', '#4c72b0'
]
plt.rcParams.update({
    'text.usetex' : True,
    'font.family' : 'serif',
    'font.serif' : ['Computer Modern Serif'],
    'font.size': 15,
    'axes.prop_cycle': cycler('color', colors)
})

def parse_output_file(filepath):
    data = {}  # Dict[N] = (times, comms, constraints)
    current_N = None

    with open(filepath, 'r') as f:
        for line in f:
            if match := re.match(r'\s*N\s*=\s*(\d+)', line):
                current_N = int(match.group(1))
                data[current_N] = ([], [], [])
            elif match := re.match(r'\s*([\d.]+): commutator = ([\deE\+\-\.]+), constraint = ([\deE\+\-\.]+)', line):
                time = float(match.group(1))
                commutator = float(match.group(2))
                constraint = float(match.group(3))
                data[current_N][0].append(time)
                data[current_N][1].append(commutator)
                data[current_N][2].append(constraint)
    return data

for test_id in [1, 2, 3, 4, 5, 6, 7]:
    filename = f'./test{test_id}.out'
    print(f"Creating plot for test{test_id}...")

    data = parse_output_file(filename)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    for N in sorted(data.keys()):
        times, commutators, constraints = data[N]
        axes[0].plot(times, commutators, label=f'N = {N}')
        axes[1].plot(times, constraints, label=f'N = {N}')

    axes[0].set_title('Residuals')
    axes[0].set_ylabel('Commutator residual')
    axes[1].set_xlabel('Relaxation time')
    axes[1].set_ylabel('Constraint residual')
    axes[0].set_xticklabels([])
    axes[0].legend()

    for ax in axes:
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.025)
    fig.savefig(f'./test{test_id}.png', bbox_inches='tight')
    plt.close()
