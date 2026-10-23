import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

def plotScalingTestTimeBreakdown(csvFilePath):
    data = []
    with open(csvFilePath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                "setup": row["Setup"],
                "fields": float(row["Task Total[fields] (s)"]),
                "particles": float(row["Task Total[particles] (s)"]),
                "moments": float(row["Task Total[moments] (s)"]),
                "total": float(row["Simulation Time (s)"])
            })
    threads = [int(int(d["setup"].split('_')[1].split('x')[0]) * int(d["setup"].split('_')[1].split('x')[1]) * int(d["setup"].split('_')[1].split('x')[2])) for d in data]
    fields_times = [d["fields"] for d in data]
    GPU_times = [d["particles"] + d["moments"] for d in data]
    total_times = [d["total"] for d in data]
    other_times = [
        total_times[i] - (fields_times[i] + GPU_times[i]) 
        for i in range(len(threads))
    ]
    plt.figure(figsize=(14, 8))  
    bar_width = 0.3  
    index = np.arange(len(threads)) * 1.5  
    p1 = plt.bar(index, fields_times, bar_width, label='Solver-CPU')
    p2 = plt.bar(index, GPU_times, bar_width, bottom=fields_times, label='Mover+Moments-GPU')

    previous_bottoms = [0] * len(threads)
    for i in range(len(threads)):
        x_pos = index[i] + bar_width + 0.1 
        label_spacing = 15  
        y_pos = previous_bottoms[i] + fields_times[i] / 2
        plt.text(x_pos, y_pos, f'{fields_times[i]:.2f}s\n({fields_times[i]/total_times[i]*100:.1f}%)', 
                 ha='left', va='center', fontsize=8)
        plt.plot([index[i] + bar_width / 2, x_pos - 0.05], [y_pos, y_pos], color='black', lw=0.5)

        previous_bottoms[i] += fields_times[i]
        y_pos = previous_bottoms[i] + GPU_times[i] / 2 + label_spacing
        plt.text(x_pos, y_pos, f'{GPU_times[i]:.2f}s\n({GPU_times[i]/total_times[i]*100:.1f}%)', 
                 ha='left', va='center', fontsize=8)
        plt.plot([index[i] + bar_width / 2, x_pos - 0.05], [y_pos - label_spacing, y_pos], color='black', lw=0.5)


    plt.xlabel('Number of MPI Processes')
    plt.ylabel('Time (s)')
    plt.title('GEM3Dsmall 100 cycle | ' + csvFilePath)
    plt.xticks(index, threads)
    plt.legend()
    plt.tight_layout()
    plt.savefig(csvFilePath + '.png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot scaling test time breakdown from CSV file.')
    parser.add_argument('csvFilePath', type=str, help='Path to the CSV file with test data.')
    args = parser.parse_args()
    plotScalingTestTimeBreakdown(args.csvFilePath)
