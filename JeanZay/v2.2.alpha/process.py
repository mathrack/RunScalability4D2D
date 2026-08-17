#!/usr/bin/env -S python3 -i

import glob as glob
import numpy as np
import matplotlib.pyplot as plt

#
# Create a class to store the result of one test
#
class run(object):
    #
    def __init__(self, prow, pcol, nx, ny, nz, ntest, name, timefor, timeback):
        self.prow = int(prow)
        self.pcol = int(pcol)
        if self.prow == 0 or self.pcol == 0:
            self.prow = 1
            self.pcol = 1
        self.ngpu = self.prow * self.pcol
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        self.ntest = int(ntest)
        self.name = name
        if "_a2a_" in name:
            self.mpi = "a2a"
        elif "_a2av_" in name:
            self.mpi = "a2av"
        else:
            self.mpi = "?"
        if "_dp_" in name:
            self.prec = "dp"
        elif "_sp_" in name:
            self.prec = "sp"
        else:
            self.prec = "?"
        if "/c2c/" in name:
            self.type = "c2c"
        elif "/r2c_x/" in name:
            self.type = "r2c_x"
        elif "/r2c_z/" in name:
            self.type = "r2c_z"
        else:
            self.type = "?"
        self.forward = timefor
        self.backward = timeback
        self.timer = self.forward + self.backward
        if self.prow==1 and self.pcol==1:
            self.serial = True
            self.slab = False
            self.pencil = False
        else:
            self.serial = False
            if self.prow==1 or self.pcol==1:
                self.slab = True
                self.pencil = False
            else:
                self.slab = False
                self.pencil = True
    #
    def __repr__(self):
        return self.name
    def __str__(self):
        return "\n" \
               "Name of the file : " + str(self.name) + "\n" \
               "Grid nx, ny, nz : " + str(self.nx) + ", " + str(self.ny) + ", " + str(self.nz) + "\n" \
               "Communication backend : " + self.mpi + "\n" \
               "Precision : " + self.prec + "\n" \
               "Type of transform : " + self.type + "\n" \
               "Number of GPU(s) : " + str(self.ngpu) + "\n" \
               "Domain decomp. : " + str(self.prow) + ", " + str(self.pcol) + "\n" \
               "Number of iterations : " + str(self.ntest) + "\n" \
               "Averaged forward time in seconds : " + str(self.forward) + "\n" \
               "Averaged backward time in seconds : " + str(self.backward) + "\n" \
               "Averaged time in seconds : " + str(self.timer) + "\n" \
               "Serial / slab / pencil : " + str(self.serial) + " " + str(self.slab) + " " + str(self.pencil)

#
# Process a c2c, r2c_x or r2c_z folder
#
def read_runs(folder = ".", header = "c2c_x_"):
    output = []
    for file in glob.glob(folder + "/*[0123456789a].slurm"):
        # Open the file and extract prow, pcol, nx, ny, nz, ntest
        for line in open(file).readlines():
            if 'srun' in line.split():
                prow = int(line.split()[-6])
                pcol = int(line.split()[-5])
                nx = line.split()[-4]
                ny = line.split()[-3]
                nz = line.split()[-2]
                ntest = line.split()[-1]
        # Read the timer in the corresponding files
        name = header
        if int(nx) < 1000:
            name = name + "0" + nx + "*.out"
        else:
            name = name + nx + "*.out"
        for file in glob.glob(folder + "/" + name):
            if "_a" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol
                pcol = 1
            elif "_b" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 2
                pcol = 2
            elif "_c" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 4
                pcol = 4
            elif "_d" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 8
                pcol = 8
            elif "_e" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 16
                pcol = 16
            elif "_f" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 32
                pcol = 32
            elif "_g" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 64
                pcol = 64
            elif "_h" in file.split(".")[-3].split("/")[-1]:
                prow = prow * pcol / 128
                pcol = 128
            for line in open(file).readlines():
                if 'Avg time (sec)' in line:
                    timefor = float(line.split()[-2])
                    timeback = float(line.split()[-1])
            # Store
            output.append(run(prow, pcol, nx, ny, nz, ntest, file, timefor, timeback))
    return output

#
# Return best timers in the set
#
def timer_best(data, prec = None, nx = False, ngpu = False):
    if prec:
        return timer_best([run for run in data if run.prec == prec], prec=None, nx=nx, ngpu=ngpu)
    nn = []
    if nx:
        for run in data:
            if run.nx in nn: continue
            nn.append(run.nx)
    elif ngpu:
        for run in data:
            if run.ngpu in nn: continue
            nn.append(run.ngpu)
    else:
        print("Error")
        exit()
    nn.sort()
    best = []
    for n in nn:
        for run in data:
            if (nx and run.nx == n) or (ngpu and run.ngpu == n):
                # Best was initialized ?
                if not best:
                    best.append(run)
                    continue
                # Last item in best with correct size ?
                if (nx and best[-1].nx != n) or (ngpu and best[-1].ngpu != n):
                    best.append(run)
                    continue
                # Last item with correct size was slow
                if best[-1].timer > run.timer:
                    best[-1] = run
    return best

# Set default parameters
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'lines.linewidth': 2})

# Show and save figures ?
show_fig = True
save_fig = True

# Global benchmark parameters
mpi = ["a2a", "a2av"]
nmpi = len(mpi)
prec = ["sp", "dp"]
nprec = len(prec)
case = ["c2c", "r2c_x", "r2c_z"]
ncase = len(case)
gpus = ["1", "2", "4", "8", "16", "32", "64", "128"]
ngpus = len(gpus)

# Process all the cases
data = []
for impi in range(len(mpi)):
    a2a = mpi[impi]
    for iprec in range(len(prec)):
        dp = prec[iprec]
        for icas in range(len(case)):
            cas = case[icas]
            for igpu in range(len(gpus)):
                gpu = gpus[igpu]
                folder = "dev_cumpi_" + a2a + "_" + dp + "_inplace/" + cas + "/" + gpu
                if icas == 0:
                    data.append(read_runs(folder))
                else:
                    data.append(read_runs(folder, header = cas + "_"))

# Process c2c/1
case_c2c_gpu_1_nx_x = []
for runs in data:
    if runs[0].type == "c2c" and runs[0].ngpu == 1:
        for run in runs:
            case_c2c_gpu_1_nx_x.append(run)

# One GPU, plot all c2c
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_c2c_gpu_1_nx_x:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.nx, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$n_x = n_y = n_z$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle('Single GPU, variable problem size')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("c2c_gpu_1.png")
    if show_fig: fig.show()

# Process r2c_x/1
case_r2cX_gpu_1_nx_x = []
for runs in data:
    if runs[0].type == "r2c_x" and runs[0].ngpu == 1:
        for run in runs:
            case_r2cX_gpu_1_nx_x.append(run)

# One GPU, plot all r2c_x
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_r2cX_gpu_1_nx_x:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.nx, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$n_x = n_y = n_z$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in X')
    fig.suptitle('Single GPU, variable problem size')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("r2c_x_gpu_1.png")
    if show_fig: fig.show()

# Process r2c_z/1
case_r2cZ_gpu_1_nx_x = []
for runs in data:
    if runs[0].type == "r2c_z" and runs[0].ngpu == 1:
        for run in runs:
            case_r2cZ_gpu_1_nx_x.append(run)

# One GPU, plot all r2c_x
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_r2cZ_gpu_1_nx_x:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.nx, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$n_x = n_y = n_z$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in Z')
    fig.suptitle('Single GPU, variable problem size')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("r2c_z_gpu_1.png")
    if show_fig: fig.show()

# Process c2c, mesh 512^3
case_c2c_gpu_x_nx_512 = []
for runs in data:
    if runs[0].type == "c2c":
        for run in runs:
            if run.nx == 512:
                case_c2c_gpu_x_nx_512.append(run)

# Plot
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_c2c_gpu_x_nx_512:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $512^3$, all cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("c2c_nx_512.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_c2c_gpu_x_nx_512, prec="sp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"       
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $512^3$, single prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_sp_c2c_nx_512.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_c2c_gpu_x_nx_512, prec="dp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $512^3$, double prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_dp_c2c_nx_512.png")
    if show_fig: fig.show()

# Process c2c, mesh 1024^3
case_c2c_gpu_x_nx_1024 = []
for runs in data:
    if runs[0].type == "c2c":
        for run in runs:
            if run.nx == 1024:
                case_c2c_gpu_x_nx_1024.append(run)

# Plot
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_c2c_gpu_x_nx_1024:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $1024^3$")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("c2c_nx_1024.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_c2c_gpu_x_nx_1024, prec="sp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"       
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $1024^3$, single prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_sp_c2c_nx_1024.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_c2c_gpu_x_nx_1024, prec="dp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $1024^3$, double prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_dp_c2c_nx_1024.png")
    if show_fig: fig.show()

# Process r2c_x, mesh 1024^3
case_r2cX_gpu_x_nx_1024 = []
for runs in data:
    if runs[0].type == "r2c_x":
        for run in runs:
            if run.nx == 1024:
                case_r2cX_gpu_x_nx_1024.append(run)

# Plot
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_r2cX_gpu_x_nx_1024:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in X')
    fig.suptitle(r"Fixed problem size $1024^3$")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("r2c_x_nx_1024.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_r2cX_gpu_x_nx_1024, prec="sp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"       
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in X')
    fig.suptitle(r"Fixed problem size $1024^3$, single prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_sp_r2c_x_nx_1024.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_r2cX_gpu_x_nx_1024, prec="dp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in X')
    fig.suptitle(r"Fixed problem size $1024^3$, double prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_dp_r2c_x_nx_1024.png")
    if show_fig: fig.show()

# Process r2c_z, mesh 1024^3
case_r2cZ_gpu_x_nx_1024 = []
for runs in data:
    if runs[0].type == "r2c_z":
        for run in runs:
            if run.nx == 1024:
                case_r2cZ_gpu_x_nx_1024.append(run)

# Plot
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_r2cZ_gpu_x_nx_1024:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in Z')
    fig.suptitle(r"Fixed problem size $1024^3$")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("r2c_z_nx_1024.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_r2cZ_gpu_x_nx_1024, prec="sp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"       
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in Z')
    fig.suptitle(r"Fixed problem size $1024^3$, single prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_sp_r2c_z_nx_1024.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_r2cZ_gpu_x_nx_1024, prec="dp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in Z')
    fig.suptitle(r"Fixed problem size $1024^3$, double prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_dp_r2c_z_nx_1024.png")
    if show_fig: fig.show()

# r2c_z, 1024^3, plot the impact of p_row and p_col when using 64 GPUs
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_r2cZ_gpu_x_nx_1024:
        if run.ngpu != 64:
            continue
        marker = ""
        if run.prec == "dp":
            marker = marker + "+" 
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$" 
        if run.mpi == "a2a":
            marker = marker + "k" 
        else:                  
            marker = marker + "b"
        ax.plot(run.prow, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel(r"$prow$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward real-to-complex 3D FFT, physical in Z')
    fig.suptitle(r"Fixed problem size $1024^3$, 64 GPUs")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("r2c_z_gpu_64_nx_1024.png")
    if show_fig: fig.show()

# Process c2c, mesh 2048^3
case_c2c_gpu_x_nx_2048 = []
for runs in data:
    if runs[0].type == "c2c":
        for run in runs:
            if run.nx == 2048:
                case_c2c_gpu_x_nx_2048.append(run)

# Plot
if True and (save_fig or show_fig):
    # Absolute timer
    fig, ax = plt.subplots(1, layout="constrained")
    for run in case_c2c_gpu_x_nx_2048:
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $2048^3$")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("c2c_nx_2048.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_c2c_gpu_x_nx_2048, prec="sp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"       
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $2048^3$, single prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_sp_c2c_nx_2048.png")
    if show_fig: fig.show()
    fig, ax = plt.subplots(1, layout="constrained")
    for run in timer_best(case_c2c_gpu_x_nx_2048, prec="dp", ngpu = True):
        marker = ""
        if run.prec == "dp":
            marker = marker + "+"
        else:
            marker = marker + "o"
        label = "$" + run.prec.upper() + " - " + run.mpi + "$"
        if run.mpi == "a2a":
            marker = marker + "k"
        else:                  
            marker = marker + "b"
        ax.plot(run.ngpu, run.timer, marker, markerfacecolor='none', label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log') 
    ax.set_xlabel(r"$Number$ $of$ $GPUs$")
    ax.set_ylabel(r"$Absolute$ $time$ $T$")
    ax.set_title('Forward + backward complex-to-complex 3D FFT')
    fig.suptitle(r"Fixed problem size $2048^3$, double prec, best cases")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    if save_fig: fig.savefig("best_dp_c2c_nx_2048.png")
    if show_fig: fig.show()
