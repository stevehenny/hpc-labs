################################################################################

# Sources and targets
# use TARGET for host only programs (no GPU)
# use NVTARGET for GPU programs
# use MPITARGET for MPI programs
# TARGET = solution
# NVTARGET = solution
MPITARGET = solution
MODULES = $(if $(wildcard solution.*),solution,template)
OBJECTS = $(addsuffix .o,$(MODULES))

################################################################################

include common.mak

# Include "data" as an order-only prerequisite to generate data
# e.g. run: all | data
.PHONY: run
run: all
	@rm -f run_log.txt
	sbatch -W --nodes=32 --output=run_log.txt jobscript.sh
	@sleep 3
	-@cat run_log.txt

# Note: If you are trying to run your MPI program outside of the scheduled lab
# time, remove the --reservation=... option from the sbatch command. When the
# reservation option is removed, you may wait hours or even days for your job
# to run. If this is the case, you can also remove the -W option to not wait
# for the job to complete. The sbatch command will submit your job to the
# scheduler and return immediately. You can then use the 'squeue -u <netid>'
# command to see the status of your job.
