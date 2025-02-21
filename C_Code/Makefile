# Compiler and Flags
CC = mpicc
CFLAGS = -Wall -O3
LDFLAGS = -llapacke -llapack -lblas -lm 

# Program Names
EXEC_TSQR = tsqr
EXEC_TSQR_SCALING = tsqr_scaling
EXEC_TEST = tsqr_test

# Source Files
SRC_TSQR = tsqr.c
SRC_TSQR_SCALING = tsqr_scaling.c
SRC_TEST = tsqr_test.c

# Hostfile for MPI
HOSTFILE = hostfile

# Object Files
OBJ_TSQR = $(SRC_TSQR:.c=.o)
OBJ_TSQR_SCALING = $(SRC_TSQR_SCALING:.c=.o)
OBJ_TEST = $(SRC_TEST:.c=.o)

# Default Target
all: $(EXEC_TSQR) $(EXEC_TEST) $(EXEC_TSQR_SCALING)

# Rule to build tsqr_scaling program
$(EXEC_TSQR_SCALING): $(OBJ_TSQR_SCALING)
	$(CC) $(OBJ_TSQR_SCALING) -o $@ $(LDFLAGS)

# Rule to build tsqr program
$(EXEC_TSQR): $(OBJ_TSQR)
	$(CC) $(OBJ_TSQR) -o $@ $(LDFLAGS)

# Rule to build tsqr_test program
$(EXEC_TEST): $(OBJ_TEST)
	$(CC) $(OBJ_TEST) -o $@ $(LDFLAGS)

# Rule to compile .c files to .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean the build files
clean:
	rm -f $(OBJ_TSQR) $(OBJ_TEST) $(OBJ_TSQR_SCALING) $(EXEC_TSQR) $(EXEC_TEST) $(EXEC_TSQR_SCALING)

# Run the test program using MPI with 4 processes
run_test: $(EXEC_TEST)
	mpirun -np 4 -hostfile $(HOSTFILE) ./$(EXEC_TEST)

# Run the test program using MPI with 4 processes
run_tsqr_scaling: $(EXEC_TSQR_SCALING)
	mpirun -np 4 -hostfile $(HOSTFILE) ./$(EXEC_TSQR_SCALING)

# Run the tsqr program using MPI with 4 processes
run_tsqr: $(EXEC_TSQR)
	mpirun -np 4 -hostfile $(HOSTFILE) ./$(EXEC_TSQR)

.PHONY: all clean run_test run_tsqr run_tsqr_scaling
