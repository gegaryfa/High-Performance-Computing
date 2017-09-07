# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= Convolution2D

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -O4 --generate-code arch=compute_30,code=sm_30

#These are the flags passed to the linker. Nothing in our case
#LDFLAGS = -lm


# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .cu 
# file with the same name.
%: %.cu
	$(CC) $(CFLAGS) $< -o $@

# make clean will remove all executables, jpg files and the 
# output of previous executions.
clean:
	rm -f $(EXECUTABLES)

execute:
	./Convolution2D
