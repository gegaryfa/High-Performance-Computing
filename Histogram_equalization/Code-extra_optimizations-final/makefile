# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= Histogram
OBJS   = cuda_check.o con_enh.o hist_equ.o main.o

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -O4 -w --generate-code arch=compute_30,code=sm_30

#These are the flags passed to the linker
LDFLAGS = -lm


# make all will create all executables
all: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXECUTABLES) $(LDFLAGS)

cuda_check.o: CudaErrorChecks.cu CudaErrorChecks.h
	$(CC) $(CFLAGS) -c CudaErrorChecks.cu -o cuda_check.o

con_enh.o: contrast-enhancement.cu hist-equ.h
	$(CC) $(CFLAGS) -c contrast-enhancement.cu -o con_enh.o

hist_equ.o: histogram-equalization.cu hist-equ.h
	$(CC) $(CFLAGS) -c histogram-equalization.cu -o hist_equ.o

main.o: main.cu
	$(CC) $(CFLAGS) -c main.cu -o main.o


# make clean will remove all executables.
clean:
	rm -f $(EXECUTABLES) $(OBJS)

execute:
	./Histogram
