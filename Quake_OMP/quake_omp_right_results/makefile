# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= quake

#This is the compiler to use
CC = icc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -w -fast -openmp
CFLAGS_TEST = -g -Wall -openmp

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm



build:
	$(CC) $(CFLAGS) quake.c -o quake $(LDFLAGS)


test:
	$(CC) $(CFLAGS_TEST) quake.c -o quake $(LDFLAGS)

# make clean will remove all executables, jpg files and the 
# output of previous executions.
clean:
	rm -f $(EXECUTABLES) 

execute_test:
	./quake<small_input

execute:
	./quake<input

time_test:
	time ./quake<small_input

time:
	time ./quake<input
