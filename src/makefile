

CC         := gcc
CFLAGS     := -O2 -Wall -msse3
LDFLAGS    := -lm


MulMethods := $(patsubst %.c, %.o, $(wildcard MMult*.c))
OBJS  := main.o utils.o $(MulMethods)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all:
	make init
	echo $(MulMethods)
	make test_MMult.x
	make clean

test_MMult.x: $(OBJS) params.h
	$(CC) $(OBJS) $(LDFLAGS) $(BLAS_LIB) -o $@ 

clean:
	rm -f *.o

init:
	rm -f test_MMult.x
	rm -f *.o