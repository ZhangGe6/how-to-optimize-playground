

CC         := gcc
CFLAGS     := -O2 -Wall -msse3
LDFLAGS    := -lm


UTIL := utils.o
MulMethods := $(patsubst %.c, %.o, $(wildcard MMult*.c))
TEST_OBJS  := main.o $(MulMethods)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all:
	make clean
	echo $(MulMethods)
	make test_MMult.x
	make clean_tmp

test_MMult.x: $(TEST_OBJS) $(UTIL) params.h
	$(CC) $(TEST_OBJS) $(UTIL) $(LDFLAGS) $(BLAS_LIB) -o $(TEST_BIN) $@ 

clean:
	rm -f *.o *.x
clean_tmp:
	rm -f *.o



# objects = MMult_base.o main.o  utils.o 
# $(objects): %.o: %.c 
# 	$(CC) -c $(CFLAGS) $< -o $@

# TARGET := Multest
# $(TARGET): $(objects)

# SRC_PATH := src
# OBJ_PATH := objs
# SRCS := $(wildcard  $(SRC_PATH)/*.c)
# OBJS := $(addprefix $(OBJ_PATH)/, $(patsubst %.c,%.o, $(SRCS)))

# TARGET := Multest

# $(OBJ_PATH)/%.o: $(SRC_PATH)/%.c
# 	$(CC) $(CFLAGS) -c $< -o $@

# $(SRCS): %.o: %.c
# 	$(CC) -c $(CFLAGS) $< -o $@


# all: 
# 	mkdir -p $(OBJ_PATH)
# 	make info
# # 	make Multest

# info:
# 	@echo $(SRCS)
# 	@echo $(OBJS)



# Multest: $(OBJS)
	

# Multest : main.o utils.o
# 	$(LINKER) -o Multest main.o utils.o