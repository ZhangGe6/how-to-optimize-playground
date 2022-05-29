

CC         := gcc
# -mavx is needed to use __m256
# -mfma is needede to use fma
CFLAGS     := -O2 -Wall -msse3 -mavx -mfma
LDFLAGS    := -lm

BUILD=../build
MulMethod_OBJs := $(addprefix $(BUILD)/, $(patsubst %.c, %.o, $(wildcard MMult*.c)))

$(BUILD)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $(BUILD)/$@

all:
	@echo $(MulMethod_OBJs)
	rm -f test_MMult.x
	make test_MMult.x

test_MMult.x: $(BUILD)/main.o  $(BUILD)/utils.o $(MulMethod_OBJs)
	$(CC) $(LDFLAGS)  $^ -o $@ 

clean:
	rm -rf ../build
	mkdir ../build