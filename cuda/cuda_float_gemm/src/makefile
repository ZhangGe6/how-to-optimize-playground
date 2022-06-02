

CC         := nvcc
CFLAGS     := -std=c++11 -O2 -g
LDFLAGS    := -lm -lcublas -lopenblas
GENCODE_FLAGS=-gencode=arch=compute_52,code=sm_52 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_61,code=sm_61 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_60,code=compute_60

BUILD=../build
MulMethod_OBJs := $(addprefix $(BUILD)/, $(patsubst %.cu, %.o, $(wildcard MMult*.cu)))

$(BUILD)/%.o: %.cpp
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $(BUILD)/$@

$(BUILD)/%.o: %.cu
	$(CC) $(CFLAGS) $(GENCODE_FLAGS)  -c $< -o $(BUILD)/$@

all:
	@echo $(MulMethod_OBJs)
	rm -f test_MMult.x
	make test_MMult.x

test_MMult.x: $(BUILD)/main.o  $(BUILD)/utils.o $(MulMethod_OBJs)
	$(CC) $(LDFLAGS) $(BLAS_LIB) $^ -o $@ 

clean:
	rm -f *.o
