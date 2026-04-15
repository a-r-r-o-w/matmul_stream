.PHONY := all clean

CC := nvcc
CFLAGS := -O3 -std=c++20 -Xcompiler -Wall -Xcompiler -Wextra --threads 4

all: matmul

matmul: matmul.cu
	$(CC) $(CFLAGS) matmul.cu -o matmul

clean:
	rm matmul
