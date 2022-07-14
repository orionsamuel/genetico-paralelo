NVCC=nvcc
CUDA_FLAGS = -I include -arch=sm_37
EXECUTABLE = exec

all: clean $(EXECUTABLE)

$(EXECUTABLE): functions.o genetic.o main.o
	$(NVCC) $(CUDA_FLAGS) -o build/$(EXECUTABLE) main.o genetic.o functions.o 

main.o: src/main.cu
	$(NVCC) $(CUDA_FLAGS) -c src/main.cu

genetic.o: src/genetic.cu include/genetic.hpp 
	$(NVCC) $(CUDA_FLAGS) -c src/genetic.cu

functions.o: src/functions.cu include/functions.hpp include/utils.hpp
	$(NVCC) $(CUDA_FLAGS) -c src/functions.cu

clean:
	rm -f *.o
	rm -f build/$(EXECUTABLE)
	rm -f build/out.txt
