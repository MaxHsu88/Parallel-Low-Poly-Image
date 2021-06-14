CXX := g++
CXXFLAGS := -std=c++11 -O3
NVFLAGS := $(CXXFLAGS) # --rdc-true
TARGET := Solver
OBJS := point.o triangle.o delauney.o img_utils.o LowPolySolver.o
CV2FLAGS := `pkg-config --cflags --libs opencv`

.PHONY: all
all: $(TARGET)

.PHONY: Solver
Solver: $(OBJS)
	nvcc $(NVFLAGS) $(OBJS) -o Solver $(CV2FLAGS)

LowPolySolver.o: LowPolySolver.cu
	nvcc -dc $(NVFLAGS) LowPolySolver.cu -o LowPolySolver.o $(CV2FLAGS)

img_utils.o: img_utils.cu
	nvcc -dc $(NVFLAGS) img_utils.cu -o img_utils.o $(CV2FLAGS)

delauney.o: delauney.cu
	nvcc -dc $(NVFLAGS) delauney.cu -o delauney.o $(CV2FLAGS)

point.o: point.cu
	nvcc -dc $(NVFLAGS) point.cu -o point.o

triangle.o: triangle.cu
	nvcc -dc $(NVFLAGS) triangle.cu -o triangle.o

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)