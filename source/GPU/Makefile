CXX := g++
CXXFLAGS := -std=c++11 -O3
NVFLAGS := $(CXXFLAGS)
TARGET := Solver
OBJS := point.o triangle.o delauney.o img_utils.o LowPolySolver.o
CV2FLAGS := `pkg-config --cflags --libs opencv`

.PHONY: all
all: $(TARGET)

.PHONY: Solver
Solver: $(OBJS)
	nvcc $(NVFLAGS) $(OBJS) -o Solver $(CV2FLAGS)

LowPolySolver.o: LowPolySolver.cu
	nvcc -c $(NVFLAGS) LowPolySolver.cu -o LowPolySolver.o $(CV2FLAGS)

img_utils.o: img_utils.cu
	nvcc -c $(NVFLAGS) img_utils.cu -o img_utils.o $(CV2FLAGS)

delauney.o: delauney.cu
	nvcc -c $(NVFLAGS) delauney.cu -o delauney.o

point.o: point.cu
	nvcc -c $(NVFLAGS) point.cu -o point.o

triangle.o: triangle.cu
	nvcc -c $(NVFLAGS) triangle.cu -o triangle.o

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)