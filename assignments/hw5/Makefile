INC_DIR := include

CUDIR := /usr/local/nvidia
NVCC := $(CUDIR)/bin/nvcc
NVCCFLAGS := -O3 -I$(CUDIR)/include -I$(INC_DIR) --compiler-options -fopenmp -m64 -arch=compute_61 -code=sm_61 -Xptxas -v

CXX := g++
CXXFLAGS :=-O3 -I$(INC_DIR) -Wall -Wextra -std=c++17

SRC_DIR := src
BUILD_DIR := build

CXX_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)

CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%_cpp.o,$(CXX_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%_cu.o,$(CU_SRCS))
OBJS := $(CPP_OBJS) $(CU_OBJS)

TARGET := output

.PHONY: all clean

all: $(BUILD_DIR) $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%_cpp.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%_cu.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)