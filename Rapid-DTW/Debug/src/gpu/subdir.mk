################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/gpu/euclidean_distance.cu \
../src/gpu/euclidean_distance_fast.cu 

CPP_SRCS += \
../src/gpu/dtw_gpu.cpp 

OBJS += \
./src/gpu/dtw_gpu.o \
./src/gpu/euclidean_distance.o \
./src/gpu/euclidean_distance_fast.o 

CU_DEPS += \
./src/gpu/euclidean_distance.d \
./src/gpu/euclidean_distance_fast.d 

CPP_DEPS += \
./src/gpu/dtw_gpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/gpu/%.o: ../src/gpu/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O3 -maxrregcount 32 -gencode arch=compute_75,code=sm_75  -odir "src/gpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O3 -maxrregcount 32 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/gpu/%.o: ../src/gpu/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O3 -maxrregcount 32 -gencode arch=compute_75,code=sm_75  -odir "src/gpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O3 -maxrregcount 32 --compile --relocatable-device-code=false -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


