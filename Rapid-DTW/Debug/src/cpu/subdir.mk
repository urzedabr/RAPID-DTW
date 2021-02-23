################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cpu/dtw_cpu.cpp 

OBJS += \
./src/cpu/dtw_cpu.o 

CPP_DEPS += \
./src/cpu/dtw_cpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/cpu/%.o: ../src/cpu/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O3 -maxrregcount 32 -gencode arch=compute_75,code=sm_75  -odir "src/cpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O3 -maxrregcount 32 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


