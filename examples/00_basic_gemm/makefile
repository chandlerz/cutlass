ECHO = @echo
AT =

CUDA_PATH  =/usr/local/cuda-10.1

OUTNAME = basic_gemm
CC   := g++
CUCC := $(CUDA_PATH)/bin/nvcc

CFLAGS  := -m64 -std=c++11 
CUFLAGS := -m64 -std=c++11 

# CFLAGS  += -g -lineinfo
# CUFLAGS += -g -G

# Gencode arguments
SMS ?= 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_COM_PATH = $(CUDA_PATH)/samples/common/inc
CUTLASS_INC_PATH = ../../

INCLUDES := -I$(CUDA_COM_PATH) -I$(CUDA_INC_PATH) -I$(CUTLASS_INC_PATH) -I/usr/include
CFLAGS += $(INCLUDES)
CUFLAGS += $(INCLUDES)

CUDA_LIB_PATH  = $(CUDA_PATH)/lib64
LDPATH := -L$(CUDA_LIB_PATH) 

LDFLAGS :=  $(LDPATH) -lcudart -lcublas 
LDFLAGS += -Wl,-rpath=$(CUDA_LIB_PATH)

SRCDIR := ./
OBJDIR := ../obj
BINDIR := ../bin

OBJS   = $(patsubst %.cpp, $(OBJDIR)/%.o, $(notdir $(wildcard $(SRCDIR)/*.cpp)))
CUOBJS = $(patsubst %.cu,  $(OBJDIR)/%.o, $(notdir $(wildcard $(SRCDIR)/*.cu)))

all: build

build: $(BINDIR)/$(OUTNAME)

$(BINDIR)/$(OUTNAME): $(OBJS) $(CUOBJS)
	$(ECHO) Linking: $@
	$(AT)$(CC) -o $@ $^ $(LDFLAGS) 

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CC) $(CFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(AT)if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(call make-depend,$<,$@,$(subst .o,.d,$@))
	$(ECHO) Compiling: $<
	$(AT)$(CUCC) $(CUFLAGS) $(GENCODE_FLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR)/* $(BINDIR)/$(OUTNAME)
