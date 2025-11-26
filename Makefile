NVCC ?= nvcc
NVCCFLAGS ?= -O3 -lineinfo
# Override ARCH to match your GPU (e.g. sm_86 for Ampere, sm_90 for Hopper)
ARCH ?= sm_70

SRC_DIR := src
BIN_DIR := bin

SRC_FILES := $(SRC_DIR)/galaxy_shared_mem.cu \
             $(SRC_DIR)/galaxy_no_cache.cu \
             $(SRC_DIR)/galaxy_manual.cu

TARGETS := $(BIN_DIR)/galaxy_shared_mem \
           $(BIN_DIR)/galaxy_no_cache \
           $(BIN_DIR)/galaxy_manual

.PHONY: all clean dirs

all: dirs $(TARGETS)

dirs:
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

clean:
	@rm -rf $(BIN_DIR)/*
