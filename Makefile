CPP_STANDARD := c++17
CXXFLAGS := -O3 -fPIC -std=$(CPP_STANDARD)
PREFIX:=.
EMULATOR_EXTRAS := ../../hls4mlEmulatorExtras
AP_TYPES := $(EMULATOR_EXTRAS)/include/ap_types
HLS_ROOT := ../../hls
HLS4ML_INCLUDE := $(EMULATOR_EXTRAS)/include/hls4ml
NN_INCLUDE := TOPO_v1/NN
INCLUDES := -I$(HLS4ML_INCLUDE) -I$(AP_TYPES) -I$(HLS_ROOT)/include -I$(NN_INCLUDE)
LD_FLAGS := -L$(EMULATOR_EXTRAS)/lib64 -lemulator_interface
ALL_VERSIONS:=TOPO_v1/topo_v1.so 

.DEFAULT_GOAL := all
.PHONY: all clean install

all: $(ALL_VERSIONS)
	@cp $(ALL_VERSIONS) ./
	@echo All OK

install: all
	@rm -rf $(PREFIX)/lib64
	@mkdir -p $(PREFIX)/lib64
	cp topo_*.so $(PREFIX)/lib64

%.so:
	$(MAKE) -C $(@D) INCLUDES="$(INCLUDES)" LD_FLAGS="$(LD_FLAGS)" CXXFLAGS="$(CXXFLAGS)"

clean:
	rm topo_*.so $(ALL_VERSIONS)
