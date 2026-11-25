CXX ?= g++
AFRESH_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ASTER_DIR ?= $(abspath $(AFRESH_DIR)/../ASTER)
CPPFLAGS ?=
CXXFLAGS ?= -O3 -std=gnu++17 -pthread
LDFLAGS ?= -pthread

IILS_STATISTICS_SOURCES := src/iils_statistics.cpp src/iils_statistics_cli.cpp
IILS_STATISTICS_HEADERS := include/iils/statistics.hpp
IILS_STATISTICS_BIN := bin/iils-statistics

.PHONY: all iils-statistics clean

all: iils-statistics

iils-statistics: $(IILS_STATISTICS_BIN)

$(IILS_STATISTICS_BIN): $(IILS_STATISTICS_SOURCES) $(IILS_STATISTICS_HEADERS)
	@test -f "$(ASTER_DIR)/src/algorithms.hpp" || { \
		echo "error: ASTER_DIR=$(ASTER_DIR) does not contain the original ASTER sources" >&2; \
		exit 1; \
	}
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Iinclude -I"$(ASTER_DIR)/src" \
		$(IILS_STATISTICS_SOURCES) $(LDFLAGS) -o "$@"

clean:
	rm -f "$(IILS_STATISTICS_BIN)"
