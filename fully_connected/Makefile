APP = test

APP_SRCS = test.c
APP_SRCS += pulp_nn_linear_u8_i32_i8.c

ifndef CORES
CORES=1
else
CORES = $(CORES)
endif



APP_CFLAGS += -O3 -Iinclude 
APP_CFLAGS += -DNUM_CORES=$(CORES)

APP_LDFLAGS += 


include $(RULES_DIR)/pmsis_rules.mk
