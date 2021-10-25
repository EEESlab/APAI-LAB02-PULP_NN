APP = test

APP_SRCS = test.c
APP_SRCS += pulp_nn_linear_u8_i32_i8.c

ifndef cores
cores=1
else
cores = $(cores)
endif



APP_CFLAGS += -O3 -Iinclude 
APP_CFLAGS += -DNUM_CORES=$(cores)

APP_LDFLAGS += 


include $(RULES_DIR)/pmsis_rules.mk
