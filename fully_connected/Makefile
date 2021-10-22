APP = test

APP_SRCS = test.c
APP_SRCS += pulp_nn_linear_u8_i32_i8.c

ifndef kernel
kernel=88
else
kernel = $(kernel)
endif


ifndef cores
cores=1
else
cores = $(cores)
endif

ifeq ($(perf), 1)
APP_CFLAGS += -DVERBOSE_PERF
endif

ifeq ($(check), 1)
APP_CFLAGS += -DVERBOSE_CHECK
endif

APP_CFLAGS += -O3 -Iinclude 
APP_CFLAGS += -DNUM_CORES=$(cores) -DKERNEL=$(kernel)

APP_LDFLAGS += 


include $(RULES_DIR)/pmsis_rules.mk
