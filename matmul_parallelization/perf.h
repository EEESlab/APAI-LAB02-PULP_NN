#ifndef _PERF_H
#define _PERF_H

#define HOTTING 2
#define REPEAT  5


#ifdef PERF

#define INIT_STATS()

#define ENTER_STATS_LOOP()  \
    unsigned long _cycles = 0; \
    unsigned long _instr = 0; \
    unsigned long _active = 0; \
    unsigned long _ldext = 0; \
    unsigned long _tcdmcont = 0; \
    unsigned long _ldstall = 0; \
    unsigned long _imiss = 0; \
    for(int _k=0; _k<HOTTING+REPEAT; _k++) { \
      pi_perf_conf((1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR) | (1<<PI_PERF_ACTIVE_CYCLES) | (1<<PI_PERF_LD_EXT) | (1<<PI_PERF_TCDM_CONT) | (1<<PI_PERF_LD_STALL) | (1<<PI_PERF_IMISS) );


#define START_STATS()  \
    pi_perf_reset(); \
    pi_perf_start();

#define STOP_STATS() \
     pi_perf_stop(); \
     if (_k >= HOTTING) \
      { \
        _cycles   += pi_perf_read (PI_PERF_CYCLES); \
        _instr    += pi_perf_read (PI_PERF_INSTR); \
    	_active   += pi_perf_read (PI_PERF_ACTIVE_CYCLES); \
        _ldext    += pi_perf_read (PI_PERF_LD_EXT); \
    	_tcdmcont += pi_perf_read (PI_PERF_TCDM_CONT); \
    	_ldstall  += pi_perf_read (PI_PERF_LD_STALL); \
        _imiss    += pi_perf_read (PI_PERF_IMISS); \
      }

#define EXIT_STATS_LOOP()  \
    } \
    printf("[%d] total cycles = %lu\n", pi_core_id(), _cycles/REPEAT); \
    printf("[%d] instructions = %lu\n", pi_core_id(), _instr/REPEAT); \
    printf("[%d] active cycles = %lu\n", pi_core_id(), _active/REPEAT); \
    printf("[%d] external loads (L2+synch) = %lu\n", pi_core_id(), _ldext/REPEAT); \
    printf("[%d] TCDM cont = %lu\n", pi_core_id(), _tcdmcont/REPEAT); \
    printf("[%d] LD stalls = %lu\n", pi_core_id(), _ldstall/REPEAT); \
    printf("[%d] I$ misses = %lu\n", pi_core_id(), _imiss/REPEAT);

#else // PERF not defined

#define INIT_STATS()
#define ENTER_STATS_LOOP()
#define START_STATS()
#define STOP_STATS()
#define EXIT_STATS_LOOP()

#endif  // STATS


#endif
