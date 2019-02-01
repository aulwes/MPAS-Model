.SUFFIXES: .F .o

OBJS = mpas_ocn_core.o \
       mpas_ocn_core_interface.o

ifeq "$(OPENACC)" "true"
OBJS += mpas_gpu.o
endif

all: core_ocean

core_ocean: $(OBJS) 

ifeq "$(OPENACC)" "true"
mpas_ocn_core.o: mpas_gpu.o
else
mpas_ocn_core.o:
endif

mpas_ocn_core_interface.o: mpas_ocn_core.o

clean:
	$(RM) *.o *.mod *.f90

ifneq (,$(findstring CPRIBM,$(CPPFLAGS)))
FFLAGS_noSMP := $(filter-out -qsmp%,$(FFLAGS))
else
FFLAGS_noSMP := $(FFLAGS)
endif

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPINCLUDES) -c $<

.F.o:
	$(RM) $@ $*.mod
ifeq "$(GEN_F90)" "true"
	$(CPP) $(CPPFLAGS) $(CPPINCLUDES) $< > $*.f90
	$(FC) $(FFLAGS_noSMP) -c $*.f90 $(FCINCLUDES)
else
	$(FC) $(CPPFLAGS) $(FFLAGS_noSMP) -c $*.F $(CPPINCLUDES) $(FCINCLUDES)
endif
