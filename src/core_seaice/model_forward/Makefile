.SUFFIXES: .F .o

OBJS = 	mpas_seaice_core.o \
	mpas_seaice_core_interface.o

ifeq "$(OPENACC)" "true"
OBJS += mpas_gpu.o
endif

all: $(OBJS)

ifeq "$(OPENACC)" "true"
mpas_seaice_core.o: mpas_gpu.o
endif

mpas_seaice_core_interface.o: mpas_seaice_core.o

clean:
	$(RM) *.o *.i *.mod *.f90

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

	$(FC) $(FFLAGS_noSMP) -c $*.f90 $(FCINCLUDES) -I../../framework -I../../operators -I../../external/esmf_time_f90 -I../column -I../shared
else
	$(FC) $(CPPFLAGS) $(FFLAGS_noSMP) -c $*.F $(CPPINCLUDES) $(FCINCLUDES) -I../../framework -I../../operators -I../../external/esmf_time_f90 -I../column -I../shared
endif
