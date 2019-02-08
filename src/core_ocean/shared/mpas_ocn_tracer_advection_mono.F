! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_tracer_advection_mono
!
!> \brief MPAS monotonic tracer advection with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen
!> \date   October 2017
!> \details
!>  This module contains routines for monotonic advection of tracers using a FCT
!
!-----------------------------------------------------------------------

module ocn_tracer_advection_mono

#ifdef _ADV_TIMERS
   use mpas_timer
#endif
   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_io_units
   use mpas_threading
   use mpas_tracer_advection_helpers

   use ocn_constants

   implicit none
   private
   save

   real (kind=RKIND) :: coef_3rd_order
   integer :: horizOrder
   logical :: vert2ndOrder, vert3rdOrder, vert4thOrder
   logical :: positiveDzDk, monotonicityCheck

   public :: ocn_tracer_advection_mono_tend, &
             ocn_tracer_advection_mono_init

   contains

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine ocn_tracer_advection_mono_tend
!
!> \brief MPAS monotonic tracer horizontal advection tendency with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen
!> \date   October 2017
!> \details
!>  This routine computes the monotonic tracer horizontal advection tendencity using a FCT.
!
!-----------------------------------------------------------------------
   subroutine ocn_tracer_advection_mono_tend(block, tracers, adv_coefs, adv_coefs_3rd, nAdvCellsForEdge, advCellsForEdge, &!{{{
                                              normalThicknessFlux, w, layerThickness, dt, meshPool, &
                                              scratchPool, diagnosticsPool, tend, maxLevelCell, maxLevelEdgeTop, &
                                              highOrderAdvectionMask, edgeSignOnCell, tracerGroupName)

#ifdef MPAS_OPENACC
	  use ocn_gpu_support
#endif
	  type (block_type), intent(in) :: block
      real (kind=RKIND), dimension(:,:,:), intent(in) :: tracers !< Input: current tracer values
      real (kind=RKIND), dimension(:,:), intent(in) :: adv_coefs !< Input: Advection coefficients for 2nd order advection
      real (kind=RKIND), dimension(:,:), intent(in) :: adv_coefs_3rd !< Input: Advection coeffs for mising 3rd/4th order advection
      integer, dimension(:), intent(in) :: nAdvCellsForEdge !< Input: Number of advection cells for each edge
      integer, dimension(:,:), intent(in) :: advCellsForEdge !< Input: List of advection cells for each edge
      real (kind=RKIND), dimension(:,:), intent(in) :: normalThicknessFlux !< Input: Thichness weighted velocitiy
      real (kind=RKIND), dimension(:,:), intent(in) :: w !< Input: Vertical velocity
      real (kind=RKIND), dimension(:,:), intent(in) :: layerThickness !< Input: Thickness
      real (kind=RKIND), intent(in) :: dt !< Input: Timestep
      type (mpas_pool_type), intent(in), pointer :: meshPool !< Input: Mesh information
      type (mpas_pool_type), intent(in) :: scratchPool !< Input: Scratch fields
      type (mpas_pool_type), intent(in) :: diagnosticsPool !< Input: pool for traceradvection budget term
      real (kind=RKIND), dimension(:,:,:), intent(inout) :: tend !< Input/Output: Tracer tendency

      ! These variables are passed in to match the version of the interface in operators.
      integer, dimension(:), pointer :: maxLevelCell !< Input: Index to max level at cell center
      integer, dimension(:), pointer :: maxLevelEdgeTop !< Input: Index to max level at edge with non-land cells on both sides
      integer, dimension(:,:), pointer :: highOrderAdvectionMask !< Input: Mask for high order advection
      integer, dimension(:, :), pointer :: edgeSignOnCell !< Input: Sign for flux from edge on each cell.

      character (len=*), intent(in) :: tracerGroupName ! variable to check for tracer budget

      logical, pointer :: config_compute_active_tracer_budgets
	  logical :: isActiveTracers, compute_active_tracer

      integer :: i, iCell, iEdge, k, iTracer, cell1, cell2, nVertLevels, num_tracers, nCells, nEdges, mxLevelCell
      integer :: iblk
      integer, pointer :: maxEdges
      integer, dimension(:), pointer :: nCellsArray, nEdgesArray
      integer, dimension(:), pointer :: nEdgesOnCell
      integer, dimension(:,:), pointer :: cellsOnEdge, cellsOnCell, edgesOnCell

      real (kind=RKIND) :: signedFactor, tracer_new, Hof, upwind_tend,  fin, fout 
      real (kind=RKIND) :: flux_upwind, tracer_min_new, tracer_max_new, tracer_upwind_new, scale_factor
      real (kind=RKIND) :: flux, tracer_weight, invAreaCell1, invAreaCell2
      real (kind=RKIND) :: verticalWeightK, verticalWeightKm1
      real (kind=RKIND), dimension(:), pointer :: dvEdge, areaCell
      real (kind=RKIND), dimension(:,:,:), pointer :: tmp_tend
      real (kind=RKIND), dimension(:,:), pointer :: tracer_cur, work_tend, h_new_inv
      real (kind=RKIND), dimension(:,:), pointer :: tracer_max, tracer_min
      real (kind=RKIND), dimension(:,:), pointer :: flux_incoming, flux_outgoing
      real (kind=RKIND), dimension(:,:), pointer :: high_order_flux
      real (kind=RKIND), dimension(:,:), pointer :: h_prov, h_prov_inv

      type (field2DReal), pointer :: tracerCurField, workTendencyField, hNewInvField, tracerMinField, &
                                     tracerMaxField, fluxIncomingField, fluxOutgoingField, &
                                     hProvInvField, hProvField, highOrderFluxField

      real (kind=RKIND), dimension(:,:,:), pointer :: &
              activeTracerHorizontalAdvectionTendency, &
              activeTracerVerticalAdvectionTendency

      real (kind=RKIND), parameter :: eps = 1.e-10_RKIND

#ifdef _ADV_TIMERS
      call mpas_timer_start('startup')
#endif
      ! Get dimensions
      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_dimension(meshPool, 'maxEdges', maxEdges)
      nVertLevels = size(tracers,dim=2)
      num_tracers = size(tracers,dim=1)

      ! Initialize pointers
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
!      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
!      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
!      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
!      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)

#ifdef MPAS_OPENACC
	  iblk = block%localBlockID + 1 ! localBlockID is base 0
	  call ocn_gpu_get_cellsOnCell(iblk, cellsOnCell)
	  call ocn_gpu_get_cellsOnEdge(iblk, cellsOnEdge)
	  call ocn_gpu_get_nEdgesOnCell(iblk, nEdgesOnCell)
	  call ocn_gpu_get_edgesOnCell(iblk, edgesOnCell)
#else
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
#endif

      call mpas_pool_get_config(ocnConfigs, 'config_compute_active_tracer_budgets', config_compute_active_tracer_budgets)
      compute_active_tracer = config_compute_active_tracer_budgets
      if (config_compute_active_tracer_budgets) then
         call mpas_pool_get_array(diagnosticsPool,'activeTracerHorizontalAdvectionTendency', &
                 activeTracerHorizontalAdvectionTendency)
         call mpas_pool_get_array(diagnosticsPool,'activeTracerVerticalAdvectionTendency', &
                 activeTracerVerticalAdvectionTendency)
      end if

      call mpas_pool_get_field(scratchPool, 'tracerCur', tracerCurField)
      call mpas_pool_get_field(scratchPool, 'workTendency', workTendencyField)
      call mpas_pool_get_field(scratchPool, 'hNewInv', hNewInvField)
      call mpas_pool_get_field(scratchPool, 'tracerMin', tracerMinField)
      call mpas_pool_get_field(scratchPool, 'tracerMax', tracerMaxField)
      call mpas_pool_get_field(scratchPool, 'fluxIncoming', fluxIncomingField)
      call mpas_pool_get_field(scratchPool, 'fluxOutgoing', fluxOutgoingField)
      call mpas_pool_get_field(scratchPool, 'hProvInv', hProvInvField)
      call mpas_pool_get_field(scratchPool, 'hProv', hProvField)
      call mpas_pool_get_field(scratchPool, 'highOrderFlux', highOrderFluxField)

      call mpas_allocate_scratch_field(tracerCurField, .true., .false.)
      call mpas_allocate_scratch_field(workTendencyField, .true., .false.)
      call mpas_allocate_scratch_field(hNewInvField, .true., .false.)
      call mpas_allocate_scratch_field(tracerMinField, .true., .false.)
      call mpas_allocate_scratch_field(tracerMaxField, .true., .false.)
      call mpas_allocate_scratch_field(fluxIncomingField, .true., .false.)
      call mpas_allocate_scratch_field(fluxOutgoingField, .true., .false.)
      call mpas_allocate_scratch_field(hProvInvField, .true., .false.)
      call mpas_allocate_scratch_field(hProvField, .true., .false.)
      call mpas_allocate_scratch_field(highOrderFluxField, .true., .false.)
      call mpas_threading_barrier()

	  isActiveTracers = .false.
	  if ( tracerGroupName == 'activeTracers' ) isActiveTracers = .true.

      ! allocate nCells arrays
      h_prov_inv => hProvInvField % array
      h_prov => hProvField % array
      h_new_inv => hNewInvField % array
      tracer_cur => tracerCurField % array
      tracer_max => tracerMaxField % array
      tracer_min => tracerMinField % array
      work_tend => workTendencyField % array
      flux_incoming => fluxIncomingField % array
      flux_outgoing => fluxOutgoingField % array
      high_order_flux => highOrderFluxField % array

      nCells = nCellsArray( size(nCellsArray) )
	  mxLevelCell = maxval(maxLevelCell)

	  !$omp master
	  allocate(tmp_tend(mxLevelCell, maxval(nCellsArray), num_tracers))
	  tmp_tend = 0
	  !$omp end master	  	  

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc enter data create(h_prov_inv,h_new_inv,h_prov,tracer_cur,work_tend,flux_incoming,flux_outgoing, &
	  !$acc             tracer_max,tracer_min,high_order_flux) &
	  !$acc      copyin(tracers,maxLevelCell,layerThickness,areaCell,edgeSignOnCell,maxLevelCell, &
	  !$acc             normalThicknessFlux,dvEdge,w, &
	  !$acc             nAdvCellsForEdge,advCellsForEdge,adv_coefs,highOrderAdvectionMask, &
	  !$acc             activeTracerHorizontalAdvectionTendency, &
	  !$acc             adv_coefs_3rd,maxLevelEdgeTop,activeTracerVerticalAdvectionTendency, tmp_tend)
	  
	  !$omp end master	  	  
#endif

      ! Note: This assumes we are in the first part of the horizontal/
      ! vertical operator splitting, which is true because currently
      ! we dont flip order and horizontal is always first.
      ! See notes in commit 2cd4a89d.
      
#ifdef MPAS_OPENACC
      !$omp master
      !$acc parallel loop gang vector collapse(2) &
      !$acc present(maxLevelCell,layerThickness,areaCell,h_prov_inv,h_new_inv,edgeSignOnCell, &
      !$acc         edgesOnCell, nEdgesOnCell, normalThicknessFlux,dvEdge,h_prov,w) & 
      !$acc private (iEdge, invAreaCell1) async(1)
#else
      !$omp do schedule(runtime) private(k, i, iEdge, invAreaCell1)
#endif
      do iCell = 1, nCells
        do k = 1, mxLevelCell
          invAreaCell1 = 1.0_RKIND / areaCell(iCell)
          if ( k <= maxLevelCell(iCell) ) then
			  h_prov(k, iCell) = layerThickness(k, iCell)
			  do i = 1, nEdgesOnCell(iCell)
				iEdge = edgesOnCell(i, iCell)
				! Provisional layer thickness is after horizontal thickness flux only
				h_prov(k, iCell) = h_prov(k, iCell) + dt * invAreaCell1 * dvEdge(iEdge) * &
												edgeSignOnCell(i, iCell) * normalThicknessFlux(k, iEdge)
			  end do
			  h_prov_inv(k, iCell) = 1.0_RKIND / h_prov(k, iCell)
			  ! New layer thickness is after horizontal and vertical thickness flux
			  h_new_inv(k, iCell) = 1.0_RKIND / (h_prov(k, iCell) - dt * w(k, iCell) + dt * w(k+1, iCell))
		   endif
        end do
      end do
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif

#ifdef _ADV_TIMERS
      call mpas_timer_stop('startup')
#endif

      ! Loop over tracers. One tracer is advected at a time. It is copied into a temporary array in order to improve locality
      do iTracer = 1, num_tracers
#ifdef _ADV_TIMERS
        call mpas_timer_start('cell init')
#endif
        nCells = nCellsArray( size(nCellsArray) )
        ! Initialize variables for use in this iTracer iteration
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) present(tracers,tracer_cur,work_tend,flux_incoming,flux_outgoing) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(k)
#endif        
        do iCell = 1, nCells
          do k=1, nVertLevels
            tracer_cur(k,iCell) = tracers(iTracer,k,iCell)
            work_tend(k, iCell) = 0.0_RKIND
            flux_incoming(k, iCell) = 0.0_RKIND
            flux_outgoing(k, iCell) = 0.0_RKIND
          end do ! k loop
        end do ! iCell loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif
        
#ifdef _ADV_TIMERS
        call mpas_timer_stop('cell init')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('horiz flux')
#endif
        nCells = nCellsArray( 2 )
        !  Compute the high and low order vertical fluxes. Also determine bounds on tracer_cur.
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) present(tracer_cur,tracer_min,tracer_max,maxLevelCell, &
        !$acc          cellsOnCell,nEdgesOnCell) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(k, i)
#endif        
        do iCell = 1, nCells
          do k=1, mxLevelCell
             if ( k <= maxLevelCell(iCell) ) then
                tracer_min(k,iCell) = tracer_cur(k,iCell)
                tracer_max(k,iCell) = tracer_cur(k,iCell)
                ! pull tracer_min and tracer_max from the (horizontal) surrounding cells
          	    do i = 1, nEdgesOnCell(iCell)
          	       if ( k <= min(maxLevelCell(iCell), maxLevelCell(cellsOnCell(i,iCell))) ) then
              		  tracer_max(k,iCell) = max(tracer_max(k,iCell),tracer_cur(k, cellsOnCell(i,iCell)))
              		  tracer_min(k,iCell) = min(tracer_min(k,iCell),tracer_cur(k, cellsOnCell(i,iCell)))
          	       endif
          	    enddo
             endif
          end do
        end do
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif

        ! Need all the edges around the 1 halo cells and owned cells
        nEdges = nEdgesArray( 3 )
        !  Compute the high order horizontal flux
#ifdef MPAS_OPENACC
		!$omp master
		!$acc parallel loop gang vector collapse(2) present(maxLevelEdgeTop,cellsOnEdge, highOrderAdvectionMask, dvEdge, normalThicknessFlux, &
		!$acc                      high_order_flux, tracer_cur, nAdvCellsForEdge, advCellsForEdge, &
		!$acc                      adv_coefs, adv_coefs_3rd, maxLevelCell) &
		!$acc async(1)
#else
        !$omp do schedule(runtime) private(cell1, cell2, k, tracer_weight, i, iCell, flux_upwind)
#endif        
        do iEdge = 1, nEdges
		   do k = 1, nVertLevels
			  Hof = 0.0
			  flux_upwind = 0.0
			  high_order_flux(k, iEdge) = 0.0_RKIND
		      if ( k <= maxLevelEdgeTop(iEdge) ) then
	       	     cell1 = cellsOnEdge(1, iEdge)
                 cell2 = cellsOnEdge(2, iEdge)
	   
				 tracer_weight = xor(highOrderAdvectionMask(k, iEdge), 1) * (dvEdge(iEdge) * 0.5_RKIND) &
						         * normalThicknessFlux(k, iEdge)

				 Hof = high_order_flux(k, iedge) + tracer_weight * (tracer_cur(k,cell1) &
					   + tracer_cur(k,cell2))

				 flux_upwind = dvEdge(iEdge) * (max(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracer_cur(k,cell1) &
							   + min(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracer_cur(k,cell2))
			  endif
	   
			  do i = 1, nAdvCellsForEdge(iEdge)
				 iCell = advCellsForEdge(i,iEdge)
				 if ( k <= maxLevelCell(iCell) ) then
				    tracer_weight = highOrderAdvectionMask(k, iEdge) * (adv_coefs(i,iEdge) &
									+ coef_3rd_order*sign(1.0_RKIND,normalThicknessFlux(k,iEdge))*adv_coefs_3rd(i,iEdge))

					tracer_weight = normalThicknessFlux(k,iEdge)*tracer_weight
					Hof = Hof + tracer_weight * tracer_cur(k,iCell)
				 endif
			  enddo   
	   
			  high_order_flux(k, iEdge) = Hof - flux_upwind 
		   enddo
        end do ! iEdge loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif
#ifdef _ADV_TIMERS
        call mpas_timer_stop('horiz flux')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('scale factor build')
#endif
        ! Need one halo of cells around owned cells
        nCells = nCellsArray( 2 )
#ifdef MPAS_OPENACC
		!$omp master
		!$acc parallel loop gang vector collapse(2) present(maxLevelCell, maxLevelEdgeTop, nEdgesOnCell, &
   		!$acc				 edgesOnCell, cellsOnEdge, edgeSignOnCell, dvEdge, areaCell, normalThicknessFlux, &
   		!$acc				 tracer_cur, high_order_flux, work_tend, flux_outgoing, flux_incoming, &
   		!$acc				 layerThickness, h_prov_inv, tracer_min, tracer_max) &
   		!$acc async(1)
#else
        !$omp do schedule(runtime) private(k, tracer_max_new, tracer_min_new, tracer_upwind_new, scale_factor, invAreaCell1, i, &
        !$omp                              iEdge, cell1, cell2, flux_upwind, signedFactor)
#endif        
        do iCell = 1, nCells
	      do k = 1, mxLevelCell
	    
			invAreaCell1 = 1.0_RKIND / areaCell(iCell)
			upwind_tend = 0.0_RKIND
			fin = 0.0_RKIND
			fout = 0.0_RKIND

            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)
				if ( k <= maxLevelEdgeTop(iEdge) ) then
					cell1 = cellsOnEdge(1,iEdge)
					cell2 = cellsOnEdge(2,iEdge)
					signedFactor = edgeSignOnCell(i, iCell) * invAreaCell1

					flux_upwind = dvEdge(iEdge) * (max(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracer_cur(k,cell1) &
										  + min(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracer_cur(k,cell2))
					upwind_tend = upwind_tend + signedFactor * flux_upwind
					fout = fout + min(0.0_RKIND, signedFactor * high_order_flux(k, iEdge))
					fin = fin + max(0.0_RKIND, signedFactor * high_order_flux(k, iEdge))
				endif
		     enddo
		     
             ! Here work_tend is the upwind tendency
			 work_tend(k,iCell) = work_tend(k,iCell) + upwind_tend
			 flux_outgoing(k,iCell) = flux_outgoing(k,iCell) + fout
			 flux_incoming(k,iCell) = flux_incoming(k,iCell) + fin
		    
             ! Build the factors for the FCT
             ! Computed using the bounds that were computed previously, and the bounds on the newly updated value
             ! Factors are placed in the flux_incoming and flux_outgoing arrays
			 if ( k <= maxLevelCell(iCell) ) then       
				tracer_min_new = (tracer_cur(k,iCell)*layerThickness(k,iCell) + dt*(work_tend(k,iCell)+flux_outgoing(k,iCell))) &
							   * h_prov_inv(k,iCell)
				tracer_max_new = (tracer_cur(k,iCell)*layerThickness(k,iCell) + dt*(work_tend(k,iCell)+flux_incoming(k,iCell))) &
							   * h_prov_inv(k,iCell)
				tracer_upwind_new = (tracer_cur(k,iCell)*layerThickness(k,iCell) + dt*work_tend(k,iCell)) * h_prov_inv(k,iCell)

				! mrp 170818 WARNING: scale factor is only chosen from horizontal
				! min and max.  This is still monotonic scheme, and in the great majority of cells this creates a
				! tighter bound than if vertical neighbors are included.  This
				! results in a lower scale factor, making advection more diffusive.
				scale_factor = (tracer_max(k,iCell)-tracer_upwind_new)/(tracer_max_new-tracer_upwind_new+eps)
				flux_incoming(k,iCell) = min( 1.0_RKIND, max( 0.0_RKIND, scale_factor) )

				scale_factor = (tracer_upwind_new-tracer_min(k,iCell))/(tracer_upwind_new-tracer_min_new+eps)
				flux_outgoing(k,iCell) = min( 1.0_RKIND, max( 0.0_RKIND, scale_factor) )                      
			 endif          	 
          end do ! k loop
        end do ! iCell loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif
#ifdef _ADV_TIMERS
        call mpas_timer_stop('scale factor build')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('rescale horiz fluxes')
#endif
        ! Need all of the edges around owned cells
        nEdges = nEdgesArray( 2 )
        !  rescale the high order horizontal fluxes
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(cellsOnEdge,maxLevelEdgeTop,high_order_flux,&
        !$acc                       flux_outgoing,flux_incoming) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(cell1, cell2, k, flux)
#endif        
        do iEdge = 1, nEdges
           do k = 1, mxLevelCell        
              flux = 0
              if ( k <= maxLevelEdgeTop(iEdge) ) then
       		     cell1 = cellsOnEdge(1,iEdge)
                 cell2 = cellsOnEdge(2,iEdge)
            	 flux = high_order_flux(k,iEdge)
            	 flux = max(0.0_RKIND,flux) * min(flux_outgoing(k,cell1), flux_incoming(k,cell2)) &
                	    + min(0.0_RKIND,flux) * min(flux_incoming(k,cell1), flux_outgoing(k,cell2))
              endif
              high_order_flux(k,iEdge) = flux
   		   enddo     
        end do ! iEdge loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif
#ifdef _ADV_TIMERS
        call mpas_timer_stop('rescale horiz fluxes')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('flux accumulate')
#endif

        nCells = nCellsArray( 1 )
        ! Accumulate the scaled high order vertical tendencies, and the upwind tendencies
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) &
        !$acc               private(invAreaCell1,signedFactor,iEdge) &
        !$acc               present(areaCell,nEdgesOnCell,edgesOnCell,maxLevelEdgeTop,work_tend, &
        !$acc               edgeSignOnCell,high_order_flux,tracer_cur,tmp_tend,layerThickness, &
        !$acc               h_prov_inv,maxLevelCell,activeTracerHorizontalAdvectionTendency) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(invAreaCell1, signedFactor, i, iEdge, flux, k)
#endif        
        do iCell = 1, nCells
           do k = 1, mxLevelCell
              invAreaCell1 = 1.0_RKIND / areaCell(iCell)
              ! Accumulate the scaled high order horizontal tendencies

	          do i = 1, nEdgesOnCell(iCell)
       	         iEdge = edgesOnCell(i, iCell)
                 if ( k <= maxLevelEdgeTop(iEdge) ) then
  		            signedFactor = invAreaCell1 * edgeSignOnCell(i, iCell)
	                work_tend(k, iCell) = work_tend(k, iCell) + signedFactor * high_order_flux(k, iEdge)
                 endif
              enddo
              
              ! work_tend on the RHS is the total horizontal advection tendency
              ! tracer_cur on LHS is the  provisional tracer after horizontal fluxes only.
              if ( k <= maxLevelCell(iCell) ) then
                 if ( compute_active_tracer .and. isActiveTracers ) then
                    activeTracerHorizontalAdvectionTendency(iTracer,k,iCell) = work_tend(k,iCell)
                 endif
                 tracer_cur(k,iCell) = (tracer_cur(k, iCell)*layerThickness(k, iCell) + dt*work_tend(k, iCell)) &
                                  * h_prov_inv(k, iCell)
                 tmp_tend(k, iCell, iTracer) = tmp_tend(k, iCell, iTracer) + work_tend(k,iCell)
              endif
           end do ! k loop
        end do ! iCell loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif

#ifndef MPAS_OPENACC
        if (config_compute_active_tracer_budgets) then
           if (tracerGroupName == 'activeTracers') then
              !$omp do schedule(runtime) private(k)
              do iCell = 1, nCells
                do k = 1, maxLevelCell(iCell)
                    activeTracerHorizontalAdvectionTendency(iTracer,k,iCell) = work_tend(k,iCell)
                end do
              end do ! iCell loop
              !$omp end do
           end if
        end if
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('flux accumulate')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('monotonic check')
#endif
        if (.false.) then
        !if (monotonicityCheck) then
          nCells = nCellsArray( 1 )
          !build min and max bounds on old and new tracer for check on monotonicity.
          !$omp do schedule(runtime) private(k)
          do iCell = 1, nCells
            do k = 1, maxLevelCell(iCell)
              if(tracer_cur(k,iCell) < tracer_min(k, iCell)-eps) then
                 call mpas_log_write( &
                    'Horizontal minimum out of bounds on tracer: $i $r $r ', &
                    MPAS_LOG_WARN, intArgs=(/iTracer/), realArgs=(/ tracer_min(k, iCell), tracer_cur(k,iCell) /) )
              end if

              if(tracer_cur(k,iCell) > tracer_max(k,iCell)+eps) then
                 call mpas_log_write( &
                    'Horizontal maximum out of bounds on tracer: $i $r $r ', &
                    MPAS_LOG_WARN, intArgs=(/iTracer/), realArgs=(/ tracer_max(k, iCell), tracer_cur(k,iCell) /) )
              end if
            end do
          end do
          !$omp end do
        end if
#ifdef _ADV_TIMERS
        call mpas_timer_stop('monotonic check')
#endif

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  Vertical advection
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#ifdef _ADV_TIMERS
        call mpas_timer_start('cell init')
#endif
        nCells = nCellsArray( size(nCellsArray) )
        ! Initialize variables for use in this iTracer iteration
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) &
        !$acc          present(high_order_flux, work_tend) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(k)
#endif        
        do iCell = 1, nCells
          do k=1, nVertLevels+1
            if( k <= nVertLevels ) then
               work_tend(k, iCell) = 0.0_RKIND
            endif
            high_order_flux(k, iCell) = 0.0_RKIND
          end do ! k loop
        end do ! iCell loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif
#ifdef _ADV_TIMERS
        call mpas_timer_stop('cell init')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('vertical flux')
#endif

        ! Need all owned and 1 halo cells
        nCells = nCellsArray( 2 )
        !  Compute the high and low order vertical fluxes. Also determine bounds on tracer_cur.
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop &
        !$acc             present(tracer_cur, high_order_flux, tracer_max, &
        !$acc                     tracer_min, h_prov, &
        !$acc                     maxLevelCell, nEdgesOnCell, cellsOnCell, &
        !$acc                     w, flux_incoming, flux_outgoing, work_tend) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(k, verticalWeightK, verticalWeightKm1, i, flux_upwind)
#endif        
        do iCell = 1, nCells

          ! Operate on top cell in column
          k = 1
          tracer_max(k,iCell) = max(tracer_cur(k,iCell),tracer_cur(k+1,iCell))
          tracer_min(k,iCell) = min(tracer_cur(k,iCell),tracer_cur(k+1,iCell))

          ! Operate on next-to-top cell in column
          k = max(1, min(maxLevelCell(iCell), 2))
          if ( k >= 2 ) then
            verticalWeightK = h_prov(k-1, iCell) / (h_prov(k, iCell) + h_prov(k-1, iCell))
            verticalWeightKm1 = h_prov(k, iCell) / (h_prov(k, iCell) + h_prov(k-1, iCell))
            high_order_flux(k,iCell) = w(k,iCell)*(verticalWeightK*tracer_cur(k,iCell)+verticalWeightKm1 &
                *tracer_cur(k-1,iCell))
            tracer_max(k,iCell) = max(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
            tracer_min(k,iCell) = min(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
          end if

          ! Operate on internal cells in column
          if ( vert4thOrder ) then
             do k=3,maxLevelCell(iCell)-1
                high_order_flux(k, iCell) = mpas_tracer_advection_vflux4( tracer_cur(k-2,iCell),tracer_cur(k-1,iCell),  &
                                       tracer_cur(k  ,iCell),tracer_cur(k+1,iCell), w(k,iCell))

                tracer_max(k,iCell) = max(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
                tracer_min(k,iCell) = min(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
             end do
          else if ( vert3rdOrder ) then
             do k=3,maxLevelCell(iCell)-1
                high_order_flux(k, iCell) = mpas_tracer_advection_vflux3( tracer_cur(k-2,iCell),tracer_cur(k-1,iCell),  &
                                       tracer_cur(k  ,iCell),tracer_cur(k+1,iCell), w(k,iCell), coef_3rd_order )

                tracer_max(k,iCell) = max(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
                tracer_min(k,iCell) = min(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
             end do
          else if ( vert2ndOrder ) then
             do k=3,maxLevelCell(iCell)-1
                verticalWeightK = h_prov(k-1, iCell) / (h_prov(k, iCell) + h_prov(k-1, iCell))
                verticalWeightKm1 = h_prov(k, iCell) / (h_prov(k, iCell) + h_prov(k-1, iCell))
                high_order_flux(k,iCell) = w(k,iCell) * (verticalWeightK * tracer_cur(k,iCell) + verticalWeightKm1 &
                                              * tracer_cur(k-1,iCell))

                tracer_max(k,iCell) = max(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
                tracer_min(k,iCell) = min(tracer_cur(k-1,iCell),tracer_cur(k,iCell),tracer_cur(k+1,iCell))
             end do
          end if

          ! Operate on deepest vertical cell in column.
          k = max(1, maxLevelCell(iCell))
          verticalWeightK = h_prov(k-1, iCell) / (h_prov(k, iCell) + h_prov(k-1, iCell))
          verticalWeightKm1 = h_prov(k, iCell) / (h_prov(k, iCell) + h_prov(k-1, iCell))
          high_order_flux(k,iCell) = w(k,iCell)*(verticalWeightK*tracer_cur(k,iCell)+verticalWeightKm1*tracer_cur(k-1,iCell))
          tracer_max(k,iCell) = max(tracer_cur(k,iCell),tracer_cur(k-1,iCell))
          tracer_min(k,iCell) = min(tracer_cur(k,iCell),tracer_cur(k-1,iCell))

          !  low order upwind vertical flux (monotonic and diffused)
          !  Remove low order flux from the high order flux.
          !  Store left over high order flux in high_order_flux array.
          !  Upwind fluxes are accumulated in work_tend
          do k = 2, maxLevelCell(iCell)
            flux_upwind = min(0.0_RKIND,w(k,iCell))*tracer_cur(k-1,iCell) + max(0.0_RKIND,w(k,iCell))*tracer_cur(k,iCell)
            work_tend(k-1,iCell) = work_tend(k-1,iCell) + flux_upwind
            work_tend(k  ,iCell) = work_tend(k  ,iCell) - flux_upwind
            high_order_flux(k,iCell) = high_order_flux(k,iCell) - flux_upwind
          end do ! k loop

          ! flux_incoming contains the total remaining high order flux into iCell
          !          it is positive.
          ! flux_outgoing contains the total remaining high order flux out of iCell
          !           it is negative
          do k = 1, maxLevelCell(iCell)
            flux_incoming(k, iCell) = max(0.0_RKIND, high_order_flux(k+1, iCell)) &
                                    - min(0.0_RKIND, high_order_flux(k, iCell))
            flux_outgoing(k, iCell) = min(0.0_RKIND, high_order_flux(k+1, iCell)) &
                                    - max(0.0_RKIND, high_order_flux(k, iCell))
          end do ! k Loop
        end do ! iCell Loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('vertical flux')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('scale factor build')
#endif
        ! Need one halo of cells around owned cells
        nCells = nCellsArray( 2 )
#ifdef MPAS_OPENACC
		!$omp master
		!$acc parallel loop gang vector collapse(2) &
		!$acc          present(maxLevelCell,tracer_cur,h_prov,work_tend,flux_incoming, &
		!$acc                  flux_outgoing,h_new_inv,tracer_max,tracer_min) &
		!$acc async(1)
#else
        !$omp do schedule(runtime) private(k, tracer_max_new, tracer_min_new, tracer_upwind_new, scale_factor, i)
#endif        
        do iCell = 1, nCells

          ! Build the factors for the FCT
          ! Computed using the bounds that were computed previously, and the bounds on the newly updated value
          ! Factors are placed in the flux_incoming and flux_outgoing arrays
          do k = 1, mxLevelCell
            if ( k <= maxLevelCell(iCell) ) then
               ! work_tend on the RHS is the upwind tendency
               tracer_min_new = (tracer_cur(k,iCell)*h_prov(k,iCell) + dt*(work_tend(k,iCell)+flux_outgoing(k,iCell))) &
                              * h_new_inv(k,iCell)
               tracer_max_new = (tracer_cur(k,iCell)*h_prov(k,iCell) + dt*(work_tend(k,iCell)+flux_incoming(k,iCell))) &
                              * h_new_inv(k,iCell)
               tracer_upwind_new = (tracer_cur(k,iCell)*h_prov(k,iCell) + dt*work_tend(k,iCell)) * h_new_inv(k,iCell)

               scale_factor = (tracer_max(k,iCell)-tracer_upwind_new)/(tracer_max_new-tracer_upwind_new+eps)
               flux_incoming(k,iCell) = min( 1.0_RKIND, max( 0.0_RKIND, scale_factor) )

               scale_factor = (tracer_upwind_new-tracer_min(k,iCell))/(tracer_upwind_new-tracer_min_new+eps)
               flux_outgoing(k,iCell) = min( 1.0_RKIND, max( 0.0_RKIND, scale_factor) )
            endif
          end do ! k loop
        end do ! iCell loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('scale factor build')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('flux accumulate')
#endif

        nCells = nCellsArray( 1 )
        ! Accumulate the scaled high order vertical tendencies, and the upwind tendencies
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop &
        !$acc          present(maxLevelCell,flux_outgoing,flux_incoming,high_order_flux, &
        !$acc                  work_tend, tmp_tend) &
        !$acc async(1)
#else
        !$omp do schedule(runtime) private(flux, k)
#endif        
        do iCell = 1, nCells
          ! rescale the high order vertical flux
          do k = 2, maxLevelCell(iCell)
            flux =  high_order_flux(k,iCell)
            flux = max(0.0_RKIND,flux) * min(flux_outgoing(k  ,iCell), flux_incoming(k-1,iCell)) &
                 + min(0.0_RKIND,flux) * min(flux_outgoing(k-1,iCell), flux_incoming(k  ,iCell))
            high_order_flux(k,iCell) = flux
          end do ! k loop

          do k = 1,maxLevelCell(iCell)
            ! work_tend on the RHS is the upwind tendency
            ! work_tend on the LHS is the total vertical advection tendency
            work_tend(k, iCell) = work_tend(k, iCell) + (high_order_flux(k+1, iCell) &
                                  - high_order_flux(k, iCell))
            tmp_tend(k, iCell, iTracer) = tmp_tend(k, iCell, iTracer) + work_tend(k, iCell) 
          end do ! k loop

        end do ! iCell loop
#ifdef MPAS_OPENACC
	  !$omp end master
#else	  
      !$omp end do
#endif

        if (config_compute_active_tracer_budgets) then
           if (tracerGroupName == 'activeTracers') then
#ifdef MPAS_OPENACC
      	      !$omp master
              !$acc parallel loop gang vector collapse(2) &
        	  !$acc               present(maxLevelCell,activeTracerVerticalAdvectionTendency, work_tend) &
        	  !$acc async(1)
#else        	  
              !$omp do schedule(runtime) private(k)
#endif              
              do iCell = 1, nCells
           		 do k = 1, mxLevelCell
              	    if ( k <= maxLevelCell(iCell) ) then
                       activeTracerVerticalAdvectionTendency(iTracer,k,iCell) = work_tend(k,iCell)
                    endif
                end do
              end do ! iCell loop
#ifdef MPAS_OPENACC
	          !$omp end master
#else	  
              !$omp end do
#endif
           end if
        end if
        
#ifdef _ADV_TIMERS
        call mpas_timer_stop('flux accumulate')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('monotonic check')
#endif
        if (.false.) then
        !if (monotonicityCheck) then
          nCells = nCellsArray( 1 )
          ! Check for monotonicity of new tracer value
          !$omp do schedule(runtime) private(k)
          do iCell = 1, nCells
            do k = 1, maxLevelCell(iCell)
              ! work_tend on the RHS is the total vertical advection tendency
              tracer_new = (tracer_cur(k, iCell)*h_prov(k, iCell) + dt * work_tend(k, iCell)) * h_new_inv(k, iCell)

              if(tracer_new < tracer_min(k, iCell)-eps) then
                 call mpas_log_write( &
                    'Vertical minimum out of bounds on tracer: $i $i $i $r $r ', &
                    MPAS_LOG_WARN, intArgs=(/iTracer, k, iCell/), realArgs=(/ tracer_min(k, iCell), tracer_new /) )
              end if

              if(tracer_new > tracer_max(k,iCell)+eps) then
                 call mpas_log_write( &
                    'Vertical maximum out of bounds on tracer: $i $i $i $r $r ', &
                    MPAS_LOG_WARN, intArgs=(/iTracer, k, iCell/), realArgs=(/ tracer_max(k, iCell), tracer_new /) )
              end if
            end do
          end do
          !$omp end do
        end if
#ifdef _ADV_TIMERS
        call mpas_timer_stop('monotonic check')
#endif
      end do ! iTracer loop

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
      !$acc update host(tmp_tend, activeTracerVerticalAdvectionTendency)            
	  !$omp end master
#endif
      
      nCells = nCellsArray( 1 )

	  !$omp parallel do collapse(3) private(iCell,k,iTracer)
      do iTracer = 1, num_tracers
      do iCell = 1, nCells
         do k = 1, mxLevelCell
            if ( k <= maxLevelCell(iCell) ) then
               tend(iTracer, k, iCell) = tend(iTracer, k, iCell) + tmp_tend(k, iCell, iTracer)
            endif
		 enddo
	  enddo
	  enddo
	  !$omp end parallel do

#ifdef _ADV_TIMERS
      call mpas_timer_start('half step')
#endif
#ifdef _ADV_TIMERS
      call mpas_timer_stop('half step')
#endif

#ifdef _ADV_TIMERS
      call mpas_timer_start('deallocates')
#endif
      call mpas_threading_barrier()
      
#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc exit data &
	  !$acc      delete(h_prov_inv,h_new_inv,h_prov,tracer_cur,work_tend,flux_incoming,flux_outgoing, &
	  !$acc             tracer_max,tracer_min,high_order_flux, tracers,maxLevelCell,layerThickness,areaCell,edgeSignOnCell,maxLevelCell, &
	  !$acc             normalThicknessFlux,dvEdge,w, &
	  !$acc             nAdvCellsForEdge,advCellsForEdge,adv_coefs,highOrderAdvectionMask, &
	  !$acc             activeTracerHorizontalAdvectionTendency, &
	  !$acc             adv_coefs_3rd,maxLevelEdgeTop,activeTracerVerticalAdvectionTendency, tmp_tend)
	  !$omp end master
#endif

      
      call mpas_deallocate_scratch_field(tracerCurField, .true.)
      call mpas_deallocate_scratch_field(workTendencyField, .true.)
      call mpas_deallocate_scratch_field(hNewInvField, .true.)
      call mpas_deallocate_scratch_field(fluxIncomingField, .true.)
      call mpas_deallocate_scratch_field(fluxOutgoingField, .true.)
      call mpas_deallocate_scratch_field(hProvInvField, .true.)
      call mpas_deallocate_scratch_field(hProvField, .true.)
      call mpas_deallocate_scratch_field(highOrderFluxField, .true.)

#ifdef MPAS_OPENACC
	  !$omp master
	  deallocate(tmp_tend)
	  !$omp end master
#endif

#ifdef _ADV_TIMERS
      call mpas_timer_stop('deallocates')
#endif

   end subroutine ocn_tracer_advection_mono_tend!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine ocn_tracer_advection_mono_init
!
!> \brief MPAS initialize monotonic tracer advection tendency with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen
!> \date   October 2017
!> \details
!>  This routine initializes the monotonic tracer advection tendencity using a FCT.
!
!-----------------------------------------------------------------------
   subroutine ocn_tracer_advection_mono_init(nHalos, horiz_adv_order, vert_adv_order, coef_3rd_order_in, dzdk_positive, & !{{{
                                             check_monotonicity, err)

      use mpas_dmpar
      integer, intent(in) :: nHalos !< Input: number of halos in current simulation
      integer, intent(in) :: horiz_adv_order !< Input: Order for horizontal advection
      integer, intent(in) :: vert_adv_order !< Input: Order for vertical advection
      real (kind=RKIND), intent(in) :: coef_3rd_order_in !< Input: coefficient for blending advection orders.
      logical, intent(in) :: dzdk_positive !< Input: Logical flag determining if dzdk is positive or negative.
      logical, intent(in) :: check_monotonicity !< Input: Logical flag determining check on monotonicity of tracers
      integer, intent(inout) :: err !< Input/Output: Error Flag

      err = 0

      vert2ndOrder = .false.
      vert3rdOrder = .false.
      vert4thOrder = .false.

      if ( horiz_adv_order == 3) then
          coef_3rd_order = coef_3rd_order_in
      else if(horiz_adv_order == 2 .or. horiz_adv_order == 4) then
          coef_3rd_order = 0.0_RKIND
      end if

      horizOrder = horiz_adv_order

      if (vert_adv_order == 3) then
          vert3rdOrder = .true.
      else if (vert_adv_order == 4) then
          vert4thOrder = .true.
      else
          vert2ndOrder = .true.
          if(vert_adv_order /= 2) then
             call mpas_log_write( &
                'Invalid value for vert_adv_order, defaulting to 2nd order', &
                MPAS_LOG_WARN)
          end if
      end if

      if (nHalos < 3) then
         call mpas_log_write( &
            'Monotonic advection cannot be used with less than 3 halos.', &
            MPAS_LOG_CRIT)
      end if

      positiveDzDk = dzdk_positive
      monotonicityCheck = check_monotonicity

   end subroutine ocn_tracer_advection_mono_init!}}}

end module ocn_tracer_advection_mono

