! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_tracer_advection_mono
!
!> \brief MPAS monotonic tracer advection with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen, Phil Jones
!> \date   October 2017, updated May 2019
!> \details
!>  This module contains routines for monotonic advection of tracers
!>  using a Flux Corrected Transport (FCT) algorithm
!
!-------------------------------------------------------------------------------

module ocn_tracer_advection_mono

   ! module includes
#ifdef _ADV_TIMERS
   use mpas_timer
#endif
   use mpas_kind_types
   use mpas_derived_types
   use mpas_dmpar
   use mpas_pool_routines
   use mpas_io_units
   use mpas_threading
   use mpas_tracer_advection_helpers

   use ocn_constants

   implicit none
   private
   save

   ! module private variables
   real (kind=RKIND) ::  &
      coef3rdOrder        !< high-order horizontal coefficient
   logical ::            &
      monotonicityCheck   !< flag to check monotonicity
   integer ::            &
      vertOrder           !< choice of order for vertical advection
   integer, parameter :: &! enum for supported vertical algorithm order
      vertOrder2 = 2,    &!< 2nd order scheme
      vertOrder3 = 3,    &!< 3rd order scheme
      vertOrder4 = 4      !< 4th order scheme

   ! These temporary allocatable arrays will eventually be moved back into
   ! the tendency routine, but must be declared here for now so that they
   ! retain the shared attribute. Once the threading model has been changed,
   ! these should become thread private again with reduced size

   real (kind=RKIND), dimension(:,:), allocatable :: &
      tracerCur,     &! reordered current tracer
      tracerMax,     &! max tracer in neighbors for limiting
      tracerMin,     &! min tracer in neighbors for limiting
      hNewInv,       &! inverse of new layer thickness
      hProv,         &! provisional layer thickness
      hProvInv,      &! inverse of provisional layer thickness
      flxIn,         &! flux coming into each cell
      flxOut,        &! flux going out of each cell
      workTend,      &! temp for holding some tendency values
      lowOrderFlx,   &! low order flux for FCT
      highOrderFlx    ! high order flux for FCT

   ! public method interfaces
   public :: ocn_tracer_advection_mono_tend, &
             ocn_tracer_advection_mono_init

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

   contains

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine ocn_tracer_advection_mono_tend
!
!> \brief MPAS monotonic tracer horizontal advection tendency with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen, Phil Jones
!> \date   October 2017, updated May 2019
!> \details
!>  This routine computes the monotonic tracer horizontal advection tendency
!>  using a flux-corrected transport (FCT) algorithm.
!
!-------------------------------------------------------------------------------

   subroutine ocn_tracer_advection_mono_tend(tend, tracers, layerThickness,    &
                                             normalThicknessFlux, w, dt,       &
                                             advCoefs, advCoefs3rd,            &
                                             nAdvCellsForEdge, advCellsForEdge,&
                                             maxLevelCell, maxLevelEdgeTop,    &
                                             highOrderAdvectionMask,           &
                                             edgeSignOnCell, meshPool,         &
                                             diagnosticsPool, computeBudgets)!{{{

      !*** Input/Output parameters

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
         tend            !< [in,out] Tracer tendency to which advection added

      !*** Input parameters

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         tracers             !< [in] Current tracer values
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickness      !< [in] Thickness
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalThicknessFlux !< [in] Thichness weighted velocitiy
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         w                   !< [in] Vertical velocity
      real (kind=RKIND), intent(in) :: &
         dt                  !< [in] Timestep
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         advCoefs            !< [in] Coefficients for 2nd order advection
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         advCoefs3rd         !< [in] Coeffs for missing 3rd/4th order advection
      integer, dimension(:), intent(in) :: &
         nAdvCellsForEdge    !< [in] Number of advection cells for each edge
      integer, dimension(:,:), intent(in) :: &
         advCellsForEdge     !< [in] List of advection cells for each edge
      integer, dimension(:), intent(in) :: &
         maxLevelCell        !< [in] Index to max level at cell center
      integer, dimension(:), intent(in) :: &
         maxLevelEdgeTop     !< [in] Indx max edge lvl with both non-land cells
      real(kind=RKIND), dimension(:,:), intent(in) :: &
         highOrderAdvectionMask !< [in] Mask for high order advection
      real(kind=RKIND), dimension(:,:), intent(in) :: &
         edgeSignOnCell         !< [in] Sign for flux from edge on each cell
      type (mpas_pool_type), intent(in) :: &
         meshPool               !< [in] Mesh information
      type (mpas_pool_type), intent(in) :: &
         diagnosticsPool        !< [in] pool for traceradvection budget term
      logical, intent(in) :: &
         computeBudgets         !< [in] Flag to compute active tracer budgets

      !*** Local variables

      integer ::          &
         i, iCell, iEdge, &! horz indices
         mxlvl,     &! max value in maxLevelCell
         cell1, cell2,    &! neighbor cell indices
         nCells, nEdges,  &! numbers of cells or edges
         k,k2,kmax,       &! vert index variants
         nVertLevels,     &! total num vertical levels
         iTracer,         &! tracer index
         numTracers        ! total number of tracers

      ! pointers for mesh variables retrieved from mesh pool
      integer, dimension(:), pointer, contiguous :: &
         nCellsArray,   &! number of cells for each halo depth
         nEdgesArray,   &! number of edges for each halo depth
         nEdgesOnCell    ! number of edges on each cell
      integer, dimension(:,:), pointer, contiguous :: &
         cellsOnEdge,   &! indices of cells on edges
         cellsOnCell,   &! indices of cells on cells
         edgesOnCell     ! indices of edges on cells
      real (kind=RKIND), dimension(:),   pointer, contiguous :: &
         dvEdge,            &! edge metric
         areaCell            ! cell area

      real (kind=RKIND) ::  &
         fluxUpwind,        &
         fin,               &
         fout,              &
         inv12,             &
         inv,               &
         upwindTend

      real (kind=RKIND) ::  &
         signedFactor,      &! temp factor including flux sign
         tracerNew,         &! updated tracer
         tracerMinNew,      &! updated tracer minimum
         tracerMaxNew,      &! updated tracer maximum
         tracerUpwindNew,   &! tracer updated with upwind flx
         scaleFactor,       &! factor for normalizing fluxes
         flux,              &! flux temporary
         tracerWeight,      &! tracer weighting temporary
         invAreaCell1,      &! inverse cell area
         invAreaCell2,      &! inverse cell area
         verticalWeightK,   &! vertical weighting
         verticalWeightKm1, &! vertical weighting
         coef1, coef3        ! temporary coefficients

      real (kind=RKIND), dimension(size(tracers, dim=2)) :: &
         wgtTmp,            &! vertical temporaries for
         flxTmp, sgnTmp      !   high-order flux computation

      ! pointers for tendencies used in diagnostic budget computation
      real (kind=RKIND), dimension(:,:,:), pointer, contiguous :: &
         activeTracerHorizontalAdvectionTendency, &
         activeTracerVerticalAdvectionTendency,   &
         activeTracerHorizontalAdvectionEdgeFlux, &
         activeTracerVerticalAdvectionTopFlux

      real (kind=RKIND), parameter :: &
         eps = 1.e-10_RKIND  ! small number to avoid numerical difficulties

      ! end of preamble
      !----------------
      ! begin code

#ifdef _ADV_TIMERS
      call mpas_timer_start('startup')
#endif
      ! Get mesh data
      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)

      if (computeBudgets) then
         call mpas_pool_get_array(diagnosticsPool,         &
                'activeTracerVerticalAdvectionTopFlux',    &
                 activeTracerVerticalAdvectionTopFlux)
         call mpas_pool_get_array(diagnosticsPool,         &
                'activeTracerHorizontalAdvectionEdgeFlux', &
                 activeTracerHorizontalAdvectionEdgeFlux)
         call mpas_pool_get_array(diagnosticsPool,         &
                'activeTracerHorizontalAdvectionTendency', &
                 activeTracerHorizontalAdvectionTendency)
         call mpas_pool_get_array(diagnosticsPool,         &
                'activeTracerVerticalAdvectionTendency',   &
                 activeTracerVerticalAdvectionTendency)
      end if

      ! Get dimensions
      nVertLevels = size(tracers,dim=2)
      numTracers  = size(tracers,dim=1)
      nCells = nCellsArray(size(nCellsArray))
      nEdges = nEdgesArray(size(nEdgesArray))
	  mxlvl = maxval(maxLevelCell)

      ! allocate temporary arrays
      allocate(tracerCur   (nVertLevels  ,nCells+1), &
               tracerMin   (nVertLevels  ,nCells), &
               tracerMax   (nVertLevels  ,nCells), &
               hNewInv     (nVertLevels  ,nCells), &
               hProv       (nVertLevels  ,nCells), &
               hProvInv    (nVertLevels  ,nCells), &
               flxIn       (nVertLevels  ,nCells+1), &
               flxOut      (nVertLevels  ,nCells+1), &
               workTend    (nVertLevels  ,nCells+1), &
               lowOrderFlx (nVertLevels+1,max(nCells,nEdges)+1), &
               highOrderFlx(nVertLevels+1,max(nCells,nEdges)+1))

#ifdef MPAS_OPENACC
		  !$acc enter data create(hProvInv,hNewInv,hProv,tracerCur,workTend,flxIn,flxOut, &
		  !$acc             tracerMax,tracerMin,highOrderFlx,lowOrderFlx) &
		  !$acc      copyin(tracers,layerThickness, advCoefs, advCoefs3rd, &
		  !$acc             normalThicknessFlux,w, &
		  !$acc             tend,nAdvCellsForEdge,advCellsForEdge) async(1)

      if (computeBudgets) then
          !$acc enter data create( &
		  !$acc             activeTracerHorizontalAdvectionTendency, &
          !$acc             activeTracerVerticalAdvectionTendency, &
          !$acc             activeTracerVerticalAdvectionTopFlux, &
          !$acc             activeTracerHorizontalAdvectionEdgeFlux) async(1)
      endif
#endif

      ! Compute some provisional layer thicknesses
      ! Note: This assumes we are in the first part of the horizontal/
      ! vertical operator splitting, which is true because currently
      ! we dont flip order and horizontal is always first.
      ! See notes in commit 2cd4a89d.

#ifdef MPAS_OPENACC
      !$omp master
      !$acc parallel loop gang vector collapse(2) &
      !$acc present(maxLevelCell,layerThickness,areaCell,hProvInv,hNewInv,edgeSignOnCell, &
      !$acc         edgesOnCell, nEdgesOnCell, normalThicknessFlux,dvEdge,hProv,w) & 
      !$acc private (iEdge, invAreaCell1) async(1)
#else
      !$omp parallel
      !$omp do schedule(runtime) private(invAreaCell1, kmax, k, i, iEdge, signedFactor)
#endif
      do iCell = 1, nCells
        do k = 1, mxlvl
          invAreaCell1 = dt / areaCell(iCell)
          if ( k <= maxLevelCell(iCell) ) then
			  hProv(k, iCell) = layerThickness(k, iCell)
			  do i = 1, nEdgesOnCell(iCell)
				iEdge = edgesOnCell(i, iCell)
				! Provisional layer thickness is after horizontal thickness flux only
				hProv(k, iCell) = hProv(k, iCell) + invAreaCell1 * dvEdge(iEdge) * &
												edgeSignOnCell(i, iCell) * normalThicknessFlux(k, iEdge)
			  end do
			  hProvInv(k, iCell) = 1.0_RKIND / hProv(k, iCell)
			  ! New layer thickness is after horizontal and vertical thickness flux
			  hNewInv(k, iCell) = 1.0_RKIND / (hProv(k, iCell) - dt * w(k, iCell) + dt * w(k+1, iCell))
		   endif
        end do
      end do
#ifndef MPAS_OPENACC
      !$omp end do
      !$omp end parallel
#endif

      !$acc update host(hProv,hProvInv,hNewInv) async(1)

#ifdef _ADV_TIMERS
      call mpas_timer_stop('startup')
#endif

      ! Loop over tracers. One tracer is advected at a time.
      do iTracer = 1, numTracers

#ifdef _ADV_TIMERS
        call mpas_timer_start('cell init')
#endif
        ! reset nCells to include all cells
        nCells = nCellsArray( size(nCellsArray) )

        ! Extract current tracer and change index order to improve locality
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) present(tracers,tracerCur) &
        !$acc async(1)
#else
        !$omp parallel
        !$omp do schedule(runtime) private(k)
#endif        
        do iCell = 1, nCells+1
        do k=1, nVertLevels
           tracerCur(k,iCell) = tracers(iTracer,k,iCell)
        end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif        

        ! Compute the high and low order horizontal fluxes.
#ifdef _ADV_TIMERS
        call mpas_timer_stop('cell init')
        call mpas_timer_start('horiz flux')
#endif

        ! set nCells to first halo level
        nCells = nCellsArray( 2 )

        ! Determine bounds on tracer (tracerMin and tracerMax) from
        ! surrounding cells for later limiting.

#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) present(tracerCur,tracerMin,tracerMax,maxLevelCell, &
        !$acc          cellsOnCell,nEdgesOnCell) private(cell2) &
        !$acc async(1)
#else
        !$omp parallel do schedule(runtime) private(k, i, iCell, cell2)
#endif        
        do iCell = 1, nCells
          do k=1, mxlvl
             if ( k <= maxLevelCell(iCell) ) then
                tracerMin(k,iCell) = tracerCur(k,iCell)
                tracerMax(k,iCell) = tracerCur(k,iCell)
                ! pull tracerMin and tracerMax from the (horizontal) surrounding cells
          	    do i = 1, nEdgesOnCell(iCell)
          	       cell2 = cellsOnCell(i,iCell)
          	       if ( k <= min(maxLevelCell(iCell), maxLevelCell(cell2)) ) then
              		  tracerMax(k,iCell) = max(tracerMax(k,iCell),tracerCur(k, cell2))
              		  tracerMin(k,iCell) = min(tracerMin(k,iCell),tracerCur(k, cell2))
          	       endif
          	    enddo
             endif
          end do
        end do
#ifndef MPAS_OPENACC
        !$omp end parallel do
#endif

        ! Need all the edges around the 1 halo cells and owned cells
        nEdges = nEdgesArray( 3 )

        ! Compute the high order horizontal flux

#ifdef MPAS_OPENACC
		!$acc parallel loop gang vector collapse(2) present(maxLevelEdgeTop,cellsOnEdge, highOrderAdvectionMask, dvEdge, normalThicknessFlux, &
		!$acc                      highOrderFlx, lowOrderFlx, tracerCur, nAdvCellsForEdge, advCellsForEdge, &
		!$acc                      advCoefs, advCoefs3rd, maxLevelCell) &
		!$acc               private(cell1, cell2, k, tracerWeight, i, iCell,coef1,coef3) &
		!$acc async(1)
#else
        !$omp parallel do schedule(runtime) &
        !$omp             private(iEdge, cell1, cell2, k, tracerWeight, i, iCell,coef1,coef3)
#endif        
        do iEdge = 1, nEdges
		   do k = 1, nVertLevels
              ! Compute 3rd or 4th fluxes where requested.
              highOrderFlx(k,iEdge) = 0.0_RKIND
              do i = 1, nAdvCellsForEdge(iEdge)
                 iCell = advCellsForEdge(i,iEdge)
                 if ( k <= maxLevelCell(iCell) ) then
                    coef1 = advCoefs       (i,iEdge)
                    coef3 = advCoefs3rd    (i,iEdge)*coef3rdOrder
                    highOrderFlx(k,iEdge) = highOrderFlx(k,iEdge) + tracerCur(k,iCell)* &
                          normalThicknessFlux(k,iEdge)*highOrderAdvectionMask(k,iEdge)* &
                          (coef1 + coef3*sign(1.0_RKIND,normalThicknessFlux(k,iEdge)))
                 endif
              end do ! i loop over nAdvCellsForEdge

		      if ( k <= maxLevelEdgeTop(iEdge) ) then
	       	     cell1 = cellsOnEdge(1, iEdge)
                 cell2 = cellsOnEdge(2, iEdge)
	   
				 tracerWeight = xor(int(highOrderAdvectionMask(k, iEdge)), 1) * (dvEdge(iEdge) * 0.5_RKIND) &
						         * normalThicknessFlux(k, iEdge)
						         
                 lowOrderFlx(k,iEdge) = dvEdge(iEdge) * &
                                       (max(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracerCur(k,cell1) &
                                       + min(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracerCur(k,cell2))

                 highOrderFlx(k, iEdge) = highOrderFlx(k, iedge) &
                                   + tracerWeight * (tracerCur(k, cell1) &
                                                   + tracerCur(k, cell2)) &
                                   -  lowOrderFlx(k,iEdge)
			  endif              
		   enddo
        end do ! iEdge loop
#ifndef MPAS_OPENACC
        !$omp end parallel do
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('horiz flux')
        call mpas_timer_start('scale factor build')
#endif
        ! Initialize flux arrays for all cells
        nCells = nCellsArray(size(nCellsArray))

#ifdef MPAS_OPENACC
		!$omp master
		!$acc parallel loop gang vector collapse(2) present(workTend,flxIn,flxOut) async(1)
#else
        !$omp parallel
        !$omp do schedule(runtime)
#endif        
        do iCell = 1, nCells+1
        do k=1, nVertLevels
           workTend(k, iCell) = 0.0_RKIND
           flxIn   (k, iCell) = 0.0_RKIND
           flxOut  (k, iCell) = 0.0_RKIND
        end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif

        ! Need one halo of cells around owned cells
        nCells = nCellsArray( 2 )

#ifdef MPAS_OPENACC
		!$omp master
		!$acc parallel loop gang vector collapse(2) present(maxLevelCell, maxLevelEdgeTop, nEdgesOnCell, &
   		!$acc				 edgesOnCell, cellsOnEdge, edgeSignOnCell, areaCell, normalThicknessFlux, &
   		!$acc				 workTend, flxOut, flxIn, hProvInv, lowOrderFlx, highOrderFlx, &
   		!$acc				 layerThickness, tracerMin, tracerMax, tracerCur) &
   		!$acc               private(k, tracerMinNew, tracerMaxNew, tracerUpwindNew, scaleFactor, invAreaCell1, i, &
        !$acc                              upwindTend, fin, fout, iEdge, cell1, cell2, signedFactor) &
   		!$acc async(1)
#else
        !$omp parallel
        !$omp do schedule(runtime) &
        !$omp private(invAreaCell1, i, iEdge, cell1, cell2, signedFactor, k, &
        !$omp         tracerUpwindNew, tracerMinNew, tracerMaxNew, scaleFactor)
#endif        
        do iCell = 1, nCells
	      do k = 1, mxlvl
	    
			invAreaCell1 = 1.0_RKIND / areaCell(iCell)
            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)
				if ( k <= maxLevelEdgeTop(iEdge) ) then
					cell1 = cellsOnEdge(1,iEdge)
					cell2 = cellsOnEdge(2,iEdge)
					signedFactor = edgeSignOnCell(i, iCell) * invAreaCell1

					workTend(k,iCell) = workTend(k,iCell) &
					                  + signedFactor * lowOrderFlx(k,iEdge)
					flxOut(k,iCell) = flxOut(k,iCell) + min(0.0_RKIND, &
					                  signedFactor * highOrderFlx(k, iEdge))
					flxIn(k,iCell) = flxIn(k,iCell) + max(0.0_RKIND, &
					                 signedFactor * highOrderFlx(k, iEdge))
				endif
		     enddo
		     
             ! Build the factors for the FCT
             ! Computed using the bounds that were computed previously,
             ! and the bounds on the newly updated value
             ! Factors are placed in the flxIn and flxOut arrays
			 if ( k <= maxLevelCell(iCell) ) then       
                ! Here workTend is the upwind tendency
				tracerUpwindNew = (tracerCur(k,iCell)*layerThickness(k,iCell) &
				                  + dt*workTend(k,iCell)) * hProvInv(k,iCell)
                tracerMinNew = tracerUpwindNew &
                             + dt*flxOut(k,iCell)*hProvInv(k,iCell)
                tracerMaxNew = tracerUpwindNew &
                             + dt*flxIn (k,iCell)*hProvInv(k,iCell)

                scaleFactor = (tracerMax(k,iCell) - tracerUpwindNew)/ &
                              (tracerMaxNew - tracerUpwindNew + eps)
                flxIn (k,iCell) = min(1.0_RKIND, max(0.0_RKIND, scaleFactor))

                scaleFactor = (tracerUpwindNew - tracerMin(k,iCell))/ &
                              (tracerUpwindNew - tracerMinNew + eps)
                flxOut(k,iCell) = min(1.0_RKIND, max(0.0_RKIND, scaleFactor))
			 endif          	 
          end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('scale factor build')
        call mpas_timer_start('rescale horiz fluxes')
#endif
        ! Need all of the edges around owned cells
        nEdges = nEdgesArray( 2 )
        mxlvl = maxval(maxLevelEdgeTop)
        
        !  rescale the high order horizontal fluxes
#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(cellsOnEdge,maxLevelEdgeTop,highOrderFlx,&
        !$acc                       flxOut,flxIn) &
        !$acc               private(iEdge, cell1, cell2, k, flux) &
        !$acc async(1)
#else
        !$omp parallel
        !$omp do schedule(runtime) private(cell1, cell2, k, flux)
#endif
        do iEdge = 1, nEdges
           do k = 1, mxlvl        
              flux = 0
              if ( k <= maxLevelEdgeTop(iEdge) ) then
       		     cell1 = cellsOnEdge(1,iEdge)
                 cell2 = cellsOnEdge(2,iEdge)
            	 flux = highOrderFlx(k,iEdge)
            	 flux = max(0.0_RKIND,flux) * min(flxOut(k,cell1), flxIn(k,cell2)) &
                	    + min(0.0_RKIND,flux) * min(flxIn(k,cell1), flxOut(k,cell2))
              endif
              highOrderFlx(k,iEdge) = flux
   		   enddo     
        end do ! iEdge loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('rescale horiz fluxes')
        call mpas_timer_start('flux accumulate')
#endif

        nCells = nCellsArray( 1 )
        mxlvl = maxval(maxLevelCell)
        ! Accumulate the scaled high order vertical tendencies, and the upwind tendencies
#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               private(invAreaCell1,signedFactor,iEdge) &
        !$acc               present(areaCell,nEdgesOnCell,edgesOnCell,maxLevelEdgeTop,workTend, &
        !$acc               edgeSignOnCell,highOrderFlx,tracerCur,layerThickness, &
        !$acc               tend,hProvInv,maxLevelCell) &
        !$acc async(1)
#else
        !$omp parallel
        !$omp do schedule(runtime) collapse(2) private(iCell, invAreaCell1, signedFactor, i, iEdge, k)
#endif        
        do iCell = 1, nCells
           do k = 1, mxlvl
              invAreaCell1 = 1.0_RKIND / areaCell(iCell)
              ! Accumulate the scaled high order horizontal tendencies

	          do i = 1, nEdgesOnCell(iCell)
       	         iEdge = edgesOnCell(i, iCell)
                 if ( k <= maxLevelEdgeTop(iEdge) ) then
  		            signedFactor = invAreaCell1 * edgeSignOnCell(i, iCell)
	                workTend(k, iCell) = workTend(k, iCell) &
	                                   + signedFactor * highOrderFlx(k, iEdge)
                 endif
              enddo
              
              ! workTend on the RHS is the total horizontal advection tendency
              ! tracerCur on LHS is the  provisional tracer after horizontal fluxes only.
              if ( k <= maxLevelCell(iCell) ) then
                 tracerCur(k,iCell) = (tracerCur(k, iCell)*layerThickness(k, iCell) &
                                    + dt*workTend(k, iCell)) * hProvInv(k, iCell)
                 !tmpTend(k, iCell, iTracer) = tmpTend(k, iCell, iTracer) + workTend(k,iCell)
                 tend(iTracer,k,iCell) = tend(iTracer,k,iCell) + workTend(k,iCell)
              endif
           end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif


#ifdef _ADV_TIMERS
        call mpas_timer_stop('flux accumulate')
        call mpas_timer_start('advect diags')
#endif
        ! Compute budget and monotonicity diagnostics if needed
        if (computeBudgets) then

           nEdges = nEdgesArray( 2 )
           mxlvl = maxval(maxLevelEdgeTop)
#ifdef MPAS_OPENACC
           !$acc parallel loop gang vector collapse(2) &
           !$acc               present(maxLevelEdgeTop,activeTracerHorizontalAdvectionEdgeFlux, &
           !$acc                       lowOrderFlx,highOrderFlx,dvEdge) async(1)
#else           
           !$omp parallel do schedule(runtime) private(k)
#endif           
           do iEdge = 1,nEdges
           do k = 1,mxlvl
              if ( k <= maxLevelEdgeTop(iEdge) ) then
                 ! Save u*h*T flux on edge for analysis. This variable will be
                 ! divided by h at the end of the time step.
                 activeTracerHorizontalAdvectionEdgeFlux(iTracer,k,iEdge) = &
                    (lowOrderFlx(k,iEdge) + highOrderFlx(k,iEdge))/dvEdge(iEdge)
              endif
           enddo
           enddo
#ifndef MPAS_OPENACC
           !$omp end parallel do
#endif


           nCells = nCellsArray( 1 )
           mxlvl = maxval(maxLevelCell)
#ifdef MPAS_OPENACC
           !$acc parallel loop gang vector collapse(2) &
           !$acc               present(maxLevelCell,activeTracerHorizontalAdvectionTendency, &
           !$acc                       workTend) async(1)
#else           
           !$omp parallel do schedule(runtime) private(k)
#endif           
           do iCell = 1, nCells
           do k = 1,mxlvl
              if ( k <= maxLevelCell(iCell) ) then
                 activeTracerHorizontalAdvectionTendency(iTracer,k,iCell) = &
                                                     workTend(k,iCell)
              endif                                                     
           end do
           end do ! iCell loop
#ifndef MPAS_OPENACC
           !$omp end parallel do
#endif

        end if ! computeBudgets

        if (monotonicityCheck) then
#ifdef MPAS_OPENACC
	       !$acc wait
           !$acc update host(hProv,hProvInv,hNewInv)
#endif
           nCells = nCellsArray( 1 )
           ! Check tracer values against local min,max to detect
           ! non-monotone values and write warning if found
           !$omp parallel
           !$omp do schedule(runtime) private(k)
           do iCell = 1, nCells
           do k = 1, maxLevelCell(iCell)
              if(tracerCur(k,iCell) < tracerMin(k, iCell)-eps) then
                 call mpas_log_write( &
                    'Horizontal minimum out of bounds on tracer: $i $r $r ', &
                    MPAS_LOG_WARN, intArgs=(/iTracer/),                      &
                    realArgs=(/ tracerMin(k, iCell), tracerCur(k,iCell) /) )
              end if

              if(tracerCur(k,iCell) > tracerMax(k,iCell)+eps) then
                 call mpas_log_write( &
                    'Horizontal maximum out of bounds on tracer: $i $r $r ', &
                    MPAS_LOG_WARN, intArgs=(/iTracer/),                      &
                    realArgs=(/ tracerMax(k, iCell), tracerCur(k,iCell) /) )
              end if
           end do
           end do
           !$omp end do
           !$omp end parallel
        end if ! monotonicity check
#ifdef _ADV_TIMERS
        call mpas_timer_stop('advect diags')
#endif

        !-----------------------------------------------------------------------
        !
        !  Horizontal advection complete
        !  Begin vertical advection
        !
        !-----------------------------------------------------------------------

#ifdef _ADV_TIMERS
        call mpas_timer_start('cell init')
#endif
        nCells = nCellsArray( size(nCellsArray) )
        ! Initialize variables for use in this iTracer iteration
#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(highOrderFlx, workTend) async(1)
#else           
        !$omp parallel
        !$omp do schedule(runtime) private(k)
#endif           
        do iCell = 1, nCells
          do k=1, nVertLevels+1
            highOrderFlx(k, iCell) = 0.0_RKIND
            if ( k <= nVertLevels ) then 
               workTend(k, iCell) = 0.0_RKIND
            endif   
          end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('cell init')
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_start('vertical flux')
#endif

        ! Need all owned and 1 halo cells
        nCells = nCellsArray( 2 )

        ! Determine bounds on tracerCur from neighbor values for limiting
#ifdef MPAS_OPENACC
        !$acc parallel loop gang &
        !$acc               present(tracerMax,tracerMin,tracerCur) async(1)
#else           
        !$omp parallel
        !$omp do schedule(runtime) private(k)
#endif           
        do iCell = 1, nCells
          ! take care of top cell
          tracerMax(1,iCell) = max(tracerCur(1,iCell), &
                                   tracerCur(2,iCell))
          tracerMin(1,iCell) = min(tracerCur(1,iCell), &
                                   tracerCur(2,iCell))
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
#endif

		mxlvl = maxval(maxLevelCell)
#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(maxLevelCell,tracerMax,tracerMin,tracerCur) async(1)
#else           
        !$omp do schedule(runtime)
#endif           
        do iCell = 1, nCells
          do k=2,mxlvl
             if ( k <  maxLevelCell(iCell) ) then
                tracerMax(k,iCell) = max(tracerCur(k-1,iCell), &
                                     tracerCur(k  ,iCell), &
                                     tracerCur(k+1,iCell))
                tracerMin(k,iCell) = min(tracerCur(k-1,iCell), &
                                     tracerCur(k  ,iCell), &
                                     tracerCur(k+1,iCell))
             endif
          end do
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
#endif
          
#ifdef MPAS_OPENACC
        !$acc parallel loop gang private(kmax) &
        !$acc               present(maxLevelCell,tracerMax,tracerMin,tracerCur) async(1)
#else           
        !$omp do schedule(runtime)
#endif           
        do iCell = 1, nCells          
          ! finish with bottom cell
          kmax = maxLevelCell(iCell)
          tracerMax(kmax,iCell) = max(tracerCur(kmax  ,iCell), &
                                      tracerCur(kmax-1,iCell))
          tracerMin(kmax,iCell) = min(tracerCur(kmax  ,iCell), &
                                      tracerCur(kmax-1,iCell))
        end do ! cell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif

        ! Compute the high order vertical fluxes based on order selected.
        ! Special cases for top and bottom layers handled in following loop.

        inv12 = 1.0_RKIND / 12.0_RKIND
        
        select case (vertOrder)
        case (vertOrder4)

#ifdef MPAS_OPENACC
          !$acc parallel loop gang vector collapse(2) &
          !$acc               present(highOrderFlx,maxLevelCell,w,tracerCur) async(1)
#else           
          !$omp parallel
          !$omp do schedule(runtime) private(k)
#endif           
          do iCell = 1, nCells
            do k=3,mxlvl
               if ( k < maxLevelCell(iCell) ) then
                  highOrderFlx(k, iCell) = w(k,iCell)*( &
                     7.0_RKIND*(tracerCur(k  ,iCell) + tracerCur(k-1,iCell)) - &
                               (tracerCur(k+1,iCell) + tracerCur(k-2,iCell))) * &
                     inv12
               endif
            end do
          end do ! cell loop
#ifndef MPAS_OPENACC
          !$omp end do
          !$omp end parallel
#endif

        case (vertOrder3)

#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(highOrderFlx,maxLevelCell,w,tracerCur) async(1)
#else           
          !$omp parallel
          !$omp do schedule(runtime) private(k)
#endif           
          do iCell = 1, nCells
            do k=3,mxlvl
               if ( k < maxLevelCell(iCell) ) then
                  highOrderFlx(k, iCell) = (w(k,iCell)* &
                      (7.0_RKIND*(tracerCur(k  ,iCell) + tracerCur(k-1,iCell)) - &
                                 (tracerCur(k+1,iCell) + tracerCur(k-2,iCell))) - &
                        coef3rdOrder*abs(w(k,iCell))* &
                                ((tracerCur(k+1,iCell) - tracerCur(k-2,iCell)) - &
                       3.0_RKIND*(tracerCur(k  ,iCell) - tracerCur(k-1,iCell)))) * &
                       inv12
               endif
            end do
          end do ! cell loop
#ifndef MPAS_OPENACC
          !$omp end do
          !$omp end parallel
#endif

       case (vertOrder2)

#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(hProv,highOrderFlx,maxLevelCell,w,tracerCur) async(1)
#else           
          !$omp parallel
          !$omp do schedule(runtime) private(k, verticalWeightK, verticalWeightKm1)
#endif           
          do iCell = 1, nCells
            do k=3,mxlvl
               if ( k < maxLevelCell(iCell) ) then
                  inv = 1.0_RKIND / (hProv(k  ,iCell) + hProv(k-1,iCell))
                  verticalWeightK   = hProv(k-1,iCell) * inv
                  verticalWeightKm1 = hProv(k  ,iCell) * inv
                  highOrderFlx(k,iCell) = w(k,iCell)* &
                                     (verticalWeightK  *tracerCur(k  ,iCell) + &
                                      verticalWeightKm1*tracerCur(k-1,iCell))
               endif
            end do
          end do ! cell loop
#ifndef MPAS_OPENACC
          !$omp end do
          !$omp end parallel
#endif

        end select ! vertical order

        ! Treat special cases for high order flux
        ! Compute low order upwind vertical flux (monotonic and diffused)
        ! Remove low order flux from the high order flux.
        ! Store left over high order flux in highOrderFlx array.

#ifdef MPAS_OPENACC
        !$acc parallel loop gang private(kmax,k,verticalWeightK,verticalWeightKm1) &
        !$acc               present(hProv,w,lowOrderFlx,highOrderFlx,tracerCur,maxLevelCell) async(1)
#else           
        !$omp parallel
        !$omp do schedule(runtime) private(kmax, verticalWeightK, verticalWeightKm1, k)
#endif           
        do iCell = 1, nCells
          ! Next-to-top cell in column is second-order
          highOrderFlx(1,iCell) = 0.0_RKIND
          lowOrderFlx(1,iCell) = 0.0_RKIND
          kmax = maxLevelCell(iCell)
          if (kmax > 1) then
            verticalWeightK   = hProv(1,iCell) / &
                               (hProv(2,iCell) + hProv(1, iCell))
            verticalWeightKm1 = hProv(2,iCell) / &
                               (hProv(2,iCell) + hProv(1, iCell))
            highOrderFlx(2,iCell) = w(2,iCell)* &
                               (verticalWeightK  *tracerCur(2,iCell) + &
                                verticalWeightKm1*tracerCur(1,iCell))
          end if

          ! Deepest vertical cell in column is second order
          k = max(2,kmax)
          verticalWeightK   = hProv(k-1,iCell) / &
                             (hProv(k  ,iCell) + hProv(k-1,iCell))
          verticalWeightKm1 = hProv(k  ,iCell) / &
                             (hProv(k  ,iCell) + hProv(k-1,iCell))
          highOrderFlx(k,iCell) = w(k,iCell)* &
                             (verticalWeightK  *tracerCur(k  ,iCell) + &
                              verticalWeightKm1*tracerCur(k-1,iCell))
          highOrderFlx(k+1,iCell) = 0.0_RKIND
          lowOrderFlx(kmax+1,iCell) = 0.0_RKIND
	    end do
#ifndef MPAS_OPENACC
        !$omp end do
#endif
        
#ifdef MPAS_OPENACC
        !$acc parallel loop gang private(kmax,k,verticalWeightK,verticalWeightKm1) &
        !$acc               present(lowOrderFlx,w,highOrderFlx,tracerCur,maxLevelCell) async(1)
#else           
        !$omp do schedule(runtime) private(k)
#endif           
        do iCell = 1, nCells
          ! Compute low order (upwind) flux and remove from high order
          do k = 2, mxlvl
             if ( k <= maxLevelCell(iCell) ) then
                lowOrderFlx(k,iCell) = &
                   min(0.0_RKIND,w(k,iCell))*tracerCur(k-1,iCell) + &
                   max(0.0_RKIND,w(k,iCell))*tracerCur(k  ,iCell)
                highOrderFlx(k,iCell) = highOrderFlx(k,iCell) - &
                                     lowOrderFlx(k,iCell)
             endif
          end do ! k loop
        end do ! iCell Loop
#ifndef MPAS_OPENACC
        !$omp end do
#endif

#ifdef MPAS_OPENACC
        !$acc parallel loop gang private(k) &
        !$acc               present(hProv,lowOrderFlx,highOrderFlx,tracerCur,maxLevelCell, &
        !$acc               workTend,flxIn,flxOut) &
        !$acc async(1)
#else           
        !$omp do schedule(runtime) private(k)
#endif           
        do iCell = 1, nCells
          do k = 1, mxlvl
             if ( k <= maxLevelCell(iCell) ) then
                workTend(k,iCell) = lowOrderFlx(k+1,iCell) &
                              - lowOrderFlx(k  ,iCell)
                flxIn (k, iCell) = max(0.0_RKIND, highOrderFlx(k+1, iCell)) &
                             - min(0.0_RKIND, highOrderFlx(k  , iCell))
                flxOut(k, iCell) = min(0.0_RKIND, highOrderFlx(k+1, iCell)) &
                             - max(0.0_RKIND, highOrderFlx(k  , iCell))
             endif
          end do ! k Loop
        end do ! iCell Loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('vertical flux')
        call mpas_timer_start('scale factor build')
#endif
        ! Need one halo of cells around owned cells
        nCells = nCellsArray( 2 )
#ifdef MPAS_OPENACC
		!$omp master
        !$acc parallel loop gang vector collapse(2) &
        !$acc               private(tracerMinNew,tracerMaxNew,tracerUpwindNew,scaleFactor) &
        !$acc               present(hProv,hNewInv,tracerCur,maxLevelCell, &
        !$acc                       workTend,flxIn,flxOut,tracerMax,tracerMin) async(1)
#else           
        !$omp parallel
        !$omp do schedule(runtime) &
        !$omp private(k, tracerMinNew, tracerMaxNew, tracerUpwindNew, scaleFactor)
#endif           
        do iCell = 1, nCells
          ! Build the scale factors to limit flux for FCT
          ! Computed using the bounds that were computed previously,
          ! and the bounds on the newly updated value
          ! Factors are placed in the flxIn and flxOut arrays

          do k = 1, mxlvl
             if ( k <= maxLevelCell(iCell) ) then
                ! workTend on the RHS is the upwind tendency
                tracerMinNew = (tracerCur(k,iCell)*hProv(k,iCell) &
                             + dt*(workTend(k,iCell)+flxOut(k,iCell))) &
                             * hNewInv(k,iCell)
                tracerMaxNew = (tracerCur(k,iCell)*hProv(k,iCell) &
                             + dt*(workTend(k,iCell)+flxIn(k,iCell))) &
                             * hNewInv(k,iCell)
                tracerUpwindNew = (tracerCur(k,iCell)*hProv(k,iCell) &
                                + dt*workTend(k,iCell)) * hNewInv(k,iCell)
    
                scaleFactor = (tracerMax(k,iCell)-tracerUpwindNew)/ &
                              (tracerMaxNew-tracerUpwindNew+eps)
                flxIn (k,iCell) = min(1.0_RKIND, max(0.0_RKIND, scaleFactor))
    
                scaleFactor = (tracerUpwindNew-tracerMin(k,iCell))/ &
                              (tracerUpwindNew-tracerMinNew+eps)
                flxOut(k,iCell) = min(1.0_RKIND, max(0.0_RKIND, scaleFactor))
             endif
          end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif        

#ifdef _ADV_TIMERS
        call mpas_timer_stop('scale factor build')
        call mpas_timer_start('flux accumulate')
#endif

        nCells = nCellsArray( 1 )
	    mxlvl = maxval(maxLevelCell)
        ! Accumulate the scaled high order vertical tendencies
        ! and the upwind tendencies
#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               private(flux) &
        !$acc               present(maxLevelCell, &
        !$acc                       flxIn,flxOut,highOrderFlx) async(1)
#else           
        !$omp parallel
        !$omp do schedule(runtime) private(k, flux)
#endif           
        do iCell = 1, nCells
          ! rescale the high order vertical flux
          do k = 2, mxlvl
             if ( k <= maxLevelCell(iCell) ) then
                flux =  highOrderFlx(k,iCell)
                highOrderFlx(k,iCell) = max(0.0_RKIND,flux)*   &
                                    min(flxOut(k  ,iCell), &
                                        flxIn (k-1,iCell)) &
                                  + min(0.0_RKIND,flux)*   &
                                    min(flxOut(k-1,iCell), &
                                        flxIn (k  ,iCell))
             endif                                        
          end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
#endif        
                
#ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc               present(maxLevelCell, highOrderFlx, &
        !$acc                       tend,workTend) async(1)
#else           
        !$omp do schedule(runtime) private(k)
#endif           
        do iCell = 1, nCells
          do k = 1, mxlvl
             if ( k <= maxLevelCell(iCell) ) then
                ! workTend on the RHS is the upwind tendency
                ! workTend on the LHS is the total vertical advection tendency
                workTend(k, iCell) = workTend(k, iCell)       &
                               + (highOrderFlx(k+1,iCell) &
                                - highOrderFlx(k  ,iCell))
                !tmpTend(k, iCell, iTracer) = tmpTend(k, iCell, iTracer) + workTend(k,iCell)
                tend(iTracer,k,iCell) = tend(iTracer,k,iCell) + workTend(k,iCell)
             endif
          end do ! k loop
        end do ! iCell loop
#ifndef MPAS_OPENACC
        !$omp end do
        !$omp end parallel
#endif        

        ! Compute advection diagnostics and monotonicity checks if requested
#ifdef _ADV_TIMERS
        call mpas_timer_stop('flux accumulate')
        call mpas_timer_start('advect diags')
#endif
        if (computeBudgets) then
           mxlvl = maxval(maxLevelCell)
#ifdef MPAS_OPENACC
	       !$acc parallel loop gang vector collapse(2) &
	       !$acc          present(highOrderFlx,lowOrderFlx,workTend, &
	       !$acc                  activeTracerVerticalAdvectionTopFlux, &
	       !$acc                  activeTracerVerticalAdvectionTendency, &
	       !$acc                  maxLevelCell) async(1)
#else
           !$omp parallel
           !$omp do schedule(runtime) private(k)
#endif
           do iCell = 1, nCells
              do k = 1, mxlvl
                 if ( k <=  maxLevelCell(iCell) ) then
                    if ( k > 1 ) then
                       activeTracerVerticalAdvectionTopFlux(iTracer,k,iCell) = &
                          lowOrderFlx(k,iCell) + highOrderFlx(k,iCell)
                    endif                    
                    activeTracerVerticalAdvectionTendency(iTracer,k,iCell) = &
                       workTend(k,iCell)
                endif
              end do
           end do ! iCell loop
#ifndef MPAS_OPENACC
           !$omp end do
           !$omp end parallel
#endif        
        end if ! computeBudgets

        if (monotonicityCheck) then
          nCells = nCellsArray( 1 )
#ifdef MPAS_OPENACC
	       !$acc wait
           !$acc update host(workTend,tracerCur,tracerMin,tracerMax)	       
#endif
          ! Check for monotonicity of new tracer value
          !$omp parallel
          !$omp do schedule(runtime) private(k, tracerNew)
          do iCell = 1, nCells
          do k = 1, maxLevelCell(iCell)
             ! workTend on the RHS is the total vertical advection tendency
             tracerNew = (tracerCur(k, iCell)*hProv(k, iCell) &
                          + dt*workTend(k, iCell))*hNewInv(k, iCell)

             if (tracerNew < tracerMin(k, iCell)-eps) then
                call mpas_log_write( &
                   'Vertical minimum out of bounds on tracer: $i $i $i $r $r ',&
                   MPAS_LOG_WARN, intArgs=(/iTracer, k, iCell/), &
                   realArgs=(/ tracerMin(k, iCell), tracerNew /) )
             end if

             if (tracerNew > tracerMax(k,iCell)+eps) then
                call mpas_log_write( &
                   'Vertical maximum out of bounds on tracer: $i $i $i $r $r ',&
                    MPAS_LOG_WARN, intArgs=(/iTracer, k, iCell/), &
                    realArgs=(/ tracerMax(k, iCell), tracerNew /) )
             end if
          end do
          end do
          !$omp end do
          !$omp end parallel
        end if ! monotonicity check
#ifdef _ADV_TIMERS
        call mpas_timer_stop('advect diags')
#endif
      end do ! iTracer loop

      !$acc update host(tend) async(1)

#ifdef _ADV_TIMERS
      call mpas_timer_start('deallocates')
#endif

#ifdef MPAS_OPENACC
      !$acc wait
      if (computeBudgets) then
		  !$acc exit data delete( &
		  !$acc             activeTracerHorizontalAdvectionTendency, &
          !$acc             activeTracerVerticalAdvectionTendency, &
          !$acc             activeTracerVerticalAdvectionTopFlux, &
          !$acc             activeTracerHorizontalAdvectionEdgeFlux) async(1)
      endif      

	  !$acc exit data delete(hProvInv,hNewInv,hProv,tracerCur,workTend,flxIn,flxOut, &
	  !$acc             tracerMax,tracerMin,highOrderFlx,lowOrderFlx, &
	  !$acc             tracers,layerThickness, advCoefs, advCoefs3rd, &
	  !$acc             normalThicknessFlux,w, &
	  !$acc             tend,nAdvCellsForEdge,advCellsForEdge) async(1)
#endif

      deallocate(tracerCur,    &
                 tracerMin,    &
                 tracerMax,    &
                 hNewInv,      &
                 hProv,        &
                 hProvInv,     &
                 flxIn,        &
                 flxOut,       &
                 workTend,     &
                 lowOrderFlx,  &
                 highOrderFlx)

#ifdef _ADV_TIMERS
      call mpas_timer_stop('deallocates')
#endif

   end subroutine ocn_tracer_advection_mono_tend!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  routine ocn_tracer_advection_mono_init
!
!> \brief MPAS initialize monotonic tracer advection tendency with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen, Phil Jones
!> \date   October 2017, updated May 2019
!> \details
!>  This routine initializes monotonic tracer advection quantities for
!>  the flux-corrected transport (FCT) algorithm.
!
!-------------------------------------------------------------------------------

   subroutine ocn_tracer_advection_mono_init(nHalos, horizAdvOrder,        &
                                             vertAdvOrder, inCoef3rdOrder, &
                                             checkMonotonicity, err)!{{{

      !*** input parameters

      integer, intent(in) :: &
         nHalos               !< [in] number of halos in current simulation
      integer, intent(in) :: &
         horizAdvOrder        !< [in] order for horizontal advection
      integer, intent(in) :: &
         vertAdvOrder         !< [in] order for vertical advection
      real (kind=RKIND), intent(in) :: &
         inCoef3rdOrder       !< [in] coefficient for blending advection orders
      logical, intent(in) :: &
         checkMonotonicity    !< [in] flag to check monotonicity of tracers

      !*** output parameters

      integer, intent(out) :: &
         err                   !< [out] error flag

      ! end of preamble
      !----------------
      ! begin code

      err = 0 ! initialize error code to success

      ! Check that the halo is wide enough for FCT
      if (nHalos < 3) then
         call mpas_log_write( &
            'Monotonic advection cannot be used with less than 3 halos.', &
            MPAS_LOG_CRIT)
         err = -1
      end if

      ! Set blending coefficient if 3rd order horizontal advection chosen
      select case (horizAdvOrder)
      case (2)
         coef3rdOrder = 0.0_RKIND
      case (3)
         coef3rdOrder = inCoef3rdOrder
      case (4)
         coef3rdOrder = 0.0_RKIND
      case default
         coef3rdOrder = 0.0_RKIND
         call mpas_log_write( &
            'Invalid value for horizontal advection order, defaulting to 2',&
            MPAS_LOG_WARN)
      end select ! horizontal advection order

      ! Set vertical advection order
      select case (vertAdvOrder)
      case (2)
         vertOrder = vertOrder2
      case (3)
         vertOrder = vertOrder3
      case (4)
         vertOrder = vertOrder4
      case default
         vertOrder = vertOrder2
         call mpas_log_write( &
            'Invalid value for vertical advection order, defaulting to 2', &
            MPAS_LOG_WARN)
      end select ! vertical advection order

      ! Set flag for checking monotonicity
      monotonicityCheck = checkMonotonicity

   end subroutine ocn_tracer_advection_mono_init!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

end module ocn_tracer_advection_mono

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
