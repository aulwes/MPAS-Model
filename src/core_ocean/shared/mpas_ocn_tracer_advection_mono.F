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
   use mpas_log

   use ocn_constants
   use ocn_config
   use ocn_mesh

   implicit none
   private
   save

   !----------------------------------------------------------------------------
   ! public methods
   !----------------------------------------------------------------------------

   public :: ocn_tracer_advection_mono_tend, &
             ocn_tracer_advection_mono_init

   !----------------------------------------------------------------------------
   ! module private variables
   !----------------------------------------------------------------------------

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

!*******************************************************************************

   contains

!*******************************************************************************
!
!  routine ocn_tracer_advection_mono_tend
!
!> \brief MPAS monotonic tracer horizontal advection tendency with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen, Phil Jones
!> \date   October 2017, updated May 2019, 2020
!> \details
!>  This routine computes the monotonic tracer horizontal advection tendency
!>  using a flux-corrected transport (FCT) algorithm.
!
!-------------------------------------------------------------------------------

   subroutine ocn_tracer_advection_mono_tend(tend, tracers, layerThickness,    &
                                             normalThicknessFlux, w, dt,       &
                                             computeBudgets,                   &
                                             activeTracerHorzAdvectTend,       &
                                             activeTracerVertAdvectTend,       &
                                             activeTracerHorzAdvectEdgeFlx,    &
                                             activeTracerVertAdvectTopFlx)!{{{

      !-------------------------------------------------------------------------
      ! Input/Output parameters
      !-------------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
         tend            !< [in,out] Tracer tendency to which advection added

      !-------------------------------------------------------------------------
      ! Input parameters
      !-------------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         tracers               !< [in] Current tracer values
      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThickness,      &!< [in] Thickness
         normalThicknessFlux, &!< [in] Thichness weighted velocitiy
         w                     !< [in] Vertical velocity
      real (kind=RKIND), intent(in) :: &
         dt                    !< [in] Timestep
      logical, intent(in) :: &
         computeBudgets        !< [in] Flag to compute active tracer budgets

      !-------------------------------------------------------------------------
      ! Output variables
      !-------------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(out), optional :: &
         activeTracerHorzAdvectTend,    &! tendencies and fluxes for
         activeTracerVertAdvectTend,    &!  budget diagnostics
         activeTracerHorzAdvectEdgeFlx, &
         activeTracerVertAdvectTopFlx

      !-------------------------------------------------------------------------
      ! Local variables
      !-------------------------------------------------------------------------

      integer ::          &
         i, iCell, iEdge, &! horz indices
         cell1, cell2,    &! neighbor cell indices
         nCells, nEdges,  &! numbers of cells or edges
         k,k2,kmax,       &! vert index variants
         iTracer,         &! tracer index
         numTracers        ! total number of tracers

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

      real (kind=RKIND), dimension(:), allocatable :: &
         wgtTmp,            &! vertical temporaries for
         flxTmp, sgnTmp      !   high-order flux computation

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
         tmpTend,       &! local tendency for current tracer
         lowOrderFlx,   &! low order flux for FCT
         highOrderFlx    ! high order flux for FCT

      real (kind=RKIND), parameter :: &
         eps = 1.e-10_RKIND  ! small number to avoid numerical difficulties

      ! end of preamble
      !----------------
      ! begin code

      #ifdef _ADV_TIMERS
      call mpas_timer_start('startup')
      #endif

      ! Get dimensions
      numTracers  = size(tracers,dim=1)

      ! allocate temporary arrays
      allocate(wgtTmp      (nVertLevels), &
               flxTmp      (nVertLevels), &
               sgnTmp      (nVertLevels), &
               tracerCur   (nVertLevels, nCellsAll), &
               tracerMin   (nVertLevels, nCellsAll), &
               tracerMax   (nVertLevels, nCellsAll), &
               hNewInv     (nVertLevels, nCellsAll), &
               hProv       (nVertLevels, nCellsAll), &
               hProvInv    (nVertLevels, nCellsAll), &
               flxIn       (nVertLevels, nCellsAll), &
               flxOut      (nVertLevels, nCellsAll), &
               workTend    (nVertLevels, nCellsAll), &
               tmpTend     (nVertLevels, nCellsAll), &
               lowOrderFlx (nVertLevels+1,max(nCellsAll,nEdgesAll)), &
               highOrderFlx(nVertLevels+1,max(nCellsAll,nEdgesAll)))
      !$acc enter data &
      !$acc   copyin(tracers, layerThickness, normalThicknessFlux, w) &
      !$acc   create(wgtTmp, flxTmp, sgnTmp, tracerCur, tracerMin, &
      !$acc          tracerMax, hNewInv, hProv, hProvInv, flxIn, &
      !$acc          flxOut, workTend, lowOrderFlx, highOrderFlx, &
      !$acc          tmpTend) async(1)

      ! Compute some provisional layer thicknesses
      ! Note: This assumes we are in the first part of the horizontal/
      ! vertical operator splitting, which is true because currently
      ! we dont flip order and horizontal is always first.
      ! See notes in commit 2cd4a89d.

      #ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc    present(areaCell, maxLevelCell, dvEdge, &
      !$acc            layerThickness, normalThicknessFlux, &
      !$acc            nEdgesOnCell, edgesOnCell, edgeSignOnCell, &
      !$acc            hProv, hProvInv, hNewInv, w) &
      !$acc    private(i, k, kmax, iEdge, invAreaCell1, signedFactor) async(1)
      #else
      !$omp parallel 
      !$omp do schedule(runtime) &
      !$omp    private(i, k, kmax, iEdge, invAreaCell1, signedFactor)
      #endif
      do iCell = 1, nCellsAll
        invAreaCell1 = dt/areaCell(iCell)
        kmax = maxLevelCell(iCell)
        do k = 1, kmax
          hProv(k, iCell) = layerThickness(k, iCell)
        end do
        do i = 1, nEdgesOnCell(iCell)
          iEdge = edgesOnCell(i,iCell)
          signedFactor = invAreaCell1*dvEdge(iEdge)*edgeSignOnCell(i,iCell)
          ! Provisional layer thickness is after horizontal thickness flux only
          do k = 1, kmax
            hProv(k, iCell) = hProv(k, iCell) &
                            + signedFactor*normalThicknessFlux(k,iEdge)
          end do
        end do
        ! New layer thickness is after horizontal and vertical thickness flux
        do k = 1, kmax
          hProvInv(k,iCell) = 1.0_RKIND/ hProv(k,iCell)
          hNewInv (k,iCell) = 1.0_RKIND/(hProv(k,iCell) - &
                                         dt*w(k,iCell) + dt*w(k+1, iCell))
        end do
      end do
      #ifndef MPAS_OPENACC
      !$omp end do
      #endif

      !!$acc update host(hProv, hProvInv, hNewInv)

      #ifdef _ADV_TIMERS
      call mpas_timer_stop('startup')
      #endif

      ! Loop over tracers. One tracer is advected at a time.
      do iTracer = 1, numTracers

        #ifdef _ADV_TIMERS
        call mpas_timer_start('cell init')
        #endif

        ! Extract current tracer and change index order to improve locality
        #ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc    present(tracerCur, tracers, tmpTend) &
        !$acc    private(k) async(1)
        #else
        !$omp do schedule(runtime) private(k) 
        #endif
        do iCell = 1, nCellsAll
        do k=1, nVertLevels
           tracerCur(k,iCell) = tracers(iTracer,k,iCell)
           tmpTend  (k,iCell) = 0.0_RKIND
        end do ! k loop
        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        ! Compute the high and low order horizontal fluxes.
        #ifdef _ADV_TIMERS
        call mpas_timer_stop('cell init')
        call mpas_timer_start('horiz flux')
        #endif

        ! set nCells to first halo level
        nCells = nCellsHalo( 1 )

        ! Determine bounds on tracer (tracerMin and tracerMax) from
        ! surrounding cells for later limiting.

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(nEdgesOnCell, cellsOnCell, maxLevelCell, &
        !$acc            tracerCur, tracerMin, tracerMax) &
        !$acc    private(i, k, kmax, cell2) async(1)
        #else
        !$omp do schedule(runtime) private(i, k, kmax, cell2)
        #endif
        do iCell = 1, nCells
          do k=1, maxLevelCell(iCell)
            tracerMin(k,iCell) = tracerCur(k,iCell)
            tracerMax(k,iCell) = tracerCur(k,iCell)
          end do
          do i = 1, nEdgesOnCell(iCell)
            cell2 = cellsOnCell(i,iCell)
            kmax  = min(maxLevelCell(iCell), &
                        maxLevelCell(cell2))
            do k=1, kmax
              tracerMax(k,iCell) = max(tracerMax(k,iCell), &
                                       tracerCur(k,cell2))
              tracerMin(k,iCell) = min(tracerMin(k,iCell), &
                                       tracerCur(k,cell2))
            end do ! k loop
          end do ! i loop over nEdgesOnCell
        end do
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        ! Need all the edges around the 1 halo cells and owned cells
        nEdges = nEdgesHalo( 2 )

        ! Compute the high order horizontal flux

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(maxLevelCell, maxLevelEdgeTop, cellsOnEdge, &
        !$acc            dvEdge, normalThicknessFlux, tracerCur, &
        !$acc            nAdvCellsForEdge, advCellsForEdge, advCoefs, &
        !$acc            advCoefs3rd, highOrderFlx, lowOrderFlx, &
        !$acc            highOrderAdvectionMask) &
        !$acc    private(i, k, cell1, cell2, iCell, coef1, coef3, &
        !$acc            tracerWeight, wgtTmp, sgnTmp, flxTmp) async(1)
        #else
        !$omp do schedule(runtime) &
        !$omp    private(i, k, cell1, cell2, iCell, coef1, coef3, &
        !$omp            tracerWeight, wgtTmp, sgnTmp, flxTmp)
        #endif
        do iEdge = 1, nEdges
          cell1 = cellsOnEdge(1, iEdge)
          cell2 = cellsOnEdge(2, iEdge)

          ! compute some common intermediate factors
          do k = 1, nVertLevels
             wgtTmp(k) = normalThicknessFlux   (k,iEdge)* &
                         highOrderAdvectionMask(k,iEdge)
             sgnTmp(k) = sign(1.0_RKIND, &
                              normalThicknessFlux(k,iEdge))
             flxTmp(k) = 0.0_RKIND
          end do

          ! Compute 3rd or 4th fluxes where requested.
          do i = 1, nAdvCellsForEdge(iEdge)
            iCell = advCellsForEdge(i,iEdge)
            coef1 = advCoefs       (i,iEdge)
            coef3 = advCoefs3rd    (i,iEdge)*coef3rdOrder
            do k = 1, maxLevelCell(iCell)
              flxTmp(k) = flxTmp(k) + tracerCur(k,iCell)* &
                          wgtTmp(k)*(coef1 + coef3*sgnTmp(k))
            end do ! k loop
          end do ! i loop over nAdvCellsForEdge

          do k=1,nVertLevels
             highOrderFlx(k,iEdge) = flxTmp(k)
          end do

          ! Compute 2nd order fluxes where needed.
          ! Also compute low order upwind horizontal flux (monotonic and diffused)
          ! Remove low order flux from the high order flux
          ! Store left over high order flux in highOrderFlx array
          do k = 1, maxLevelEdgeTop(iEdge)
            tracerWeight = iand(int(highOrderAdvectionMask(k, iEdge))+1, 1) &
                         * (dvEdge(iEdge) * 0.5_RKIND)                 &
                         * normalThicknessFlux(k, iEdge)

            lowOrderFlx(k,iEdge) = dvEdge(iEdge) * &
               (max(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracerCur(k,cell1) &
              + min(0.0_RKIND,normalThicknessFlux(k,iEdge))*tracerCur(k,cell2))

            highOrderFlx(k, iEdge) = highOrderFlx(k, iedge) &
                                   + tracerWeight * (tracerCur(k, cell1) &
                                                   + tracerCur(k, cell2))

            highOrderFlx(k,iEdge) = highOrderFlx(k,iEdge) &
                                  -  lowOrderFlx(k,iEdge)
          end do ! k loop
        end do ! iEdge loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('horiz flux')
        call mpas_timer_start('scale factor build')
        #endif

        ! Need one halo of cells around owned cells
        nCells = nCellsHalo( 1 )

        ! Initialize flux arrays

        #ifdef MPAS_OPENACC
        !$acc parallel loop collapse(2) async(1)
        #else
        !$omp do schedule(runtime) private(k)
        #endif
        do iCell = 1, nCells
        do k=1, nVertLevels
           workTend(k, iCell) = 0.0_RKIND
           flxIn   (k, iCell) = 0.0_RKIND
           flxOut  (k, iCell) = 0.0_RKIND
        end do ! k loop
        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(nEdgesOnCell, edgesOnCell, areaCell, &
        !$acc            cellsOnEdge, edgeSignOnCell, workTend, &
        !$acc            maxLevelEdgeTop, maxLevelCell, hProvInv, &
        !$acc            layerThickness, flxOut, flxIn, &
        !$acc            tracerCur, tracerMax, tracerMin, &
        !$acc            lowOrderFlx, highOrderFlx) &
        !$acc    private(i, k, iEdge, cell1, cell2, invAreaCell1, &
        !$acc            signedFactor, scaleFactor, &
        !$acc            tracerUpwindNew, tracerMinNew, tracerMaxNew) async(1)
        #else
        !$omp do schedule(runtime) &
        !$omp    private(i, k, iEdge, cell1, cell2, invAreaCell1, &
        !$omp            signedFactor, scaleFactor, &
        !$omp            tracerUpwindNew, tracerMinNew, tracerMaxNew)
        #endif
        do iCell = 1, nCells
          invAreaCell1 = 1.0_RKIND / areaCell(iCell)

          ! Finish computing the low order horizontal fluxes
          ! Upwind fluxes are accumulated in workTend
          do i = 1, nEdgesOnCell(iCell)
            iEdge = edgesOnCell(i, iCell)
            cell1 = cellsOnEdge(1,iEdge)
            cell2 = cellsOnEdge(2,iEdge)
            signedFactor = edgeSignOnCell(i, iCell) * invAreaCell1

            do k = 1, maxLevelEdgeTop(iEdge)
              ! Here workTend is the advection tendency due to the
              ! upwind (low order) fluxes.
              workTend(k,iCell) = workTend(k,iCell) &
                                + signedFactor*lowOrderFlx(k,iEdge)

              ! Accumulate remaining high order fluxes
              flxOut(k,iCell) = flxOut(k,iCell) + min(0.0_RKIND,  &
                                signedFactor*highOrderFlx(k,iEdge))
              flxIn (k,iCell) = flxIn (k,iCell) + max(0.0_RKIND,  &
                                signedFactor*highOrderFlx(k,iEdge))

            end do
          end do

          ! Build the factors for the FCT
          ! Computed using the bounds that were computed previously,
          ! and the bounds on the newly updated value
          ! Factors are placed in the flxIn and flxOut arrays
          do k = 1, maxLevelCell(iCell)
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
          end do ! k loop
        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('scale factor build')
        call mpas_timer_start('rescale horiz fluxes')
        #endif

        ! Need all of the edges around owned cells
        nEdges = nEdgesHalo( 1 )

        !  rescale the high order horizontal fluxes

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(maxLevelEdgeTop, cellsOnEdge, &
        !$acc            highOrderFlx, flxIn, flxOut) &
        !$acc    private(k, cell1, cell2) async(1)
        #else
        !$omp do schedule(runtime) private(k, cell1, cell2) 
        #endif
        do iEdge = 1, nEdges
          cell1 = cellsOnEdge(1,iEdge)
          cell2 = cellsOnEdge(2,iEdge)
          do k = 1, maxLevelEdgeTop(iEdge)
            highOrderFlx(k,iEdge) = max(0.0_RKIND,highOrderFlx(k,iEdge))* &
                                    min(flxOut(k,cell1), flxIn (k,cell2)) &
                                  + min(0.0_RKIND,highOrderFlx(k,iEdge))* &
                                    min(flxIn (k,cell1), flxOut(k,cell2))
          end do ! k loop
        end do ! iEdge loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('rescale horiz fluxes')
        call mpas_timer_start('flux accumulate')
        #endif

        ! Accumulate the scaled high order vertical tendencies 
        ! and the upwind tendencies

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(maxLevelEdgeTop, maxLevelCell, areaCell, &
        !$acc            nEdgesOnCell, edgesOnCell, edgeSignOnCell, &
        !$acc            workTend, tmpTend, highOrderFlx, hProvInv, &
        !$acc            tracerCur, layerThickness) &
        !$acc    private(i, k, iEdge, invAreaCell1, signedFactor) async(1) 
        #else
        !$omp do schedule(runtime) &
        !$omp    private(i, k, iEdge, invAreaCell1, signedFactor)  
        #endif
        do iCell = 1, nCellsOwned
          invAreaCell1 = 1.0_RKIND / areaCell(iCell)

          ! Accumulate the scaled high order horizontal tendencies
          do i = 1, nEdgesOnCell(iCell)
            iEdge = edgesOnCell(i, iCell)
            signedFactor = invAreaCell1 * edgeSignOnCell(i, iCell)
            do k = 1, maxLevelEdgeTop(iEdge)
              ! workTend on the RHS is the upwind tendency
              ! workTend on the LHS is the total horizontal advection 
              !   tendency
              workTend(k,iCell) = workTend(k,iCell) &
                                + signedFactor * highOrderFlx(k, iEdge)
            end do
          end do

          do k = 1, maxLevelCell(iCell)
            ! workTend on RHS is total horizontal advection tendency
            ! tracerCur on LHS is provisional tracer after horz fluxes only.
            tracerCur(k,iCell) = (tracerCur(k,iCell)*layerThickness(k,iCell) &
                                  + dt*workTend(k,iCell))*hProvInv(k,iCell)
            tmpTend(k,iCell) = tmpTend(k,iCell) + workTend(k,iCell)
          end do

        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('flux accumulate')
        call mpas_timer_start('advect diags')
        #endif

        ! Compute budget and monotonicity diagnostics if needed
        if (computeBudgets) then

           !$acc enter data &
           !$acc    create(activeTracerHorzAdvectEdgeFlx, &
           !$acc           activeTracerHorzAdvectTend) async(1)
           #ifdef MPAS_OPENACC
           !$acc parallel loop &
           !$acc    present(nEdgesHalo, maxLevelEdgeTop, dvEdge, &
           !$acc            activeTracerHorzAdvectEdgeFlx, &
           !$acc            lowOrderFlx, highOrderFlx) &
           !$acc    private(k) async(1)
           #else
           !$omp do schedule(runtime) private(k) 
           #endif
           do iEdge = 1,nEdgesHalo( 1 )
           do k = 1,maxLevelEdgeTop(iEdge)
               ! Save u*h*T flux on edge for analysis. This variable 
               ! will be divided by h at the end of the time step.
               activeTracerHorzAdvectEdgeFlx(iTracer,k,iEdge) = &
                  (lowOrderFlx(k,iEdge) + highOrderFlx(k,iEdge))/ &
                  dvEdge(iEdge)
           enddo
           enddo
           #ifdef MPAS_OPENACC
           !$acc update host(activeTracerHorzAdvectEdgeFlx) async(1)
           #else
           !$omp end do
           #endif

           #ifdef MPAS_OPENACC
           !$acc parallel loop &
           !$acc    present(activeTracerHorzAdvectTend, &
           !$acc            maxLevelCell, workTend) &
           !$acc    private(k) async(1)
           #else
           !$omp do schedule(runtime) private(k) 
           #endif
           do iCell = 1, nCellsOwned
           do k = 1, maxLevelCell(iCell)
              activeTracerHorzAdvectTend(iTracer,k,iCell) = &
                                                     workTend(k,iCell)
           end do
           end do ! iCell loop
           #ifdef MPAS_OPENACC
           !$acc update host(activeTracerHorzAdvectTend) async(1)
           #else
           !$omp end do
           #endif

           !$acc exit data &
           !$acc    delete(activeTracerHorzAdvectEdgeFlx, &
           !$acc            activeTracerHorzAdvectTend)

        end if ! computeBudgets

        ! If requested, check tracer values against local min,max to
        ! detect non-monotone values and write warning if found
        if (monotonicityCheck) then
           ! because there is output, do this on host
           !$acc update host (tracerCur, tracerMin, tracerMax)

           !$omp do schedule(runtime) private(k)
           do iCell = 1, nCellsOwned
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
        end if ! monotonicity check

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('advect diags')
        #endif

        !-----------------------------------------------------------------------
        !  Horizontal advection complete
        !  Begin vertical advection
        !-----------------------------------------------------------------------

        #ifdef _ADV_TIMERS
        call mpas_timer_start('cell init')
        #endif

        ! Reset variables for accumulating vertical fluxes, tend
        #ifdef MPAS_OPENACC
        !$acc parallel loop gang vector collapse(2) &
        !$acc    present(highOrderFlx, workTend) private(k) async(1)
        #else
        !$omp do schedule(runtime) private(k)
        #endif
        do iCell = 1, nCellsAll
           do k=1, nVertLevels+1
              highOrderFlx(k,iCell) = 0.0_RKIND
              if ( k <= nVertLevels ) then
                 workTend    (k,iCell) = 0.0_RKIND
              endif
           end do ! k loop
        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('cell init')
        call mpas_timer_start('vertical flux')
        #endif

        ! Need all owned and 1 halo cells
        nCells = nCellsHalo( 1 )

        ! Determine bounds on tracerCur from neighbor values for limiting

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(tracerCur, tracerMin, tracerMax) &
        !$acc    private(k, kmax) async(1)
        #else
        !$omp do schedule(runtime) private(k, kmax) 
        #endif
        do iCell = 1, nCells
          kmax = maxLevelCell(iCell)

          ! take care of top cell
          tracerMax(1,iCell) = max(tracerCur(1,iCell), &
                                   tracerCur(2,iCell))
          tracerMin(1,iCell) = min(tracerCur(1,iCell), &
                                   tracerCur(2,iCell))
          do k=2,kmax-1
            tracerMax(k,iCell) = max(tracerCur(k-1,iCell), &
                                     tracerCur(k  ,iCell), &
                                     tracerCur(k+1,iCell))
            tracerMin(k,iCell) = min(tracerCur(k-1,iCell), &
                                     tracerCur(k  ,iCell), &
                                     tracerCur(k+1,iCell))
          end do
          ! finish with bottom cell
          tracerMax(kmax,iCell) = max(tracerCur(kmax  ,iCell), &
                                      tracerCur(kmax-1,iCell))
          tracerMin(kmax,iCell) = min(tracerCur(kmax  ,iCell), &
                                      tracerCur(kmax-1,iCell))
        end do ! cell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        ! Compute the high order vertical fluxes based on order selected.
        ! Special cases for top and bottom layers handled in following loop.

        select case (vertOrder)
        case (vertOrder4)

          #ifdef MPAS_OPENACC
          !$acc parallel loop &
          !$acc    present(maxLevelCell, highOrderFlx, tracerCur, w) &
          !$acc    private(k, kmax) async(1)
          #else
          !$omp do schedule(runtime) private(k, kmax) 
          #endif
          do iCell = 1, nCells
            kmax = maxLevelCell(iCell)
            do k=3,kmax-1
              highOrderFlx(k, iCell) = w(k,iCell)*( &
                  7.0_RKIND*(tracerCur(k  ,iCell) + tracerCur(k-1,iCell)) - &
                            (tracerCur(k+1,iCell) + tracerCur(k-2,iCell)))/ &
                  12.0_RKIND

            end do
          end do ! cell loop
          #ifndef MPAS_OPENACC
          !$omp end do
          #endif

        case (vertOrder3)

          #ifdef MPAS_OPENACC
          !$acc parallel loop &
          !$acc    present(maxLevelCell, highOrderFlx, tracerCur, w) &
          !$acc    private(k, kmax) async(1)
          #else
          !$omp do schedule(runtime) private(k, kmax) 
          #endif
          do iCell = 1, nCells
            kmax = maxLevelCell(iCell)
            do k=3,kmax-1
              highOrderFlx(k, iCell) = (w(k,iCell)* &
                   (7.0_RKIND*(tracerCur(k  ,iCell) + tracerCur(k-1,iCell)) - &
                              (tracerCur(k+1,iCell) + tracerCur(k-2,iCell))) - &
                        coef3rdOrder*abs(w(k,iCell))* &
                             ((tracerCur(k+1,iCell) - tracerCur(k-2,iCell)) - &
                    3.0_RKIND*(tracerCur(k  ,iCell) - tracerCur(k-1,iCell))))/ &
                   12.0_RKIND


            end do
          end do ! cell loop
          #ifndef MPAS_OPENACC
          !$omp end do
          #endif

       case (vertOrder2)

          #ifdef MPAS_OPENACC
          !$acc parallel loop &
          !$acc    present(maxLevelCell, highOrderFlx, hProv, &
          !$acc            tracerCur, w) &
          !$acc    private(k, kmax, verticalWeightK, verticalWeightKm1) async(1)
          #else
          !$omp do schedule(runtime) &
          !$acc    private(k, kmax, verticalWeightK, verticalWeightKm1) 
          #endif
          do iCell = 1, nCells
            kmax = maxLevelCell(iCell)
            do k=3,kmax-1
              verticalWeightK   = hProv(k-1,iCell) / &
                                 (hProv(k  ,iCell) + hProv(k-1,iCell))
              verticalWeightKm1 = hProv(k  ,iCell) / &
                                 (hProv(k  ,iCell) + hProv(k-1,iCell))
              highOrderFlx(k,iCell) = w(k,iCell)* &
                                 (verticalWeightK  *tracerCur(k  ,iCell) + &
                                  verticalWeightKm1*tracerCur(k-1,iCell))
            end do
          end do ! cell loop
          #ifndef MPAS_OPENACC
          !$omp end do
          #endif

        end select ! vertical order

        ! Treat special cases for high order flux
        ! Compute low order upwind vertical flux (monotonic and diffused)
        ! Remove low order flux from the high order flux.
        ! Store left over high order flux in highOrderFlx array.

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(maxLevelCell, hProv, tracerCur, w, &
        !$acc            highOrderFlx, lowOrderFlx, workTend, &
        !$acc            flxIn, flxOut) &
        !$acc    private(k, kmax, verticalWeightK, verticalWeightKm1) async(1)
        #else
        !$omp do schedule(runtime) &
        !$omp    private(k, kmax, verticalWeightK, verticalWeightKm1)
        #endif
        do iCell = 1, nCells
          kmax = maxLevelCell(iCell)
          ! Next-to-top cell in column is second-order
          highOrderFlx(1,iCell) = 0.0_RKIND
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

          ! Compute low order (upwind) flux and remove from high order
          lowOrderFlx(1,iCell) = 0.0_RKIND
          do k = 2, kmax
            lowOrderFlx(k,iCell) = &
                min(0.0_RKIND,w(k,iCell))*tracerCur(k-1,iCell) + &
                max(0.0_RKIND,w(k,iCell))*tracerCur(k  ,iCell)
            highOrderFlx(k,iCell) = highOrderFlx(k,iCell) - &
                                     lowOrderFlx(k,iCell)
          end do ! k loop
          lowOrderFlx(kmax+1,iCell) = 0.0_RKIND

          ! Upwind fluxes are accumulated in workTend
          ! flxIn contains the total remaining high order flux into iCell
          !          it is positive.
          ! flxOut contains the total remaining high order flux out of iCell
          !           it is negative
          do k = 1, kmax
            workTend(k,iCell) = lowOrderFlx(k+1,iCell) &
                              - lowOrderFlx(k  ,iCell)
            flxIn (k, iCell) = max(0.0_RKIND, highOrderFlx(k+1, iCell)) &
                             - min(0.0_RKIND, highOrderFlx(k  , iCell))
            flxOut(k, iCell) = min(0.0_RKIND, highOrderFlx(k+1, iCell)) &
                             - max(0.0_RKIND, highOrderFlx(k  , iCell))
          end do ! k Loop
        end do ! iCell Loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('vertical flux')
        call mpas_timer_start('scale factor build')
        #endif

        ! Need one halo of cells around owned cells
        nCells = nCellsHalo( 1 )

        ! Build the scale factors to limit flux for FCT
        ! Computed using the bounds that were computed previously,
        ! and the bounds on the newly updated value
        ! Factors are placed in the flxIn and flxOut arrays

        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(maxLevelCell, hProv, hNewInv, workTend, &
        !$acc            tracerCur, tracerMax, tracerMin, &
        !$acc            flxIn, flxOut) &
        !$acc    private(k, scaleFactor, tracerUpwindNew, &
        !$acc            tracerMinNew, tracerMaxNew) async(1)
        #else
        !$omp do schedule(runtime) &
        !$omp    private(k, scaleFactor, tracerUpwindNew, &
        !$omp            tracerMinNew, tracerMaxNew)
        #endif
        do iCell = 1, nCells
          do k = 1, maxLevelCell(iCell)
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
          end do ! k loop
        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

#ifdef _ADV_TIMERS
        call mpas_timer_stop('scale factor build')
        call mpas_timer_start('flux accumulate')
#endif

        ! Accumulate the scaled high order vertical tendencies
        ! and the upwind tendencies
        #ifdef MPAS_OPENACC
        !$acc parallel loop &
        !$acc    present(maxLevelCell, highOrderFlx, flxIn, flxOut, &
        !$acc            workTend, tmpTend) &
        !$acc    private(k, kmax, flux) async(1)
        #else
        !$omp do schedule(runtime) private(k, kmax, flux)
        #endif
        do iCell = 1, nCellsOwned
          kmax = maxLevelCell(iCell)
          ! rescale the high order vertical flux
          do k = 2, kmax
            flux =  highOrderFlx(k,iCell)
            highOrderFlx(k,iCell) = max(0.0_RKIND,flux)*   &
                                    min(flxOut(k  ,iCell), &
                                        flxIn (k-1,iCell)) &
                                  + min(0.0_RKIND,flux)*   &
                                    min(flxOut(k-1,iCell), &
                                        flxIn (k  ,iCell))
          end do ! k loop

          do k = 1,kmax
            ! workTend on the RHS is the upwind tendency
            ! workTend on the LHS is the total vertical advection tendency
            workTend(k, iCell) = workTend(k, iCell)       &
                               + (highOrderFlx(k+1,iCell) &
                                - highOrderFlx(k  ,iCell))
            tmpTend(k,iCell) = tmpTend(k,iCell) + workTend(k,iCell)
          end do ! k loop

        end do ! iCell loop
        #ifndef MPAS_OPENACC
        !$omp end do
        #endif

        ! Compute advection diagnostics and monotonicity checks if requested

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('flux accumulate')
        call mpas_timer_start('advect diags')
        #endif

        if (computeBudgets) then
           !$acc enter data &
           !$acc    create(activeTracerVertAdvectTopFlx, &
           !$acc           activeTracerVertAdvectTend) async(1)

           #ifdef MPAS_OPENACC
           !$acc parallel loop &
           !$acc    present(maxLevelCell, highOrderFlx, lowOrderFlx, &
           !$acc            activeTracerVertAdvectTopFlx, &
           !$acc            activeTracerVertAdvectTend, workTend) &
           !$acc    private(k) async(1)
           #else
           !$omp do schedule(runtime) private(k)
           #endif
           do iCell = 1, nCellsOwned
              do k = 2, maxLevelCell(iCell)
                 activeTracerVertAdvectTopFlx(iTracer,k,iCell) = &
                    lowOrderFlx(k,iCell) + highOrderFlx(k,iCell)
              end do
              do k = 1, maxLevelCell(iCell)
                 activeTracerVertAdvectTend(iTracer,k,iCell) = &
                    workTend(k,iCell)
              end do
           end do ! iCell loop
           #ifdef MPAS_OPENACC
           !$acc update host(activeTracerVertAdvectTopFlx, &
           !$acc            activeTracerVertAdvectTend) async(1)
           #else
           !$omp end do
           #endif

           !$acc exit data &
           !$acc    delete(activeTracerVertAdvectTopFlx, &
           !$acc            activeTracerVertAdvectTend)

        end if ! computeBudgets

        if (monotonicityCheck) then
          ! Check for monotonicity of new tracer value
          ! Because there is output, do this on host
          !$acc update host (tracerCur, tracerMin, tracerMax)

          !$omp do schedule(runtime), private(k, tracerNew)
          do iCell = 1, nCellsOwned
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

        end if ! monotonicity check

        #ifdef _ADV_TIMERS
        call mpas_timer_stop('advect diags')
        #endif

        ! Add final advection tendency to the total tendency
        ! retrieve result from device and accumulate on host
        !$acc wait
        !$acc update host (tmpTend)
        
        !$omp do schedule(runtime), private(k)
        do iCell=1,nCellsOwned
        do k=1,maxLevelCell(iCell)
           tend(iTracer,k,iCell) = tend(iTracer,k,iCell) &
                                 + tmpTend(k,iCell)
        end do
        end do
        !$omp end do

      end do ! iTracer loop

      #ifdef _ADV_TIMERS
      call mpas_timer_start('deallocates')
      #endif
      !$acc exit data &
      !$acc   delete(tracers, layerThickness, normalThicknessFlux, w, &
      !$acc          wgtTmp, flxTmp, sgnTmp, tracerCur, tracerMin, &
      !$acc          tracerMax, hNewInv, hProv, hProvInv, flxIn, &
      !$acc          flxOut, workTend, lowOrderFlx, highOrderFlx, &
      !$acc          tmpTend)
      #ifndef MPAS_OPENACC
      !$omp end parallel
      #endif
      deallocate(wgtTmp,       &
                 flxTmp,       &
                 sgnTmp,       &
                 tracerCur,    &
                 tracerMin,    &
                 tracerMax,    &
                 hNewInv,      &
                 hProv,        &
                 hProvInv,     &
                 flxIn,        &
                 flxOut,       &
                 workTend,     &
                 tmpTend,      &
                 lowOrderFlx,  &
                 highOrderFlx)

      #ifdef _ADV_TIMERS
      call mpas_timer_stop('deallocates')
      #endif
   !----------------------------------------------------------------------------

   end subroutine ocn_tracer_advection_mono_tend!}}}

!*******************************************************************************
!
!  routine ocn_tracer_advection_mono_init
!
!> \brief MPAS initialize monotonic tracer advection tendency with FCT
!> \author Mark Petersen, David Lee, Doug Jacobsen, Phil Jones
!> \date   October 2017, updated May 2019, 2020
!> \details
!>  This routine initializes monotonic tracer advection quantities for
!>  the flux-corrected transport (FCT) algorithm.
!
!-------------------------------------------------------------------------------

   subroutine ocn_tracer_advection_mono_init(err)

      !-------------------------------------------------------------------------
      ! Output parameters
      !-------------------------------------------------------------------------

      integer, intent(out) :: &
         err                   !< [out] error flag

      !{{{ 
      ! end of preamble
      !----------------
      ! begin code

      err = 0 ! initialize error code to success

      ! Define local advection options from user configuration choices
      ! in ocn_config

      ! Check that the halo is wide enough for FCT
      if (config_num_halos < 3) then
         call mpas_log_write( &
            'Monotonic advection cannot be used with less than 3 halos.', &
            MPAS_LOG_CRIT)
         err = -1
      end if

      ! Set blending coefficient if 3rd order horizontal advection chosen
      select case (config_horiz_tracer_adv_order)
      case (2)
         coef3rdOrder = 0.0_RKIND
      case (3)
         coef3rdOrder = config_coef_3rd_order
      case (4)
         coef3rdOrder = 0.0_RKIND
      case default
         coef3rdOrder = 0.0_RKIND
         call mpas_log_write( &
            'Invalid value for horizontal advection order, defaulting to 2',&
            MPAS_LOG_WARN)
      end select ! horizontal advection order

      ! Set vertical advection order
      select case (config_vert_tracer_adv_order)
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
      monotonicityCheck = config_check_tracer_monotonicity

   !----------------------------------------------------------------------------

   end subroutine ocn_tracer_advection_mono_init!}}}

!*******************************************************************************

end module ocn_tracer_advection_mono

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
