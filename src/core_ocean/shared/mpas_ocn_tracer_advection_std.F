! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_tracer_advection_std
!
!> \brief MPAS standard tracer advection
!> \author Doug Jacobsen, Phil Jones
!> \date   03/09/12, updated May 2019
!> \details
!>  This module contains routines for advection of tracers using a standard
!>  FV algorithm in MPAS discretization.
!
!-------------------------------------------------------------------------------

module ocn_tracer_advection_std

   ! module includes
   use mpas_kind_types

   use ocn_config
   use ocn_mesh

   implicit none
   private
   save

   !----------------------------------------------------------------------------
   ! public method interfaces
   !----------------------------------------------------------------------------

   public :: ocn_tracer_advection_std_tend, &
             ocn_tracer_advection_std_init

   !----------------------------------------------------------------------------
   ! private module variables
   !----------------------------------------------------------------------------

   real (kind=RKIND) :: &
      coef3rdOrder       !< coefficient for blending high-order terms

   integer :: vertOrder  !< choice of order for vertical advection

   integer, parameter :: &! enumerator for supported vertical adv order
      vertOrder2=2,      &!< 2nd order
      vertOrder3=3,      &!< 3rd order
      vertOrder4=4        !< 4th order

!*******************************************************************************

   contains

!*******************************************************************************
!
!  routine ocn_tracer_advection_std_tend
!
!> \brief MPAS standard tracer advection tendency
!> \author Doug Jacobsen, Phil Jones
!> \date   03/09/12, updated May 2020
!> \details
!>  This routine computes the standard tracer advection tendency in
!>  both horizontal and vertical. It is the option used if monotone is
!>  not selected.
!
!-------------------------------------------------------------------------------

   subroutine ocn_tracer_advection_std_tend(tracers, normalThicknessFlux, &
                                            w, layerThickness, dt, tend)

      !-------------------------------------------------------------------------
      ! Input variables
      !-------------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         tracers               !< [in] current tracer values

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalThicknessFlux, &!< [in] Thichness weighted velocitiy
         w,                   &!< [in] Vertical velocity
         layerThickness        !< [in] Thickness

      real (kind=RKIND), intent(in) :: &
         dt                    !< [in] Timestep

      !-------------------------------------------------------------------------
      !  Input/output variables
      !-------------------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
         tend                  !< [inout] Tracer tendency

      !{{{
      !-------------------------------------------------------------------------
      ! Local variables
      !-------------------------------------------------------------------------

      integer :: &
         i, k, iTracer,   &! loop iterators for edgesOnCell, layers, tracers 
         iCell, iEdge,    &! loop iterators for cells, edges
         cell1, cell2,    &! indices of neighbor cells across edge
         kMax,            &! last active layer
         nTracers          ! number of tracers

      real (kind=RKIND) :: &
         tracerWgt,        &! weighting of tracers based on order (also a tmp)
         dvEdgeTmp,        &! local scalar for common factors
         advCoef1,         &! local temp to hold advection coefficients
         advCoef3,         &! local temp to hold advection coefficients
         invAreaCell1,     &! inverse of cell area
         vertWgtK,         &! vertical weights based on order of vert adv
         vertWgtKm1

      real (kind=RKIND), dimension(:,:), allocatable :: &
         tracerCur,        &! reordered array with current tracer
         highOrderHorzFlx, &! high-order flux in the horizontal
         highOrderVertFlx   ! high-order flux in the vertical

      ! End preamble
      !-------------
      ! Begin code

      ! Get number of tracers based on input array
      nTracers = size(tracers,dim=1)

      ! Allocate and transfer local temp arrays

      allocate(tracerCur       (nVertLevels,nCellsAll), &
               highOrderVertFlx(nVertLevels,nCellsAll), &
               highOrderHorzFlx(nVertLevels,nEdgesAll))
      !$acc enter data &
      !$acc    create(tracerCur, highOrderVertFlx, highOrderHorzFlx)

#ifndef MPAS_OPENACC
      !$omp parallel
#endif

      ! Loop over tracers. One tracer is advected at a time. 
      ! Index order is swapped for performance.

      do iTracer = 1, nTracers

         ! Initialize variables for use in this iTracer iteration

#ifdef MPAS_OPENACC
         !$acc parallel loop private(k) &
         !$acc   present(tracerCur, tracers, highOrderVertFlx)
#else
         !$omp do schedule(runtime) private(k)
#endif
         do iCell = 1, nCellsAll
         do k = 1, nVertLevels
            tracerCur(k, iCell) = tracers(iTracer, k, iCell)
            highOrderVertFlx(k, iCell) = 0.0_RKIND
         end do
         end do
#ifndef MPAS_OPENACC
        !$omp end do
#endif

#ifdef MPAS_OPENACC
         !$acc parallel loop private(k) &
         !$acc   present(highOrderHorzFlx)
#else
         !$omp do schedule(runtime) private(k)
#endif
         do iEdge = 1, nEdgesAll
         do k = 1, nVertLevels
            highOrderHorzFlx(k, iEdge) = 0.0_RKIND
         end do
         end do
#ifndef MPAS_OPENACC
         !$omp end do
#endif

         ! Compute the high order vertical flux based on 
         ! chosen order. Special cases near top and bottom 
         ! treated in following loop 

         select case (vertOrder)
         case (vertOrder4)

#ifdef MPAS_OPENACC
            !$acc parallel loop private(k) &
            !$acc    present(maxLevelCell, tracerCur, &
            !$acc            highOrderVertFlx, w)
#else
            !$omp do schedule(runtime) private(k)
#endif
            do iCell = 1, nCellsAll
            do k=3,maxLevelCell(iCell)-1
               highOrderVertFlx(k,iCell) = w(k,iCell)*( &
                  7.0_RKIND*(tracerCur(k  ,iCell) + tracerCur(k-1,iCell)) - &
                            (tracerCur(k+1,iCell) + tracerCur(k-2,iCell)))/ &
                  12.0_RKIND
            end do
            end do
#ifndef MPAS_OPENACC
            !$omp end do
#endif

         case (vertOrder3)

#ifdef MPAS_OPENACC
            !$acc parallel loop private(k) &
            !$acc    present(maxLevelCell, tracerCur, &
            !$acc            highOrderVertFlx, w)
#else
            !$omp do schedule(runtime) private(k)
#endif
            do iCell = 1, nCellsAll
            do k=3,maxLevelCell(iCell)-1
               highOrderVertFlx(k,iCell) = (w(k,iCell)* &
                   (7.0_RKIND*(tracerCur(k  ,iCell) + tracerCur(k-1,iCell)) - &
                              (tracerCur(k+1,iCell) + tracerCur(k-2,iCell))) - &
                        coef3rdOrder*abs(w(k,iCell))* &
                             ((tracerCur(k+1,iCell) - tracerCur(k-2,iCell)) - &
                    3.0_RKIND*(tracerCur(k  ,iCell) - tracerCur(k-1,iCell))))/ &
                   12.0_RKIND
            end do
            end do
#ifndef MPAS_OPENACC
            !$omp end do
#endif

         case (vertOrder2)

#ifdef MPAS_OPENACC
            !$acc parallel loop &
            !$acc    present(maxLevelCell, layerThickness, &
            !$acc            highOrderVertFlx, tracerCur, w) &
            !$acc    private(k, vertWgtK, vertWgtKm1)
#else
            !$omp do schedule(runtime) & 
            !$omp    private(k, vertWgtK, vertWgtKm1)
#endif
            do iCell = 1, nCellsAll
            do k=3,maxLevelCell(iCell)-1
               vertWgtK   = layerThickness(k-1,iCell)/  &
                           (layerThickness(k  ,iCell) + &
                            layerThickness(k-1,iCell))
               vertWgtKm1 = layerThickness(k  ,iCell)/  &
                           (layerThickness(k  ,iCell) + &
                            layerThickness(k-1,iCell))
               highOrderVertFlx(k,iCell) = w(k, iCell)* &
                      (vertWgtK  *tracerCur(k  ,iCell) + &
                       vertWgtKm1*tracerCur(k-1,iCell))
            end do
            end do
#ifndef MPAS_OPENACC
            !$omp end do
#endif
         end select ! vertical order
        
         ! Treat special cases for high order flux near top, bottom
         ! Mainly reverting to lower order where not enough points
         ! are available for high-order methods.

#ifdef MPAS_OPENACC
         !$acc parallel loop &
         !$acc    present(maxLevelCell, layerThickness, &
         !$acc            highOrderVertFlx, tracerCur, w) &
         !$acc    private(k, kMax, vertWgtK, vertWgtKm1)
#else
         !$omp do schedule(runtime) & 
         !$omp    private(k, kMax, vertWgtK, vertWgtKm1)
#endif
         do iCell = 1, nCellsAll
            kMax = maxLevelCell(iCell)
            highOrderVertFlx(1,iCell) = 0.0_RKIND
            ! Next-to-top layer is 2nd order
            if (kMax > 1) then
               vertWgtK   = layerThickness(1,iCell)/  &
                           (layerThickness(2,iCell) + &
                            layerThickness(1,iCell))
               vertWgtKm1 = layerThickness(2,iCell)/  &
                           (layerThickness(2,iCell) +  &
                            layerThickness(1,iCell))

               highOrderVertFlx(2,iCell) = w(2,iCell)* &
                           (vertWgtK  *tracerCur(2,iCell) + &
                            vertWgtKm1*tracerCur(1,iCell))
            endif

            ! Deepest layer is 2nd order
            k = max(2,kMax)
            vertWgtK   = layerThickness(k-1,iCell)/  &
                        (layerThickness(k  ,iCell) + &
                         layerThickness(k-1,iCell))
            vertWgtKm1 = layerThickness(k  ,iCell)/  &
                        (layerThickness(k  ,iCell) + &
                         layerThickness(k-1,iCell))

            highOrderVertFlx(k,iCell) = w(k,iCell)* &
                      (vertWgtK  *tracerCur(k  ,iCell) + &
                       vertWgtKm1*tracerCur(k-1,iCell))
         end do ! iCell Loop
#ifndef MPAS_OPENACC
         !$omp end do
#endif

         ! Finish vertical advection by adding to tendency

#ifdef MPAS_OPENACC
         !$acc parallel loop &
         !$acc    present(maxLevelCell, highOrderVertFlx, tend) &
         !$acc    private(k)
#else
         !$omp do schedule(runtime) & 
         !$omp    private(k)
#endif
         do iCell = 1, nCellsOwned
         do k = 1,maxLevelCell(iCell)
            tend(iTracer, k, iCell) = tend(iTracer, k, iCell) +    &
                                     (highOrderVertFlx(k+1, iCell) &
                                    - highOrderVertFlx(k  ,iCell))
         end do ! k loop
         end do ! iCell loop
#ifndef MPAS_OPENACC
         !$omp end do
#endif

         ! Horizontal advection

         ! Compute 2nd order fluxes where needed.

#ifdef MPAS_OPENACC
         !$acc parallel loop &
         !$acc    present(cellsOnEdge, dvEdge, maxLevelEdgeTop,  &
         !$acc            highOrderHorzFlx, normalThicknessFlux, &
         !$acc            highOrderAdvectionMask, tracerCur)     &
         !$acc    private(k, cell1, cell2, dvEdgeTmp, tracerWgt)
#else
         !$omp do schedule(runtime) & 
         !$omp    private(k, cell1, cell2, dvEdgeTmp, tracerWgt)
#endif
         do iEdge = 1, nEdgesAll
            cell1 = cellsOnEdge(1, iEdge)
            cell2 = cellsOnEdge(2, iEdge)
            dvEdgeTmp = dvEdge(iEdge)*0.5_RKIND

            do k = 1, maxLevelEdgeTop(iEdge)
               tracerWgt = (1.0_RKIND-highOrderAdvectionMask(k,iEdge))* &
                               dvEdgeTmp*normalThicknessFlux(k,iEdge)

               highOrderHorzFlx(k,iEdge) = highOrderHorzFlx(k,iedge) + &
                    tracerWgt*(tracerCur(k,cell1) + tracerCur(k,cell2))
            end do ! k loop
         end do ! edge loop 2nd order
#ifndef MPAS_OPENACC
         !$omp end do
#endif

         ! Compute 3rd or 4th fluxes where requested.

#ifdef MPAS_OPENACC
         !$acc parallel loop &
         !$acc    present(advCoefs, advCoefs3rd, maxLevelCell, &
         !$acc            nAdvCellsForEdge, advCellsForEdge,   &
         !$acc            highOrderAdvectionMask, tracerCur,   &
         !$acc            normalThicknessFlux)                 &
         !$acc    private(i, k, iCell, advCoef1, advCoef3, tracerWgt)
#else
         !$omp do schedule(runtime) & 
         !$omp    private(i, k, iCell, advCoef1, advCoef3, tracerWgt)
#endif
         do iEdge = 1, nEdgesAll
            do i = 1, nAdvCellsForEdge(iEdge)
               iCell = advCellsForEdge(i,iEdge)
               advCoef1 = advCoefs(i,iEdge)
               advCoef3 = advCoefs3rd(i,iEdge)
               do k = 1, maxLevelCell(iCell)
                  tracerWgt = highOrderAdvectionMask(k,iEdge)* &
                      (advCoef1 + coef3rdOrder* &
                       sign(1.0_RKIND,normalThicknessFlux(k,iEdge))* &
                       advCoef3)

                  tracerWgt = normalThicknessFlux(k,iEdge)*tracerWgt

                  highOrderHorzFlx(k,iEdge) = &
                  highOrderHorzFlx(k,iEdge) + &
                                      tracerWgt*tracerCur(k,iCell)
               end do ! k loop
            end do ! i loop over nAdvCellsForEdge
         end do ! iEdge loop
#ifndef MPAS_OPENACC
         !$omp end do
#endif

         ! Add horizontal advection tendencies

#ifdef MPAS_OPENACC
         !$acc parallel loop &
         !$acc    present(nEdgesOnCell, EdgesOnCell, areaCell, &
         !$acc            maxLevelEdgeTop, edgeSignOnCell,     &
         !$acc            highOrderHorzFlx, tend)              &
         !$acc    private(i, k, iEdge, invAreaCell1)
#else
         !$omp do schedule(runtime) & 
         !$omp    private(i, k, iEdge, invAreaCell1)
#endif
         do iCell = 1, nCellsOwned
            invAreaCell1 = 1.0_RKIND/areaCell(iCell)
            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)
               do k = 1, maxLevelEdgeTop(iEdge)
                  tend(iTracer,k,iCell) = tend(iTracer,k,iCell) + &
                                         edgeSignOnCell(i,iCell)* &
                                        highOrderHorzFlx(k, iEdge) &
                                      * invAreaCell1
             end do
           end do
         end do
#ifndef MPAS_OPENACC
         !$omp end do
#endif

         ! Accumulate the high order vertical tendencies.

      end do ! iTracer loop

      ! Clean up

#ifndef MPAS_OPENACC
      !$omp end parallel
#endif
      !$acc exit data &
      !$acc    delete(tracerCur, highOrderVertFlx, highOrderHorzFlx)
      deallocate(tracerCur, highOrderVertFlx, highOrderHorzFlx)

   !----------------------------------------------------------------------------

   end subroutine ocn_tracer_advection_std_tend!}}}

!*******************************************************************************
!
!  routine ocn_tracer_advection_std_init
!
!> \brief MPAS initialize standard tracer advection tendency.
!> \author Doug Jacobsen, Phil Jones
!> \date   03/09/12, updated May 2019
!> \details
!>  This routine initializes constants and choices for the standard tracer
!>  advection tendency.
!
!-------------------------------------------------------------------------------

   subroutine ocn_tracer_advection_std_init(err)

      !-------------------------------------------------------------------------
      ! Output variables
      !-------------------------------------------------------------------------

      integer, intent(out) :: err !< [out] Error Flag

      !{{{
      ! end of preamble
      !----------------
      ! begin code

      err = 0 ! set error code to success

      ! set module options based on user input in ocn_config

      ! set 3rd order coefficient based on horizontal order choice
      select case (config_horiz_tracer_adv_order)
      case (2)
         coef3rdOrder = 0.0_RKIND
      case (3)
         coef3rdOrder = config_coef_3rd_order
      case (4)
         coef3rdOrder = 0.0_RKIND
      case default
         call mpas_log_write( &
            'Invalid value for horz advection order, defaulting to 2nd order', &
            MPAS_LOG_WARN)
      end select ! horzAdvOrder

      ! set choice of vertical advection order
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
         'Invalid value for vertical advection order, defaulting to 2nd order',&
         MPAS_LOG_WARN)
      end select ! vertAdvOrder

   end subroutine ocn_tracer_advection_std_init!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

end module ocn_tracer_advection_std

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
