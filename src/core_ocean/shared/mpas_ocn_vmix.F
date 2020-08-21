! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vmix
!
!> \brief MPAS ocean vertical mixing driver
!> \author Mark Petersen, Phil Jones
!> \date   September 2011, updated June 2020
!> \details
!>  This module is the main driver for vertical mixing in the ocean.
!>
!
!-----------------------------------------------------------------------

module ocn_vmix

   use mpas_kind_types
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer

   use ocn_constants
   use ocn_config
   use ocn_mesh
   use ocn_vmix_cvmix
   use ocn_vmix_coefs_redi

   implicit none
   private
   save

   !--------------------------------------------------------------------
   ! Public parameters
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   ! Public member functions
   !--------------------------------------------------------------------

   public :: ocn_vmix_coefs, &
             ocn_vel_vmix_tend_implicit, &
             ocn_tracer_vmix_tend_implicit, &
             ocn_vmix_init, &
             ocn_vmix_implicit

   !--------------------------------------------------------------------
   ! Private module variables
   !--------------------------------------------------------------------

   logical ::       &
      velVmixOn,    &! on/off switch for velocity vertical mixing
      tracerVmixOn, &! on/off switch for tracer   vertical mixing
      rayleighOn,   &! on/off switch for Rayleigh drag
      kppNonlocalOn  ! on/off switch to treat KPP nonlocal terms
 
   real (kind=RKIND) :: &
      implicitBottomDragCoef,    &! bottom drag coefficient
      rayleighDampingCoef,       &! Rayleigh drag coefficient
      rayleighBottomDampingCoef, &! Rayleigh damping coefficient
      rayleighDepthVariable       ! Rayleigh depth parameter

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_vmix_coefs
!
!> \brief   Computes coefficients for vertical mixing
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine computes the vertical mixing coefficients for momentum
!>  and tracers based user choices of mixing parameterization.
!
!-----------------------------------------------------------------------

   subroutine ocn_vmix_coefs(statePool, forcingPool, diagnosticsPool, &
                             err, timeLevelIn)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      integer, intent(in), optional :: &
         timeLevelIn       !< [in] Optional time level for state pool

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(inout) :: &
         statePool           !< [inout] state information

      type (mpas_pool_type), intent(inout) :: &
         forcingPool         !< [inout] forcing information

      type (mpas_pool_type), intent(inout) :: &
         diagnosticsPool     !< [inout] diagnostic information

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: &
         iEdge, iCell,  &! loop iterators for cells, edges
         timeLevel,     &! time level to use for state variables
         err1, err2      ! individual error flags from child routines

      real (kind=RKIND), dimension(:,:), pointer :: &
         vertViscTopOfEdge, &! vertical viscosity (from diagnostics)
         vertDiffTopOfCell

      ! End preamble
      !-------------
      ! Begin code

      ! Set default time level (=1, current) or override with input

      err = 0
      timeLevel = 1
      if (present(timeLevelIn)) timeLevel = timeLevelIn

      ! retrieve viscosity, diffusion arrays from diag pool

      call mpas_pool_get_array(diagnosticsPool, 'vertViscTopOfEdge', &
                                                 vertViscTopOfEdge)
      call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', &
                                                 vertDiffTopOfCell)

      ! initialize viscosity and diffusivity to zero

      !$omp parallel
      !$omp do schedule(runtime)
      do iEdge = 1, nEdgesHalo(1)
         vertViscTopOfEdge(:, iEdge) = 0.0_RKIND
      end do
      !$omp end do

      !$omp do schedule(runtime)
      do iCell = 1, nCellsHalo(1)
         vertDiffTopOfCell(:, iCell) = 0.0_RKIND
      end do
      !$omp end do
      !$omp end parallel

      ! call relevant routines for computing coefficients
      call ocn_vmix_coefs_cvmix_build(statePool, forcingPool, &
                                      diagnosticsPool, err1, timeLevel)
      call ocn_vmix_coefs_redi_build(statePool, diagnosticsPool, &
                                     err2, timeLevel)

      err = ior(err1, err2)

   !--------------------------------------------------------------------

   end subroutine ocn_vmix_coefs!}}}

!***********************************************************************
!
!  routine ocn_vel_vmix_tend_implicit_rayleigh
!
!> \brief   Computes tendencies for implicit momentum vertical mixing
!> \author  Mark Petersen, Phillip J. Wolfram
!> \date    September 2011, July 2019
!> \details
!>  This routine computes the tendencies for implicit vertical mixing
!>  for momentum using computed coefficients and includes implicit
!>  rayleigh drag.
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_vmix_tend_implicit_rayleigh(dt, &
                               kineticEnergyCell, vertViscTopOfEdge, &
                               layerThickness, layerThicknessEdge, &
                               normalVelocity, err) !{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         kineticEnergyCell, &!< [in] kinetic energy at cell center
         vertViscTopOfEdge, &!< [in] vertical viscosity at top
         layerThickness      !< [in] thickness at cell center

      real (kind=RKIND), intent(in) :: &
         dt              !< [in] time step

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         normalVelocity,     &!< [inout] vertical velocity
         layerThicknessEdge   !< [inout] thickness at edge

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: &
         iEdge, k,     &! loop iterators for vertical, edge
         cell1, cell2, &! cell index for neighbor cells accross edge
         kmax           ! deepest layer index at top of edge

      real (kind=RKIND), dimension(:), allocatable :: &
         A, B, C,      &! tridiagonal matrix
         bTemp,rTemp    ! intermediate temporaries in tridiagonal solve

      real (kind=RKIND), dimension(:), allocatable :: &
         edgeThickTot,   &! total depth at edge for Rayleigh damping
         botDragTerm      ! bottom drag

      real (KIND=RKIND) :: &
         rayleighTerm,   &! contributions from Rayleigh drag
         mTmp             ! temporary tridiagonal factor

      ! End preamble
      !-------------
      ! Begin code

      ! Return if mixing not turned on

      err = 0
      if (.not. velVmixOn) return

      ! allocate various arrays

      allocate(A(nVertLevels), &
               B(nVertLevels), &
               C(nVertLevels), &
               bTemp(nVertLevels), &
               rTemp(nVertLevels), &
               edgeThickTot(nEdgesOwned), &
               botDragTerm (nEdgesOwned))

      ! Create space on device (not that subroutine arguments
      ! are assumed to have already been transferred to device)
      !$acc enter data &
      !$acc    copyin(kineticEnergyCell, vertViscTopOfEdge, &
      !$acc           layerThickness, layerThicknessEdge, &
      !$acc           normalVelocity) &
      !$acc    create(A, B, C, bTemp, rTemp, edgeThickTot, botDragTerm)

      ! Compute layer thickness and common terms

      #ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc    present(cellsOnEdge, maxLevelEdgeTop, botDragTerm, &
      !$acc            kineticEnergyCell, edgeThickTot, &
      !$acc            layerThickness, layerThicknessEdge) &
      !$acc    private(k, kmax, cell1, cell2)
      #else
      !$omp parallel
      !$omp do schedule(runtime) &
      !$omp    private(k, kmax, cell1, cell2)
      #endif
      do iEdge = 1, nEdgesOwned
         cell1 = cellsOnEdge(1,iEdge)
         cell2 = cellsOnEdge(2,iEdge)
         kmax = maxLevelEdgeTop(iEdge)
         edgeThickTot(iEdge) = 0.0_RKIND
         do k = 1,kmax
            layerThicknessEdge(k,iEdge) = 0.5_RKIND* &
                                          (layerThickness(k,cell1) + &
                                           layerThickness(k,cell2))
            edgeThickTot(iEdge) = edgeThickTot(iEdge) + &
                                  layerThicknessEdge(k,iEdge)
         end do
         botDragTerm(iEdge) = dt*implicitBottomDragCoef* &
                              sqrt(kineticEnergyCell(kmax,cell1) + &
                                   kineticEnergyCell(kmax,cell2))/ &
                              layerThicknessEdge(kmax,iEdge)
      end do
      #ifndef MPAS_OPENACC
      !$omp end do
      #endif

      ! Tridiagonal solver for velocity

      #ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc    present(maxLevelEdgeTop, vertViscTopOfEdge, &
      !$acc            layerThicknessEdge, normalVelocity, &
      !$acc            botDragTerm, edgeThickTot) &
      !$acc    private(k, kmax, mTmp, A, B, C, bTemp, rTemp, &
      !$acc            rayleighTerm)
      #else
      !$omp do schedule(runtime) &
      !$omp    private(k, kmax, mTmp, A, B, C, bTemp, rTemp, &
      !$omp            rayleighTerm)
      #endif
      do iEdge = 1, nEdgesOwned
         kmax = maxLevelEdgeTop(iEdge)
         if (kmax > 0) then

         ! Compute tridiagonal matrix

         ! A is lower diagonal term
         ! C is upper diagonal term
         A(1)=0.0_RKIND
         do k = 2, kmax
            A(k  ) = -2.0_RKIND*dt*vertViscTopOfEdge(k,iEdge) &
                   / (layerThicknessEdge(k-1,iEdge) + &
                      layerThicknessEdge(k  ,iEdge)) &
                   /  layerThicknessEdge(k  ,iEdge)
            C(k-1) = -2.0_RKIND*dt*vertViscTopOfEdge(k,iEdge) &
                   / (layerThicknessEdge(k-1,iEdge) + &
                      layerThicknessEdge(k  ,iEdge)) &
                   /  layerThicknessEdge(k-1,iEdge)
         enddo
         C(kmax) = 0.0_RKIND

         ! B is diagonal term
         rayleighTerm = dt*(rayleighDampingCoef/ &
                            ((1.0_RKIND - rayleighDepthVariable) + &
                            rayleighDepthVariable*edgeThickTot(iEdge)))
         do k = 1, kmax
            B(k) = 1.0_RKIND - A(k) - C(k) + rayleighTerm
         enddo

         ! Apply bottom drag boundary condition on the viscous term
         ! second line uses sqrt(2.0*kineticEnergyEdge(k,iEdge))
         B(kmax) = B(kmax) + botDragTerm(iEdge) &
                 + dt*rayleighBottomDampingCoef

         ! First pass: forward solve
         bTemp(1) = B(1)
         rTemp(1) = normalVelocity(1,iEdge)
         do k = 2,kmax
            mTmp = A(k)/bTemp(k-1)
            bTemp(k) = B(k) - mTmp*C(k-1)
            rTemp(k) = normalVelocity(k,iEdge) - mTmp*rTemp(k-1)
         end do

         normalVelocity(kmax,iEdge) = rTemp(kmax)/bTemp(kmax)
         do k = kmax-1, 1, -1
            normalVelocity(k,iEdge) = (rTemp(k) - &
                          C(k)*normalVelocity(k+1,iEdge))/bTemp(k)
         end do
         do k=kmax+1,nVertLevels
            normalVelocity(k,iEdge) = 0.0_RKIND
         end do

         end if
      end do
      #ifndef MPAS_OPENACC
      !$omp end do
      !$omp end parallel
      #endif

      !$acc exit data &
      !$acc    copyout(layerThicknessEdge, normalVelocity) &
      !$acc    delete(kineticEnergyCell, vertViscTopOfEdge, &
      !$acc           layerThickness, botDragTerm, &
      !$acc           A, B, C, bTemp, rTemp, edgeThickTot)

      deallocate(A,B,C,bTemp,rTemp,edgeThickTot,botDragTerm)

   !--------------------------------------------------------------------

   end subroutine ocn_vel_vmix_tend_implicit_rayleigh!}}}

!***********************************************************************
!
!  routine ocn_vel_vmix_tend_implicit
!
!> \brief   Computes tendencies for implicit momentum vertical mixing
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine computes the tendencies for implicit vertical mixing for momentum
!>  using computed coefficients.
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_vmix_tend_implicit(dt, &
                               kineticEnergyCell, vertViscTopOfEdge, &
                               layerThickness, layerThicknessEdge, &
                               normalVelocity, err) !{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         kineticEnergyCell, &!< [in] kinetic energy at cell center
         vertViscTopOfEdge, &!< [in] vertical viscosity at top
         layerThickness      !< [in] thickness at cell center

      real (kind=RKIND), intent(in) :: &
         dt              !< [in] time step

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         normalVelocity,     &!< [inout] vertical velocity
         layerThicknessEdge   !< [inout] thickness at edge

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: &
         iEdge, k,     &! loop iterators for vertical, edge
         cell1, cell2, &! cell index for neighbor cells accross edge
         kmax           ! deepest layer index at top of edge
      integer :: N

      real (kind=RKIND), dimension(:), allocatable :: &
         A, B, C,      &! tridiagonal matrix
         bTemp, rTemp, &! intermediate temporaries in tridiagonal solve
         botDragTerm    ! bottom drag

      real (KIND=RKIND) :: &
         rayleighTerm,   &! contributions from Rayleigh drag
         mTmp             ! temporary tridiagonal factor

      ! End preamble
      !-------------
      ! Begin code

      ! Return if mixing not turned on

      err = 0
      if (.not. velVmixOn) return

      ! allocate various arrays

      allocate(A(nVertLevels), &
               B(nVertLevels), &
               C(nVertLevels), &
               bTemp(nVertLevels), &
               rTemp(nVertLevels), &
               botDragTerm (nEdgesOwned))

      ! Create space on device (not that subroutine arguments
      ! are assumed to have already been transferred to device)
      !$acc enter data &
      !$acc    copyin(kineticEnergyCell, vertViscTopOfEdge, &
      !$acc           layerThickness, layerThicknessEdge, &
      !$acc           normalVelocity) &
      !$acc    create(A, B, C, bTemp, rTemp, botDragTerm)

      ! Compute layer thickness and common terms

      #ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc    present(cellsOnEdge, maxLevelEdgeTop, &
      !$acc            kineticEnergyCell, botDragTerm, &
      !$acc            layerThickness, layerThicknessEdge) &
      !$acc    private(k, kmax, cell1, cell2)
      #else
      !$omp parallel
      !$omp do schedule(runtime) &
      !$omp    private(k, kmax, cell1, cell2)
      #endif
      do iEdge = 1, nEdgesOwned
         cell1 = cellsOnEdge(1,iEdge)
         cell2 = cellsOnEdge(2,iEdge)
         kmax = maxLevelEdgeTop(iEdge)
         do k = 1,kmax
            layerThicknessEdge(k,iEdge) = 0.5_RKIND* &
                                          (layerThickness(k,cell1) + &
                                           layerThickness(k,cell2))
         end do
         botDragTerm(iEdge) = dt*implicitBottomDragCoef* &
                              sqrt(kineticEnergyCell(kmax,cell1) + &
                                   kineticEnergyCell(kmax,cell2))/ &
                              layerThicknessEdge(kmax,iEdge)
      end do
      #ifndef MPAS_OPENACC
      !$omp end do
      #endif

      ! Tridiagonal solver for velocity

      #ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc    present(maxLevelEdgeTop, vertViscTopOfEdge, &
      !$acc            layerThicknessEdge, normalVelocity, &
      !$acc            botDragTerm) &
      !$acc    private(k, kmax, mTmp, A, B, C, bTemp, rTemp)
      #else
      !$omp do schedule(runtime) &
      !$omp    private(k, kmax, mTmp, A, B, C, bTemp, rTemp)
      #endif
      do iEdge = 1, nEdgesOwned
         kmax = maxLevelEdgeTop(iEdge)
         if (kmax > 0) then

         ! Compute tridiagonal matrix

         ! A is lower diagonal term
         ! C is upper diagonal term
         A(1)=0.0_RKIND
         do k = 2, kmax
            A(k  ) = -2.0_RKIND*dt*vertViscTopOfEdge(k,iEdge) &
                   / (layerThicknessEdge(k-1,iEdge) + &
                      layerThicknessEdge(k  ,iEdge)) &
                   /  layerThicknessEdge(k  ,iEdge)
            C(k-1) = -2.0_RKIND*dt*vertViscTopOfEdge(k,iEdge) &
                   / (layerThicknessEdge(k-1,iEdge) + &
                      layerThicknessEdge(k  ,iEdge)) &
                   /  layerThicknessEdge(k-1,iEdge)
         enddo
         C(kmax) = 0.0_RKIND

         ! B is diagonal term
         do k = 1, kmax
            B(k) = 1.0_RKIND - A(k) - C(k)
         enddo

         ! Apply bottom drag boundary condition on the viscous term
         B(kmax) = B(kmax) + botDragTerm(iEdge)

         ! First pass: forward solve
         bTemp(1) = B(1)
         rTemp(1) = normalVelocity(1,iEdge)
         do k = 2,kmax
            mTmp = A(k)/bTemp(k-1)
            bTemp(k) = B(k) - mTmp*C(k-1)
            rTemp(k) = normalVelocity(k,iEdge) - mTmp*rTemp(k-1)
         end do

         ! Second pass: back-substition
         normalVelocity(kmax,iEdge) = rTemp(kmax)/bTemp(kmax)
         do k = kmax-1, 1, -1
            normalVelocity(k,iEdge) = (rTemp(k) - &
                          C(k)*normalVelocity(k+1,iEdge))/bTemp(k)
         end do
         do k=kmax+1,nVertLevels
            normalVelocity(k,iEdge) = 0.0_RKIND
         end do

        end if
      end do
      #ifndef MPAS_OPENACC
      !$omp end do
      !$omp end parallel
      #endif

      !$acc exit data &
      !$acc    copyout(normalVelocity, layerThicknessEdge) &
      !$acc    delete(kineticEnergyCell, vertViscTopOfEdge, &
      !$acc           layerThickness, botDragTerm, &
      !$acc           A, B, C, bTemp, rTemp)

      deallocate(A,B,C,bTemp,rTemp,botDragTerm)

   !--------------------------------------------------------------------

   end subroutine ocn_vel_vmix_tend_implicit!}}}

!***********************************************************************
!
!  routine ocn_tracer_vmix_tend_implicit
!
!> \brief   Computes tendencies for implicit tracer vertical mixing
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine computes the tendencies for implicit vertical mixing
!>  for tracers using computed coefficients.
!
!-----------------------------------------------------------------------

   subroutine ocn_tracer_vmix_tend_implicit(dt, vertDiffTopOfCell, &
                   layerThickness, tracers, &
                   vertNonLocalFlux, tracerGroupSurfaceFlux, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: &
         dt                      !< [in] time step

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         vertDiffTopOfCell,     &!< [in] vertical mixing coefficients
         layerThickness,        &!< [in] thickness at cell center
         tracerGroupSurfaceFlux  !< [in] surface flux for tracers

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
         vertNonLocalFlux        !< [in] non local flux at interfaces

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
         tracers        !< [inout] tracers

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: &
         iCell,k,n,   &! loop iterators for cells, layers, tracers
         numTracers,  &! number of tracers 
         kmax          ! index of deepest active layer

      real (KIND=RKIND) :: &
         mTmp          ! temp scalar for tridiag solve

      real (kind=RKIND), dimension(:), allocatable :: &
         A,B,C,       &! tridiagonal matrix
         bTemp         ! temporary for tridiag solve

      real (kind=RKIND), dimension(:,:), allocatable :: &
         rhs,         &! rhs for tridiagonal solver
         rTemp         ! temporary or tridiag solve

      ! End preamble
      !-------------
      ! Begin code

      ! exit if not turned on

      err = 0
      if( .not. tracerVmixOn) return
      call mpas_timer_start('vmix tracers tend imp loop', .false.)

      ! allocate local arrays

      numTracers = size(tracers, dim=1)

      allocate(A(nVertLevels), &
               B(nVertLevels), &
               C(nVertLevels), &
               bTemp(nVertLevels), &
               rhs(numTracers,nVertLevels), &
               rTemp(numTracers,nVertLevels))

      ! Create space on device (not that subroutine arguments
      ! are assumed to have already been transferred to device)
      !$acc enter data &
      !$acc    copyin(vertDiffTopOfCell, layerThickness, tracers, &
      !$acc           tracerGroupSurfaceFlux, vertNonLocalFlux) &
      !$acc    create(A, B, C, bTemp, rhs, rTemp)

      #ifdef MPAS_OPENACC
      !$acc parallel loop &
      !$acc    present(maxLevelCell, vertDiffTopOfCell, &
      !$acc            layerThickness, tracers, &
      !$acc            tracerGroupSurfaceFlux, vertNonLocalFlux) &
      !$acc    private(n, k, kmax, mTmp, A, B, C, bTemp, rhs, rTemp)
      #else
      !$omp parallel do schedule(runtime) &
      #endif
      do iCell = 1, nCellsOwned
         kmax = maxLevelCell(iCell)

         ! A is lower diagonal term
         ! C is upper diagonal term
         A(1)=0.0_RKIND
         do k=2,kmax
            A(k  ) = -2.0_RKIND*dt*vertDiffTopOfCell(k,iCell) &
                     /(layerThickness(k-1,iCell) + &
                       layerThickness(k  ,iCell)) &
                     / layerThickness(k  ,iCell)
            C(k-1) = -2.0_RKIND*dt*vertDiffTopOfCell(k,iCell) &
                     /(layerThickness(k-1,iCell) + &
                       layerThickness(k  ,iCell)) &
                     / layerThickness(k-1,iCell)
         enddo
         C(kmax) = 0.0_RKIND

         ! B is diagonal term
         do k=1,kmax
            B(k) = 1.0_RKIND - A(k) - C(k)
         enddo

         do k=1,kmax
         do n=1,numTracers
            rhs(n,k) = tracers(n,k,iCell)
         enddo
         enddo

         ! if necessary, add non-local term to rhs
         !   note that the temperature profile (n=1) used for 
         !   all tracers
         if ( kppNonLocalOn ) then

            do k=2,kmax-1
            do n=1,numTracers
               rhs(n,k) = rhs(n,k) + &
                          dt*tracerGroupSurfaceFlux(n,iCell)* &
                          (vertNonLocalFlux(1,k  ,iCell) - &
                           vertNonLocalFlux(1,k+1,iCell))/ &
                          layerThickness(k,iCell)
            enddo
            enddo

            ! special cases for k=1 and k=kmax
            do n=1,numTracers
               rhs(n,1) = rhs(n,1) + &
                          dt*tracerGroupSurfaceFlux(n,iCell)* &
                          (-vertNonLocalFlux(1,2,iCell))/ &
                           layerThickness(1,iCell)
               rhs(n,kmax) = rhs(n,kmax) + &
                             dt*tracerGroupSurfaceFlux(n,iCell) * &
                             vertNonLocalFlux(1,kmax,iCell)/ &
                             layerThickness(kmax,iCell)
            enddo

         endif !kpp nonlocal

         ! First pass: forward elimination
         bTemp(1) = B(1)
         do n = 1,numTracers
            rTemp(n,1) = rhs(n,1)
         end do
         do k = 2,kmax
            mTmp = A(k)/bTemp(k-1)
            bTemp(k) = B(k) - mTmp*C(k-1)
            do n = 1,numTracers
               rTemp(n,k) = rhs(n,k) - mTmp*rTemp(n,k-1)
            end do
         end do

         ! Initialize new tracers to a special value
         ! Ocean domain will be replaced by new tracers from solve
         do k = 1, nVertLevels
         do n = 1,numTracers
            tracers(n,k,iCell) = -1.e34
         end do
         end do

         ! Second pass: back-substition
         do n = 1,numTracers
            tracers(n,kmax,iCell) = rTemp(n,kmax)/bTemp(kmax)
         end do
         do k = kmax-1, 1, -1
            do n = 1,numTracers
               tracers(n,k,iCell) = (rTemp(n,k) - &
                            C(k)*tracers(n,k+1,iCell))/bTemp(k)
            end do
         end do

      end do
      #ifndef MPAS_OPENACC
      !$omp end parallel do
      #endif

      !$acc exit data &
      !$acc    copyout(tracers) &
      !$acc    delete(vertDiffTopOfCell, layerThickness, &
      !$acc           tracerGroupSurfaceFlux, vertNonLocalFlux, &
      !$acc           A, B, C, bTemp, rhs, rTemp)

      deallocate(A, B, C, bTemp, rTemp, rhs)

      call mpas_timer_stop('vmix tracers tend imp loop')

   !--------------------------------------------------------------------

   end subroutine ocn_tracer_vmix_tend_implicit!}}}

!***********************************************************************
!
!  routine ocn_vmix_implicit
!
!> \brief   Driver for implicit vertical mixing
!> \author  Mark Petersen, Phil Jones
!> \date    September 2011, updated June 2020
!> \details
!>  This routine is a driver for handling implicit vertical mixing
!>  of both momentum and tracers for a block. It's intended to reduce
!>  redundant code.
!
!-----------------------------------------------------------------------

   subroutine ocn_vmix_implicit(dt, diagnosticsPool, statePool, &
                                forcingPool, err, timeLevelIn)!{{{

      !-----------------------------------------------------------------
      ! Input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), intent(in) :: &
         dt              !< [in] time step

      integer, intent(in), optional :: &
         timeLevelIn     !< [in] optional index to specify time level

      !-----------------------------------------------------------------
      ! Input/output variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(inout) :: &
         diagnosticsPool,   &! [inout] pool containing diagnostic vars
         statePool,         &! [inout] pool containing state vars
         forcingPool         ! [inout] pool containing forcing vars

      !-----------------------------------------------------------------
      ! Output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err  !< [out] error flag

      !-----------------------------------------------------------------
      ! Local variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), pointer :: &
         tracersPool,      &! subpool for extracting tracers from state
         tracersSurfaceFluxPool ! subpool grouping sfc fluxes in forcing

      type (mpas_pool_iterator_type) :: &
         groupItr           ! pool iterator for groups of tracers

      integer :: &
         iCell, iEdge, k,  &! loop iterators for cell, edge, layer
         timeLevel,        &! time level for state variables
         cell1, cell2       ! index for neighbor cells across edge

      ! pointers for extracting fields from pools

      real (kind=RKIND), dimension(:,:), pointer :: &
         normalVelocity,            &! normal velocity at edge
         layerThickness,            &! layer thickness at cell center
         layerThicknessEdge,        &! layer thickness at edge
         vertViscTopOfEdge,         &! vert viscosity at top of edge
         vertViscTopOfCell,         &! vert viscosity at top of cell
         vertDiffTopOfCell,         &! vert diffusivity at top of cell
         kineticEnergyCell,         &! kinetic energy at cell center
         nonLocalSurfaceTracerFlux, &! non-local surface tracer flux
         tracerGroupSurfaceFlux      ! surface flux for tracers in group

      real (kind=RKIND), dimension(:,:,:), pointer :: &
         tracersGroup,              &! tracers in each group
         vertNonLocalFlux,          &! nonlocal vertical mix flux
         activeTracerVertMixTendency ! total vert mix tendency

      real (kind=RKIND), dimension(:,:,:), allocatable :: &
         nonLocalFluxTend            ! tendency from non-local flux

      character (len=StrKIND) :: &
         modifiedGroupName        ! string for creating group names


      ! End preamble
      !-------------
      ! Begin code

      ! initialize error code, timer and time level for state

      call mpas_timer_start('vmix imp')
      err = 0
      if (present(timeLevelIn)) then
         timeLevel = timeLevelIn
      else
         timeLevel = 1
      end if

      ! extract variables from pools

      call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
      call mpas_pool_get_subpool(forcingPool, 'tracersSurfaceFlux', &
                                               tracersSurfaceFluxPool)

      call mpas_pool_get_array(statePool, 'normalVelocity', &
                                           normalVelocity, timeLevel)
      call mpas_pool_get_array(statePool, 'layerThickness', &
                                           layerThickness, timeLevel)

      call mpas_pool_get_array(diagnosticsPool, 'kineticEnergyCell', &
                                                 kineticEnergyCell)
      call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', &
                                                 layerThicknessEdge)
      call mpas_pool_get_array(diagnosticsPool, 'vertViscTopOfEdge', &
                                                 vertViscTopOfEdge)
      call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', &
                                                 vertDiffTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'vertViscTopOfCell', &
                                                 vertViscTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'vertNonLocalFlux', &
                                                 vertNonLocalFlux)
      call mpas_pool_get_array(diagnosticsPool, &
                                      'activeTracerVertMixTendency', &
                                       activeTracerVertMixTendency)

      ! Compute vertical mixing coefficients

      call mpas_timer_start('vmix coefs', .false.)
      call ocn_vmix_coefs(statePool, forcingPool, diagnosticsPool, &
                          err, timeLevel)
      call mpas_timer_stop('vmix coefs')

      ! if using CVMix, then viscosity has to be averaged from cell 
      ! centers to cell edges

      if ( config_use_cvmix ) then

         call mpas_timer_start('CVMix avg', .false.)
         !$omp parallel do schedule(runtime) private(k, cell1, cell2)
         do iEdge=1,nEdgesOwned
            vertViscTopOfEdge(:, iEdge) = 0.0_RKIND
            cell1 = cellsOnEdge(1,iEdge)
            cell2 = cellsOnEdge(2,iEdge)
            do k=1,maxLevelEdgeTop(iEdge)
               vertViscTopOfEdge(k,iEdge) = 0.5_RKIND* &
                         (vertViscTopOfCell(k,cell2) + &
                          vertViscTopOfCell(k,cell1))
            end do
         end do
         !$omp end parallel do
         call mpas_timer_stop('CVMix avg')
      endif

      !
      !  Implicit vertical solve for momentum
      !
      call mpas_timer_start('vmix solve momentum', .false.)
      if (rayleighOn) then
         call ocn_vel_vmix_tend_implicit_rayleigh(dt, kineticEnergyCell, &
          vertViscTopOfEdge, layerThickness, layerThicknessEdge, normalVelocity, err)
      else
         call ocn_vel_vmix_tend_implicit(dt, kineticEnergyCell, &
          vertViscTopOfEdge, layerThickness, layerThicknessEdge, normalVelocity, err)
      end if
      call mpas_timer_stop('vmix solve momentum')

      !
      !  Implicit vertical solve for all tracers
      !

      call mpas_timer_start('vmix solve tracers', .false.)

      ! Iterate over tracer groups
      call mpas_pool_begin_iteration(tracersPool)
      do while ( mpas_pool_get_next_member(tracersPool, groupItr) )

         if ( groupItr%memberType == MPAS_POOL_FIELD ) then

            ! get tracers for this group
            call mpas_pool_get_array(tracersPool, groupItr%memberName, &
                                     tracersGroup, timeLevel)

            ! if computing tracer budgets, store current tracer
            ! value for computing tracer change later
            if (config_compute_active_tracer_budgets) then
               if (trim(groupItr%memberName) == 'activeTracers') then
                  !$omp do schedule(runtime)
                  do iCell = 1, nCellsOwned
                     activeTracerVertMixTendency(:,:,iCell)= &
                                    tracersGroup(:,:,iCell)
                  end do
                  !$omp end do
               endif
            endif

            ! if tracer group is active, retrieve surface fluxes
            ! and call the implicit solver

            if ( associated(tracersGroup) ) then
               if (trim(groupItr%memberName) == 'activeTracers') then
                  call mpas_pool_get_array(tracersSurfaceFluxPool, &
                                      'nonLocalSurfaceTracerFlux', &
                                       tracerGroupSurfaceFlux)
               else
                  modifiedGroupName = trim(groupItr%memberName) // &
                                      "SurfaceFlux"
                  call mpas_pool_get_array(tracersSurfaceFluxPool, &
                                          trim(modifiedGroupName), &
                                           tracerGroupSurfaceFlux)
               endif

               call ocn_tracer_vmix_tend_implicit(dt, &
                        vertDiffTopOfCell, layerThickness, &
                        tracersGroup, vertNonLocalFlux, &
                        tracerGroupSurfaceFlux, err)

            end if

            ! if computing tracer budgets, compute a tendency by
            ! subtracting the new-old tracers and divide by dt
            if (config_compute_active_tracer_budgets) then
               if (trim(groupItr % memberName) == 'activeTracers') then
                  !$omp do schedule(runtime)
                  do iCell = 1, nCellsOwned
                     activeTracerVertMixTendency(:,:,iCell) = &
                        (tracersGroup(:,:,iCell) - &
                         activeTracerVertMixTendency(:,:,iCell))/dt
                  end do
                  !$omp end do
               endif
            endif ! tracer budgets

         end if ! valid tracer group
      end do ! loop over tracer groups

      call mpas_timer_stop('vmix solve tracers')
      call mpas_timer_stop('vmix imp')

      !-----------------------------------------------------------------

   end subroutine ocn_vmix_implicit!}}}

!***********************************************************************
!
!  routine ocn_vmix_init
!
!> \brief   Initializes ocean vertical mixing quantities
!> \author  Mark Petersen, Phil Jones
!> \date    September 2011, updated June 2020
!> \details
!>  This routine initializes a variety of quantities related to
!>  vertical mixing in the ocean.
!
!-----------------------------------------------------------------------

   subroutine ocn_vmix_init(domain, err)!{{{

      !-----------------------------------------------------------------
      ! Input variables
      !-----------------------------------------------------------------

      type (domain_type), intent(inout) :: &
         domain           !< [in] domain information

      !-----------------------------------------------------------------
      ! Output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: err_tmp ! local flag for managing error codes

      ! End preamble
      !-------------
      ! Begin code

      ! Set module defaults

      err = 0

      velVmixOn     = .true.
      tracerVmixOn  = .true.
      rayleighOn    = .false.
      kppNonlocalOn = .false.

      implicitBottomDragCoef    = 0.0_RKIND
      rayleighDampingCoef       = 0.0_RKIND
      rayleighBottomDampingCoef = 0.0_RKIND
      rayleighDepthVariable     = 0.0_RKIND

      ! Reset on/off options based on user input in config

      if (config_disable_vel_vmix .or. &
          config_disable_vel_all_tend) velVmixOn = .false.

      if (config_disable_tr_vmix  .or. &
          config_disable_tr_all_tend) tracerVmixOn = .false.

      if (config_Rayleigh_friction        .or. &
          config_Rayleigh_bottom_friction .or. &
          config_Rayleigh_damping_depth_variable) rayleighOn = .true.

      if (config_cvmix_kpp_nonlocal_with_implicit_mix) &
         kppNonlocalOn = .true.

      ! Set various coefficients based on user input in config

      if (config_use_implicit_bottom_drag) &
         implicitBottomDragCoef = config_implicit_bottom_drag_coeff

      if (config_Rayleigh_friction) &
         rayleighDampingCoef = config_Rayleigh_damping_coeff

      if (config_Rayleigh_bottom_friction) &
         rayleighBottomDampingCoef=config_Rayleigh_bottom_damping_coeff

      if (config_Rayleigh_damping_depth_variable) &
         rayleighDepthVariable = 1.0_RKIND

      ! Set other options specific to parameterizations

      call ocn_vmix_cvmix_init(domain,err_tmp)
      err = ior(err, err_tmp)

      call ocn_vmix_coefs_redi_init(err_tmp)
      err = ior(err, err_tmp)

   !--------------------------------------------------------------------

   end subroutine ocn_vmix_init!}}}

!***********************************************************************

end module ocn_vmix

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
