! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vel_hadv_coriolis
!
!> \brief MPAS ocean horizontal momentum mixing driver
!> \author Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date   September 2011
!> \details
!>  This module contains the routine for computing
!>  tendencies from the horizontal advection and coriolis force.
!>
!
!-----------------------------------------------------------------------

module ocn_vel_hadv_coriolis

   use mpas_timer
   use mpas_derived_types
   use mpas_pool_routines
   use ocn_constants
   use ocn_config
   use ocn_mesh
   
   implicit none
   private
   save

   !--------------------------------------------------------------------
   !
   ! Public parameters
   !
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: ocn_vel_hadv_coriolis_tend, &
             ocn_vel_hadv_coriolis_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: hadvAndCoriolisOn
   integer :: RK4On

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_vel_hadv_coriolis_tend
!
!> \brief   Computes tendency term for horizontal advection and coriolis force
!> \author  Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date    September 2011
!> \details

!>  This routine computes the horizontal advection and coriolis and
!>  advection tendencies for momentum based on current state.

!
!-----------------------------------------------------------------------

   subroutine ocn_vel_hadv_coriolis_tend(meshPool, normalizedRelativeVorticityEdge, normalizedPlanetaryVorticityEdge, &
                                    layerThicknessEdge, normalVelocity, kineticEnergyCell, tend, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalizedRelativeVorticityEdge, &!< Input: relative vorticity over thickness, on an edge
         normalizedPlanetaryVorticityEdge,&!< Input: planetary vorticity over thickness, on an edge
         layerThicknessEdge,&!< Input: Thickness on edge
         normalVelocity,&    !< Input: Horizontal velocity
         kineticEnergyCell   !< Input: Kinetic Energy

      type (mpas_pool_type), intent(in) :: &
         meshPool          !< Input: mesh information

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         tend          !< Input/Output: velocity tendency

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer, dimension(:), pointer :: maxLevelEdgeTop, nEdgesOnEdge
      integer, dimension(:,:), pointer :: cellsOnEdge, edgesOnEdge
      real (kind=RKIND), dimension(:,:), pointer :: weightsOnEdge
      real (kind=RKIND), dimension(:), pointer :: dcEdge

      integer :: j, k
      integer :: cell1, cell2, iEdge, eoe, nEdges
      integer, pointer :: nVertLevels
      integer :: iblk, mxlvl
      integer, dimension(:), pointer :: nEdgesArray
      real (kind=RKIND) :: workVorticity, invLength, edgeWeight, r_tmp, work, nve

      err = 0

      if ( .not. hadvAndCoriolisOn ) return

      call mpas_timer_start("coriolis")

      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
      call mpas_pool_get_array(meshPool, 'nEdgesOnEdge', nEdgesOnEdge)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'edgesOnEdge', edgesOnEdge)
      call mpas_pool_get_array(meshPool, 'weightsOnEdge', weightsOnEdge)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)

      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)

      nEdges = nEdgesArray( 1 )
      mxlvl = maxval(maxLevelEdgeTop)
      
#ifdef MPAS_OPENACC
	  !$acc parallel loop gang vector collapse(2) &
	  !$acc          present(maxLevelEdgeTop,cellsOnEdge,dcEdge,nEdgesOnEdge, &
	  !$acc                  edgesOnEdge,weightsOnEdge,normalizedRelativeVorticityEdge, &
	  !$acc                  normalizedPlanetaryVorticityEdge,layerThicknessEdge, tend, &
	  !$acc                  edgeMask,kineticEnergyCell,normalVelocity) &
	  !$acc         private(work, cell1, cell2, invLength, j, eoe, nve, edgeWeight, workVorticity) async(1)
#else
      !$omp parallel do schedule(runtime) private(work, cell1, cell2, invLength, k, j, eoe, nve, edgeWeight, workVorticity)
#endif      
      do iEdge = 1, nEdges
         do k = 1, mxlvl
            if ( k <= maxLevelEdgeTop(iEdge) ) then
               cell1 = cellsOnEdge(1,iEdge)
               cell2 = cellsOnEdge(2,iEdge)

               invLength = 1.0_RKIND / dcEdge(iEdge)
               work = 0.0_RKIND
               nve = normalizedRelativeVorticityEdge(k, iEdge) + RK4On * normalizedPlanetaryVorticityEdge(k, iEdge)
         	   do j = 1, nEdgesOnEdge(iEdge)
            	  eoe = edgesOnEdge(j, iEdge)
            	  edgeWeight = weightsOnEdge(j, iEdge)
                  workVorticity = 0.5_RKIND &
                  				 * ( nve + normalizedRelativeVorticityEdge(k, eoe) + RK4On * normalizedPlanetaryVorticityEdge(k, eoe))
            	  work = work + edgeWeight * normalVelocity(k, eoe) * workVorticity * layerThicknessEdge(k, eoe)
               end do
               
               tend(k, iEdge) = tend(k, iEdge) + edgeMask(k, iEdge) * ( work - ( kineticEnergyCell(k, cell2) &
                          - kineticEnergyCell(k, cell1) ) * invLength )
            endif
         end do
      end do
#ifdef MPAS_OPENACC
	  !$acc update host(tend) async(1)
#else
      !$omp end parallel do
#endif      

      call mpas_timer_stop("coriolis")

   !--------------------------------------------------------------------

   end subroutine ocn_vel_hadv_coriolis_tend!}}}

!***********************************************************************
!
!  routine ocn_vel_hadv_coriolis_init
!
!> \brief   Initializes ocean momentum horizontal advection and
!>  coriolis tendencies
!> \author  Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date    September 2011
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_hadv_coriolis_init(err)!{{{

   !--------------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! Output variables
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      err = 0

      hadvAndCoriolisOn = .true.

      if ( config_disable_vel_coriolis ) hadvAndCoriolisOn = .false.

      if ( trim( config_time_integrator ) == 'RK4') then
         ! For RK4, coriolis tendency term includes f: (eta+f)/h.
         RK4On = 1
      elseif ( trim( config_time_integrator ) == 'split_explicit' &
        .or. trim( config_time_integrator ) == 'unsplit_explicit') then
         ! For split explicit, Coriolis tendency uses eta/h because the Coriolis term
         ! is added separately to the momentum tendencies.
         RK4On = 0
      end if


   !--------------------------------------------------------------------

   end subroutine ocn_vel_hadv_coriolis_init!}}}

!***********************************************************************

end module ocn_vel_hadv_coriolis

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
