! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vel_forcing_surface_stress
!
!> \brief MPAS ocean surface stress
!> \author Doug Jacobsen, Mark Petersen, Todd Ringler
!> \date   September 2011
!> \details
!>  This module contains the routine for computing
!>  tendencies from surface stress.
!
!-----------------------------------------------------------------------

module ocn_vel_forcing_surface_stress

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer

   use ocn_constants
   use ocn_config
   use ocn_forcing
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

   public :: ocn_vel_forcing_surface_stress_tend, &
             ocn_vel_forcing_surface_stress_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: surfaceStressOn

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_vel_forcing_surface_stress_tend
!
!> \brief   Computes tendency term from surface stress
!> \author  Doug Jacobsen, Mark Petersen, Todd Ringler
!> \date    15 September 2011
!> \details
!>  This routine computes the surface stress tendency for momentum
!>  based on current state.
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_forcing_surface_stress_tend(meshPool, surfaceFluxAttenuationCoefficient, surfaceStress, & !{{{
                                                  layerThicknessEdge, tend, err)

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:), intent(in) :: &
         surfaceStress, & !< Input: Wind stress at surface
         surfaceFluxAttenuationCoefficient !< Input: attenuation coefficient for surface fluxes

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThicknessEdge     !< Input: thickness at edge

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

      integer :: iEdge, k, cell1, cell2, nEdges
      integer, dimension(:), pointer :: nEdgesArray
      integer, dimension(:), pointer :: maxLevelEdgeTop
      integer, dimension(:,:), pointer :: cellsOnEdge

      real (kind=RKIND) :: transmissionCoeffTop, transmissionCoeffBot, zTop, zBot, remainingStress, &
                           attenuationCoeff, inv_rhosw, mx_force, ztmp

      !-----------------------------------------------------------------
      !
      ! call relevant routines for computing tendencies
      ! note that the user can choose multiple options and the
      !   tendencies will be added together
      !
      !-----------------------------------------------------------------

      err = 0

      if ( .not. surfaceStressOn ) return

      call mpas_timer_start('vel surface stress')

      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)

      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)

      nEdges = nEdgesArray ( 1 )

	  inv_rhosw = 1.0_RKIND / rho_sw
	  mx_force = exp( -100.0_RKIND )
	  
#ifdef MPAS_OPENACC
	  !$acc parallel loop gang vector copyin(surfaceFluxAttenuationCoefficient) &
	  !$acc          present(maxLevelEdgeTop,cellsOnEdge,layerThicknessEdge, &
	  !$acc                  tend,edgeMask,surfaceStress) &
	  !$acc         private(zTop, cell1, cell2, attenuationCoeff, transmissionCoeffTop, &
      !$acc                 remainingStress, k, zBot, transmissionCoeffBot,ztmp) async(1)
#else
      !$omp parallel
      !$omp do schedule(runtime) &
      !$omp private(zTop, cell1, cell2, attenuationCoeff, transmissionCoeffTop, &
      !$omp         remainingStress, k, zBot, transmissionCoeffBot,ztmp)
#endif      
      do iEdge = 1, nEdges
        zTop = 0.0_RKIND
        cell1 = cellsOnEdge(1,iEdge)
        cell2 = cellsOnEdge(2,iEdge)
        attenuationCoeff = 0.5_RKIND * (surfaceFluxAttenuationCoefficient(cell1) &
                                      + surfaceFluxAttenuationCoefficient(cell2))
        transmissionCoeffTop = 1.0_RKIND
        remainingStress = 1.0_RKIND
        do k = 1, maxLevelEdgeTop(iEdge)
           zBot = zTop - layerThicknessEdge(k, iEdge)

           ztmp = zBot/attenuationCoeff
		   if ( ztmp < -100.0_RKIND ) then
		      transmissionCoeffBot = mx_force
		   else
		      transmissionCoeffBot = exp(ztmp)
		   endif
           remainingStress = remainingStress - (transmissionCoeffTop - transmissionCoeffBot)

           tend(k,iEdge) =  tend(k,iEdge) + edgeMask(k, iEdge) * surfaceStress(iEdge) &
                         * (transmissionCoeffTop - transmissionCoeffBot) * inv_rhosw / layerThicknessEdge(k,iEdge)

           zTop = zBot
           transmissionCoeffTop = transmissionCoeffBot
        enddo

		k = maxLevelEdgeTop(iEdge)
        if ( k > 0 .and. remainingStress > 0.0_RKIND) then
           tend(k, iEdge) = tend(k, iEdge) &
                         + edgeMask(k, iEdge) * surfaceStress(iEdge) * remainingStress &
                         * inv_rhosw / layerThicknessEdge(k, iEdge)
        end if
      enddo
#ifdef MPAS_OPENACC
      !$acc update host(tend) async(1)
#else      
      !$omp end do
      !$omp end parallel
#endif      

      call mpas_timer_stop('vel surface stress')

   !--------------------------------------------------------------------

   end subroutine ocn_vel_forcing_surface_stress_tend!}}}

!***********************************************************************
!
!  routine ocn_vel_forcing_surface_stress_init
!
!> \brief   Initializes ocean surface stress forcing
!> \author  Doug Jacobsen, Mark Petersen, Todd Ringler
!> \date    September 2011
!> \details
!>  This routine initializes quantities related to surface stress
!>  in the ocean.
!
!-----------------------------------------------------------------------

   subroutine ocn_vel_forcing_surface_stress_init(err)!{{{

   !--------------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! call individual init routines for each parameterization
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      surfaceStressOn = .true.

      if(config_disable_vel_surface_stress) surfaceStressOn = .false.

      err = 0

   !--------------------------------------------------------------------

   end subroutine ocn_vel_forcing_surface_stress_init!}}}

!***********************************************************************

end module ocn_vel_forcing_surface_stress

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
