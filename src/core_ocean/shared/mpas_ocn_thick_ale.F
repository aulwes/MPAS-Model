! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_thick_ale
!
!> \brief MPAS ocean ALE thickness driver
!> \author Mark Petersen
!> \date   August 2013
!> \details
!>  This module contains the routines for computing ALE thickness.
!
!-----------------------------------------------------------------------

module ocn_thick_ale

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timer

   use ocn_constants
   use ocn_config
   use ocn_mesh

   implicit none
   private
   save

   !--------------------------------------------------------------------
   ! Public parameters
   !--------------------------------------------------------------------

   !--------------------------------------------------------------------
   ! Public member functions
   !--------------------------------------------------------------------

   public :: ocn_ALE_thickness, &
             ocn_thick_ale_init

   !--------------------------------------------------------------------
   ! Private module variables
   !--------------------------------------------------------------------

   integer :: configALEthicknessProportionality

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_ALE_thickness
!
!> \brief   Computes desired ALE thickness at new time
!> \author  Mark Petersen
!> \date    August 2013
!> \details
!>  This routine computes the desired Arbitrary Lagrangian-Eulerian (ALE)
!>  thickness at the new time. It uses the ALE formulation, and includes
!>  contributions from SSH variations (z-star), high-frequency divergence
!>  (z-tilde), and imposes a minimum layer thickness.
!
!-----------------------------------------------------------------------

   subroutine ocn_ALE_thickness(meshPool, verticalMeshPool, SSH, ALE_thickness, err, newHighFreqThickness)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool           !< Input: horizonal mesh information

      type (mpas_pool_type), intent(in) :: &
         verticalMeshPool   !< Input: vertical mesh information

      real (kind=RKIND), dimension(:), intent(in) :: &
         SSH     !< Input: sea surface height

      real (kind=RKIND), dimension(:,:), intent(in), optional :: &
         newHighFreqThickness   !< Input: high frequency thickness.  Alters ALE thickness.

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(out) :: &
         ALE_thickness     !< Output: desired thickness at new time

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: iCell, k, i, kMax, nlvl, nCells
      integer, pointer :: nCellsPtr, nVertLevels
      integer, dimension(:), pointer :: maxLevelCell

      real (kind=RKIND) :: weightSum, thicknessSum, remainder, newThickness, thicknessWithRemainder
      real (kind=RKIND), dimension(:), pointer :: vertCoordMovementWeights
      real (kind=RKIND), dimension(:), allocatable :: &
         SSH_ALE_thickness,      & !> ALE thickness alteration due to SSH (z-star)
         prelim_ALE_Thickness,   & !> ALE thickness at new time
         min_ALE_thickness_down, & !> ALE thickness alteration due to min/max thickness
         min_ALE_thickness_up      !> ALE thickness alteration due to min/max thickness
      real (kind=RKIND), dimension(:,:), pointer :: &
         restingThickness   !>  Layer thickness when the ocean is at rest, i.e. without SSH or internal perturbations.

      logical, pointer :: thicknessFilterActive
      logical, pointer :: config_use_min_max_thickness
      real (kind=RKIND) :: ale_thick
      real (kind=RKIND), pointer :: config_max_thickness_factor
      real (kind=RKIND), pointer :: config_min_thickness

      err = 0

      call mpas_pool_get_package(ocnPackages, 'thicknessFilterActive', thicknessFilterActive)

      call mpas_pool_get_config(ocnConfigs, 'config_use_min_max_thickness', config_use_min_max_thickness)
      call mpas_pool_get_config(ocnConfigs, 'config_max_thickness_factor', config_max_thickness_factor)
      call mpas_pool_get_config(ocnConfigs, 'config_min_thickness', config_min_thickness)

      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'vertCoordMovementWeights', vertCoordMovementWeights)

      call mpas_pool_get_array(verticalMeshPool, 'restingThickness', restingThickness)

      call mpas_pool_get_dimension(meshPool, 'nCells', nCellsPtr)
      nCells = nCellsPtr
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)

      allocate(SSH_ALE_thickness(nVertLevels), prelim_ALE_thickness(nVertLevels), &
         min_ALE_thickness_down(nVertLevels), min_ALE_thickness_up(nVertLevels))

#ifdef MPAS_OPENACC
	  !$omp master
      !$acc enter data copyin(vertCoordMovementWeights,restingThickness) &
      !$acc       async(1)
      !$omp end master
#endif
      
      !
      ! ALE thickness alteration due to SSH (z-star)
      !
      if (configALEthicknessProportionality==1) then ! restingThickness_times_weights


#ifdef MPAS_OPENACC
	     !$omp master
	     !$acc parallel loop &
	     !$acc          present(ALE_Thickness,SSH,vertCoordMovementWeights,&
	     !$acc                  maxLevelCell,restingThickness) &
	     !$acc          private(thicknessSum, kMax) async(1)
#else
         !$omp do schedule(runtime)
#endif         
         do iCell = 1, nCells
            kMax = maxLevelCell(iCell)
   
            thicknessSum = 1e-14_RKIND
            do k = 1, kMax
               thicknessSum = thicknessSum + vertCoordMovementWeights(k) * restingThickness(k, iCell)
            end do
   
   			! computing this outside of loop below does change the answer when run on GPUs
            thicknessSum = 1.0 / thicknessSum
            
            ! Note that restingThickness is nonzero, and remaining terms are perturbations about zero.
            ! This is equation 4 and 6 in Petersen et al 2015, but with eqn 6
            do k = 1, kMax
               ALE_Thickness(k, iCell) = restingThickness(k, iCell) + &
                            (SSH(iCell) * vertCoordMovementWeights(k) * restingThickness(k, iCell)) * thicknessSum
            end do
         enddo
#ifdef MPAS_OPENACC
	     !$omp end master
#else	     
         !$omp end do
#endif         
   
         elseif (configALEthicknessProportionality==2) then ! weights_only
         !$omp do schedule(runtime)
         do iCell = 1, nCells
            kMax = maxLevelCell(iCell)
   
            weightSum = 1e-14_RKIND
            do k = 1, kMax
               weightSum = weightSum + vertCoordMovementWeights(k) 
            end do
   
            do k = 1, kMax
               ! Using this, we must require that sum(restingThickness(k, iCell))
               ! summed over k is equal to bottomDepth.
               ! This is equation 4 and 6 in Petersen et al 2015, but with eqn 6
               ! altered so only the W_k weights are used. The resting
               ! thickness shown in eqn 6 is not included here.
               ALE_Thickness(k, iCell) = restingThickness(k, iCell) + ssh(iCell) * vertCoordMovementWeights(k) / weightSum
   
            end do
         enddo
         !$omp end do
      endif

      if (thicknessFilterActive) then
         nlvl = maxval(maxLevelCell)
#ifdef MPAS_OPENACC
	     !$omp master
	     !$acc parallel loop gang vector collapse(2) copyin(newHighFreqThickness) &
	     !$acc          present(ALE_Thickness,SSH,vertCoordMovementWeights, &
	     !$acc                  maxLevelCell,restingThickness) &
	     !$acc          private(thicknessSum, kMax) async(1)
#else
         !$omp do schedule(runtime) private(kMax)
#endif         
         do iCell = 1, nCells
            do k = 1,nlvl
               if ( k <= maxLevelCell(iCell) ) then

                  ALE_Thickness(k, iCell) = &
                      ALE_Thickness(k, iCell) &
                      + newHighFreqThickness(k,iCell)
               endif
            enddo
         enddo
#ifdef MPAS_OPENACC
	     !$omp end master
#else	     
         !$omp end do
#endif         
      end if

      !
      ! ALE thickness alteration due to minimum and maximum thickness
      !
      if (config_use_min_max_thickness) then

         !$omp do schedule(runtime)
         do iCell = 1, nCells
            kMax = maxLevelCell(iCell)

            ! go down the column:
            remainder = 0.0_RKIND
            do k = 1, kMax
               newThickness = max( min(ALE_Thickness(k, iCell) + remainder, &
                                      config_max_thickness_factor * restingThickness(k,iCell) ), &
                                  config_min_thickness)
               min_ALE_thickness_down(k) = newThickness - ALE_Thickness(k, iCell)
               remainder = remainder - min_ALE_thickness_down(k)
            end do

            ! go back up the column:
            min_ALE_thickness_up(kMax) = 0.0_RKIND
            do k = kMax-1, 1, -1
               ale_thick = ALE_Thickness(k, iCell) + min_ALE_thickness_down(k)
               newThickness = max( min(ale_thick + remainder, &
                                      config_max_thickness_factor * restingThickness(k,iCell) ), &
                                  config_min_thickness)
               min_ALE_thickness_up(k) = newThickness - ale_thick
               remainder = remainder - min_ALE_thickness_up(k)
            end do
            min_ALE_thickness_up(1) = min_ALE_thickness_up(1) + remainder

            ALE_Thickness(1:kMax, iCell) = ALE_Thickness(1:kMax, iCell) + min_ALE_thickness_down(1:kMax) &
                                         + min_ALE_thickness_up(1:kMax)

         enddo
         !$omp end do
      endif ! config_use_min_max_thickness

#ifdef MPAS_OPENACC
	  !$omp master
      !!$acc wait
      	  
      !$acc exit data delete(vertCoordMovementWeights,restingThickness) &
      !$acc       async(1)
      !$omp end master
#endif
      

      deallocate(min_ALE_thickness_down, min_ALE_thickness_up)

   end subroutine ocn_ALE_thickness!}}}

!***********************************************************************
!
!  routine ocn_thick_ale_init
!
!> \brief   Initializes flags used within diagnostics routines.
!> \author  Mark Petersen
!> \date    August 2013
!> \details
!>  This routine initializes flags related to quantities computed within
!>  other diagnostics routines.
!
!-----------------------------------------------------------------------
   subroutine ocn_thick_ale_init(err)!{{{
      integer, intent(out) :: err !< Output: Error flag
      character (len=StrKIND), pointer :: config_ALE_thickness_proportionality

      call mpas_pool_get_config(ocnConfigs,'config_ALE_thickness_proportionality',config_ALE_thickness_proportionality)

      if (config_ALE_thickness_proportionality=='restingThickness_times_weights') then
          configALEthicknessProportionality = 1
      elseif (config_ALE_thickness_proportionality=='weights_only') then
          configALEthicknessProportionality = 2
      else
          call mpas_log_write( &
             ' Warning: config_ALE_thickness_proportionality is not valid', &
             MPAS_LOG_CRIT)
      endif

      err = 0

    end subroutine ocn_thick_ale_init!}}}

end module ocn_thick_ale

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
