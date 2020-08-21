!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_vmix_coefs_redi
!
!> \brief MPAS ocean vertical mixing due to Redi diffusion
!> \author Doug Jacobsen
!> \date   19 September 2011
!> \version SVN:$Id:$
!> \details
!>  This module contains the routines for computing the Redi 
!>  vertical mixing coefficients.
!>
!-----------------------------------------------------------------------

module ocn_vmix_coefs_redi

   use mpas_derived_types
   use mpas_pool_routines
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

   private :: ocn_tracer_vmix_coefs_redi

   public :: ocn_vmix_coefs_redi_build, &
             ocn_vmix_coefs_redi_init

   !--------------------------------------------------------------------
   ! Private module variables
   !--------------------------------------------------------------------

   logical :: rediDiffOn ! on/off switch for use of Redi diffusion

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_vmix_coefs_redi_build
!
!> \brief   Computes coefficients for vertical mixing
!> \author  Doug Jacobsen
!> \date    19 September 2011
!> \version SVN:$Id$
!> \details
!>  This routine computes the vertical mixing coefficients for momentum
!>  and tracers based user choices of mixing parameterization.
!
!-----------------------------------------------------------------------

   subroutine ocn_vmix_coefs_redi_build(statePool, diagnosticsPool, &
                                        err, timeLevelIn)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      integer, intent(in), optional :: &
         timeLevelIn       !< [in] Time level for state pool

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(inout) :: &
         statePool,         &!< [inout] state information
         diagnosticsPool     !< [inout] diagnostic information

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), pointer :: &
        vertDiffTopOfCell, k33
      real (kind=RKIND), dimension(:), pointer :: &
        RediKappa

      ! End of preamble
      !----------------
      ! Begin code

      ! Extract variables from diagnostic pool

      err = 0

      call mpas_pool_get_array(diagnosticsPool, 'vertDiffTopOfCell', &
                                                 vertDiffTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'k33', k33)
      call mpas_pool_get_array(diagnosticsPool, 'RediKappa', RediKappa)

      if (config_use_Redi) then
         call ocn_tracer_vmix_coefs_redi(vertDiffTopOfCell, k33, &
                                         RediKappa, err)
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_vmix_coefs_redi_build!}}}

!***********************************************************************
!
!  routine ocn_tracer_vmix_coefs_redi
!
!> \brief   Computes coefficients for Redi vertical tracer mixing
!> \author  Doug Jacobsen
!> \date    19 September 2011
!> \version SVN:$Id$
!> \details
!>  This routine computes the Redi vertical mixing coefficients for 
!>  tracers
!
!-----------------------------------------------------------------------

   subroutine ocn_tracer_vmix_coefs_redi(vertDiffTopOfCell, k33, &
                                         RediKappa, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:), intent(in) :: &
         RediKappa

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         k33

      !-----------------------------------------------------------------
      ! input/output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(inout) :: &
         vertDiffTopOfCell !< Output: Vertical diffusion

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: iCell ! loop iterator for cells

      ! End of preamble
      !----------------
      ! Begin code

      ! return if not enabled, otherwise start timer

      err = 0
      if(.not.rediDiffOn) return

      call mpas_timer_start('tracer redi coef')

      ! Add Redi diffusion to the total diffusion at top of cell

      !$omp do schedule(runtime)
      do iCell = 1, nCellsOwned
         vertDiffTopOfCell(:, iCell) = vertDiffTopOfCell(:, iCell) &
                                     + RediKappa(iCell)*k33(:, iCell)
      end do
      !$omp end do

      call mpas_timer_stop('tracer redi coef')

   !--------------------------------------------------------------------

   end subroutine ocn_tracer_vmix_coefs_redi!}}}


!***********************************************************************
!
!  routine ocn_vmix_coefs_redi_init
!
!> \brief   Initializes ocean vertical mixing due to Redi diffusion
!> \author  Doug Jacobsen
!> \date    19 September 2011
!> \version SVN:$Id$
!> \details
!>  This routine initializes a quantities related to vertical mixing
!>  due to Redi diffusion.
!
!-----------------------------------------------------------------------


   subroutine ocn_vmix_coefs_redi_init(err)!{{{


      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      ! End preamble
      !-------------
      ! Begin code

      err = 0

      ! Turn on Redi diffusion if requested by user in ocn_config

      if (config_use_Redi) then
         rediDiffOn = .True.
      else
         rediDiffOn = .False.
      end if

   !--------------------------------------------------------------------

   end subroutine ocn_vmix_coefs_redi_init!}}}

!***********************************************************************

end module ocn_vmix_coefs_redi

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

! vim: foldmethod=marker
