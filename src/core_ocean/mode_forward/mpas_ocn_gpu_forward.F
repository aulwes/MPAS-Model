module ocn_gpu_forward

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants

   implicit none
   private

   !--------------------------------------------------------------------
   !
   ! Public member functions
   !
   !--------------------------------------------------------------------

   public :: &
   			ocn_gpu_enter_integration, &
   			ocn_gpu_exit_integration, &
   			ocn_gpu_enter_int_step, &
   			ocn_gpu_exit_int_step

contains

!***********************************************************************
!
!  routine ocn_gpu_enter_integration
!
!> \brief   Initialize split-explicit time stepping within MPAS-Ocean core
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine initializes variables required for the split-explicit time
!>  stepper.
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_enter_integration(domain)!{{{

      type (domain_type), intent(inout) :: domain
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: statePool,tendPool
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityNew, normalBaroclinicVelocityNew

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
		 call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocityNew, 2)
		 call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityNew, 2)

	     !$acc enter data create(normalVelocityNew, normalBaroclinicVelocityNew)
         block => block % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_enter_integration
!***********************************************************************
!
!  routine ocn_gpu_exit_int_step
!
!> \brief   Initialize split-explicit time stepping within MPAS-Ocean core
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine initializes variables required for the split-explicit time
!>  stepper.
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_exit_integration(domain)!{{{

      type (domain_type), intent(inout) :: domain
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: statePool,tendPool
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityNew, normalBaroclinicVelocityNew

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
		 call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocityNew, 2)
		 call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityNew, 2)

	     !$acc exit data delete(normalVelocityNew, normalBaroclinicVelocityNew)
         block => block % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_exit_integration
!***********************************************************************
!
!  routine ocn_gpu_exit_int_step
!
!> \brief   Initialize split-explicit time stepping within MPAS-Ocean core
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine initializes variables required for the split-explicit time
!>  stepper.
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_enter_int_step(domain)!{{{

      type (domain_type), intent(inout) :: domain
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: statePool,tendPool
      real (kind=RKIND), dimension(:), pointer :: sshNew
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityTend

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
		 call mpas_pool_get_array(tendPool, 'normalVelocity', normalVelocityTend)
         call mpas_pool_get_array(statePool, 'ssh', sshNew, 2)

	     !!$acc enter data create(sshNew,normalVelocityTend)
	     !$acc enter data create(sshNew)
         block => block % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_enter_int_step

!***********************************************************************
!
!  routine ocn_gpu_exit_int_step
!
!> \brief   Initialize split-explicit time stepping within MPAS-Ocean core
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine initializes variables required for the split-explicit time
!>  stepper.
!
!-----------------------------------------------------------------------
   
   subroutine ocn_gpu_exit_int_step(domain)!{{{

      type (domain_type), intent(inout) :: domain
      type (block_type), pointer :: block
      type (mpas_pool_type), pointer :: statePool,tendPool
      real (kind=RKIND), dimension(:), pointer :: sshNew
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityTend

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
         call mpas_pool_get_array(statePool, 'ssh', sshNew, 2)
		 call mpas_pool_get_array(tendPool, 'normalVelocity', normalVelocityTend)

	     !!$acc exit data delete(sshNew,normalVelocityTend)
	     !$acc exit data delete(sshNew)
         block => block % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_exit_int_step
   
   
end module ocn_gpu_forward
