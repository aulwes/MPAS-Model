module ocn_gpu_forward

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use openacc
   
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
   			!ocn_gpu_enter_int_step, &
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
      type (mpas_pool_type), pointer :: statePool,diagnosticsPool
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityNew, normalBaroclinicVelocityNew
      real (kind=RKIND), dimension(:,:), pointer :: normalBaroclinicVelocityCur
      real (kind=RKIND), dimension(:), pointer :: barotropicForcing

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
		 call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocityNew, 2)
         call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityCur, 1)
		 call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityNew, 2)
         call mpas_pool_get_array(diagnosticsPool, 'barotropicForcing', barotropicForcing)

	     !$acc enter data copyin(normalVelocityNew, normalBaroclinicVelocityNew)
	     !!$acc                  normalBaroclinicVelocityCur,layerThicknessEdge,barotropicForcing)
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
      type (mpas_pool_type), pointer :: statePool,diagnosticsPool
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityNew, normalBaroclinicVelocityNew
      real (kind=RKIND), dimension(:,:), pointer :: normalBaroclinicVelocityCur
      real (kind=RKIND), dimension(:), pointer :: barotropicForcing

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
		 call mpas_pool_get_array(statePool, 'normalVelocity', normalVelocityNew, 2)
         call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityCur, 1)
		 call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityNew, 2)
         call mpas_pool_get_array(diagnosticsPool, 'barotropicForcing', barotropicForcing)

	     !$acc exit data delete(normalVelocityNew, normalBaroclinicVelocityNew)
	     !!$acc                  normalBaroclinicVelocityCur,layerThicknessEdge,barotropicForcing)
         block => block % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_exit_integration
!***********************************************************************
!
!  routine ocn_gpu_enter_int_step
!
!> \brief   Initialize split-explicit time stepping within MPAS-Ocean core
!> \author  Mark Petersen
!> \date    September 2011
!> \details
!>  This routine initializes variables required for the split-explicit time
!>  stepper.
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_enter_int_step_old(domain)!{{{

      type (domain_type), intent(inout) :: domain
      type (block_type), pointer :: block,tmpblk
      type (mpas_pool_type), pointer :: statePool,diagnosticsPool,tendPool
      real (kind=RKIND), dimension(:), pointer :: sshNew
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityTend,layerThicknessEdge
      real (kind=RKIND), dimension(:,:), pointer :: normalBaroclinicVelocityCur

#ifdef MPAS_OPENACC
	  !$omp master
	  !call acc_present_dump()
	  
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
		 call mpas_pool_get_array(tendPool, 'normalVelocity', normalVelocityTend)
         call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityCur, 1)
         call mpas_pool_get_array(statePool, 'ssh', sshNew, 2)
         call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)

	     !$acc enter data copyin(sshNew)
	     !!$acc enter data copyin(sshNew,layerThicknessEdge,normalBaroclinicVelocityCur)
         block => block % next
      end do
	  !$omp end master
#endif

#ifdef MPAS_OPENACC
			   !$omp master
   	  tmpblk => domain % blocklist
	  do while (associated(tmpblk))
         call mpas_pool_get_subpool(tmpblk % structs, 'diagnostics', diagnosticsPool)
         call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)

	     !!$acc enter data copyin(layerThicknessEdge)
         tmpblk => tmpblk % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_enter_int_step_old

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
      type (mpas_pool_type), pointer :: statePool,diagnosticsPool,tendPool
      real (kind=RKIND), dimension(:), pointer :: sshNew
      real (kind=RKIND), dimension(:,:), pointer :: normalVelocityTend,layerThicknessEdge
      real (kind=RKIND), dimension(:,:), pointer :: normalBaroclinicVelocityCur

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
   	  block => domain % blocklist
	  do while (associated(block))
	     call mpas_pool_get_subpool(block % structs, 'state', statePool)
         call mpas_pool_get_subpool(block % structs, 'tend', tendPool)
         call mpas_pool_get_subpool(block % structs, 'diagnostics', diagnosticsPool)
         call mpas_pool_get_array(statePool, 'ssh', sshNew, 2)
		 call mpas_pool_get_array(tendPool, 'normalVelocity', normalVelocityTend)
         call mpas_pool_get_array(statePool, 'normalBaroclinicVelocity', normalBaroclinicVelocityCur, 1)
         call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)

	     !$acc exit data delete(sshNew)
	     !!$acc exit data delete(sshNew,layerThicknessEdge,normalBaroclinicVelocityCur)
         block => block % next
      end do
	  !$omp end master
#endif

   end subroutine ocn_gpu_exit_int_step
   
   
end module ocn_gpu_forward
