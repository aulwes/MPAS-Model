!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_velocity_solver_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

module seaice_velocity_solver_variational

  use mpas_derived_types
  use mpas_pool_routines
  use mpas_timer

  implicit none

  private
  save

  public :: &
       seaice_init_velocity_solver_variational, &
       seaice_strain_tensor_variational, &
       seaice_stress_divergence_variational, &
       seaice_internal_stress_variational, &
       seaice_stress_init_device, &
       seaice_stress_dealloc_device, &
#ifdef MPAS_OPENACC       
       seaice_internal_stress_update_host, &
#endif       
       seaice_final_divergence_shear_variational

#ifdef MPAS_OPENACC
  real(kind=RKIND), dimension(:,:,:), allocatable, target :: &
         rmem_block

  real(kind=RKIND), dimension(:,:,:,:), allocatable, target :: &
         rmem2d_block

  real(kind=RKIND), dimension(:,:,:,:,:), allocatable, target :: &
         rmem3d_block

  integer, dimension(:,:,:), allocatable, target :: &
         imem_block

  integer, dimension(:,:,:,:), allocatable, target :: &
         imem2d_block


  integer, parameter :: &
  		tanlatp = 1, &    ! tanLatVertexRotatedOverRadius
  		areatrip = 2, &   ! areaTriangle
  		areacellp = 3, &  ! areaCell
  		endflistp = 3
  		
  integer, parameter :: &
  		replp =   1, & ! replacementPressure
  		stra11p = 2, & ! strain11
  		stra12p = 3, &
  		stra22p = 4, &
  		stre11p = 5, &
  		stre12p = 6, &
  		stre22p = 7, &
  		endflist2dp = 7
  		
  integer, parameter :: &
  		gradup =   1, & ! basisGradientU
  		gradvp = 2, &   ! basisGradientV
  		bintup = 3, &   ! basisIntegralsU
  		bintvp = 4, &   ! basisIntegralV
  		metricp = 5, &  ! basisIntegralsMetric
  		endflist3dp = 5

  integer, parameter :: &
  		nedgep =    1,  & ! nEdgesOnCell
  		endilistp = 1
  		
  integer, parameter :: &
  		cellvvp =     1, & ! cellVerticesAtVertex
  		cellsvp =     2, & ! cellsOnVertex
  		voncellsp =   3, & ! verticesOnCell
  		endilist2dp = 3
  		
#endif

contains

!-----------------------------------------------------------------------
! initialization
!-----------------------------------------------------------------------

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_init_velocity_solver_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 24 October 2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_init_velocity_solver_variational(&
       mesh, &
       velocity_variational, &
       boundary, &
       rotateCartesianGrid, &
       includeMetricTerms, &
       variationalBasisType, &
       integrationType, &
       integrationOrder)!{{{

    use seaice_velocity_solver_wachspress, only: &
         seaice_init_velocity_solver_wachspress

    use seaice_velocity_solver_pwl, only: &
         seaice_init_velocity_solver_pwl

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    type(MPAS_pool_type), pointer :: &
         velocity_variational, & !< Input/Output:
         boundary                !< Input/Output:

    logical, intent(in) :: &
         rotateCartesianGrid, & !< Input:
         includeMetricTerms     !< Input:

    character(len=*), intent(in) :: &
         variationalBasisType, & !< Input:
         integrationType         !< Input:

    integer, intent(in) :: &
         integrationOrder !< Input:

    if (trim(variationalBasisType) == "wachspress") then

       call seaice_init_velocity_solver_wachspress(&
            mesh, &
            velocity_variational, &
            boundary, &
            rotateCartesianGrid, &
            includeMetricTerms, &
            integrationType, &
            integrationOrder)

    else if (trim(variationalBasisType) == "pwl") then

       call seaice_init_velocity_solver_pwl(&
            mesh, &
            velocity_variational, &
            boundary, &
            rotateCartesianGrid, &
            includeMetricTerms)

    endif

  end subroutine seaice_init_velocity_solver_variational

!-----------------------------------------------------------------------
! time step
!-----------------------------------------------------------------------

#ifndef MPAS_OPENACC  

  subroutine set_velocity_arrays(iblk, block, &
  							 strain11,strain12,strain22,stress11,stress12,stress22, &
							 cellVerticesAtVertex, tanLatVertexRotatedOverRadius, &
							 basisGradientU,basisGradientV,basisIntegralsU, basisIntegralsV, &
							 basisIntegralsMetric,replacementPressure,nEdgesOnCell, &
							 verticesOnCell,cellsOnVertex,areaCell,areaTriangle)  

	integer, intent(in) :: iblk

    type(block_type), intent(in), pointer :: &
         block

    real(kind=RKIND), dimension(:), intent(out), pointer :: &
    	 tanLatVertexRotatedOverRadius, &
    	 areaCell, &
    	 areaTriangle

    real(kind=RKIND), dimension(:,:), intent(out), pointer :: &
         replacementPressure, &
         strain11, &
         strain22, &
         strain12, &
         stress11, &
         stress22, &
         stress12

    real(kind=RKIND), dimension(:,:,:), intent(out), pointer :: &
         basisGradientU, &
         basisGradientV, &
         basisIntegralsU, &
         basisIntegralsV, &
         basisIntegralsMetric


    integer, dimension(:), intent(out), pointer :: &
         nEdgesOnCell

    integer, dimension(:,:), intent(out), pointer :: &
         cellVerticesAtVertex, &
         cellsOnVertex, &
         verticesOnCell

    type (MPAS_pool_type), pointer :: &
         mesh, &
         velocityVariationalPool, &
         velocitySolverPool

       call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
       call MPAS_pool_get_subpool(block % structs, "velocity_variational", velocityVariationalPool)
       call MPAS_pool_get_subpool(block % structs, "velocity_solver", velocitySolverPool)

       call MPAS_pool_get_array(velocityVariationalPool, "strain11", strain11)
       call MPAS_pool_get_array(velocityVariationalPool, "strain22", strain22)
       call MPAS_pool_get_array(velocityVariationalPool, "strain12", strain12)
       call MPAS_pool_get_array(velocityVariationalPool, "stress11", stress11)
       call MPAS_pool_get_array(velocityVariationalPool, "stress22", stress22)
       call MPAS_pool_get_array(velocityVariationalPool, "stress12", stress12)
       call MPAS_pool_get_array(velocityVariationalPool, "cellVerticesAtVertex", cellVerticesAtVertex)
       call MPAS_pool_get_array(velocityVariationalPool, "tanLatVertexRotatedOverRadius", tanLatVertexRotatedOverRadius)
       call MPAS_pool_get_array(velocityVariationalPool, "basisGradientU", basisGradientU)
       call MPAS_pool_get_array(velocityVariationalPool, "basisGradientV", basisGradientV)
       call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsU", basisIntegralsU)
       call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsV", basisIntegralsV)
       call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsMetric", basisIntegralsMetric)
       call MPAS_pool_get_array(velocityVariationalPool, "replacementPressure", replacementPressure)

	   call MPAS_pool_get_array(mesh, "nEdgesOnCell", nEdgesOnCell)
       call MPAS_pool_get_array(mesh, "cellsOnVertex", cellsOnVertex)
	   call MPAS_pool_get_array(mesh, "verticesOnCell", verticesOnCell)
	   call MPAS_pool_get_array(mesh, "areaCell", areaCell)
       call MPAS_pool_get_array(mesh, "areaTriangle", areaTriangle)
	   							
  end subroutine set_velocity_arrays

#else

  subroutine seaice_internal_stress_update_host(domain)

    type(domain_type), intent(in) :: &
         domain

    type(block_type), pointer :: &
         block

    type (MPAS_pool_type), pointer :: &
         velocityVariationalPool

    real(kind=RKIND), dimension(:,:), pointer :: &
         replacementPressure, &
         strain11, &
         strain22, &
         strain12, &
         stress11, &
         stress22, &
         stress12

    integer :: iblk
        
    ! copy stress/strain back to host
    !$acc update host(rmem2d_block)
    
    iblk = 1
	block => domain % blocklist
	do while (associated(block))
       call MPAS_pool_get_subpool(block % structs, "velocity_variational", velocityVariationalPool)
	
       call MPAS_pool_get_array(velocityVariationalPool, "strain11", strain11)
       call MPAS_pool_get_array(velocityVariationalPool, "strain22", strain22)
       call MPAS_pool_get_array(velocityVariationalPool, "strain12", strain12)
       call MPAS_pool_get_array(velocityVariationalPool, "stress11", stress11)
       call MPAS_pool_get_array(velocityVariationalPool, "stress22", stress22)
       call MPAS_pool_get_array(velocityVariationalPool, "stress12", stress12)
       call MPAS_pool_get_array(velocityVariationalPool, "replacementPressure", replacementPressure)

	   strain11 = rmem2d_block(:,:,iblk,stra11p)
	   strain12 = rmem2d_block(:,:,iblk,stra12p)
	   strain22 = rmem2d_block(:,:,iblk,stra22p)
	   stress11 = rmem2d_block(:,:,iblk,stre11p)
	   stress12 = rmem2d_block(:,:,iblk,stre12p)
	   stress22 = rmem2d_block(:,:,iblk,stre22p)
	   replacementPressure = rmem2d_block(:,:,iblk,replp)

       block => block % next
       iblk = iblk + 1
	enddo    
  
  end subroutine seaice_internal_stress_update_host



  subroutine set_velocity_arrays(iblk, block, &
  							 strain11,strain12,strain22,stress11,stress12,stress22, &
							 cellVerticesAtVertex, tanLatVertexRotatedOverRadius, &
							 basisGradientU,basisGradientV,basisIntegralsU, basisIntegralsV, &
							 basisIntegralsMetric,replacementPressure,nEdgesOnCell, &
							 verticesOnCell,cellsOnVertex,areaCell,areaTriangle)  

	integer, intent(in) :: iblk

    type(block_type), intent(in), pointer :: &
         block

    real(kind=RKIND), dimension(:), intent(out), pointer :: &
    	 tanLatVertexRotatedOverRadius, &
    	 areaCell, &
    	 areaTriangle

    real(kind=RKIND), dimension(:,:), intent(out), pointer :: &
         replacementPressure, &
         strain11, &
         strain22, &
         strain12, &
         stress11, &
         stress22, &
         stress12

    real(kind=RKIND), dimension(:,:,:), intent(out), pointer :: &
         basisGradientU, &
         basisGradientV, &
         basisIntegralsU, &
         basisIntegralsV, &
         basisIntegralsMetric


    integer, dimension(:), intent(out), pointer :: &
         nEdgesOnCell

    integer, dimension(:,:), intent(out), pointer :: &
         cellVerticesAtVertex, &
         cellsOnVertex, &
         verticesOnCell

	   tanLatVertexRotatedOverRadius => rmem_block(:,iblk,tanlatp)
	   areaCell => rmem_block(:,iblk,areacellp)
	   areaTriangle => rmem_block(:,iblk,areatrip)

	   strain11 => rmem2d_block(:,:,iblk,stra11p)
	   strain12 => rmem2d_block(:,:,iblk,stra12p)
	   strain22 => rmem2d_block(:,:,iblk,stra22p)
	   stress11 => rmem2d_block(:,:,iblk,stre11p)
	   stress12 => rmem2d_block(:,:,iblk,stre12p)
	   stress22 => rmem2d_block(:,:,iblk,stre22p)
	   replacementPressure => rmem2d_block(:,:,iblk,replp)

	   basisGradientU => rmem3d_block(:,:,:,iblk,gradup)
	   basisGradientV => rmem3d_block(:,:,:,iblk,gradvp)
	   basisIntegralsU => rmem3d_block(:,:,:,iblk,bintup)
	   basisIntegralsV => rmem3d_block(:,:,:,iblk,bintvp)
	   basisIntegralsMetric => rmem3d_block(:,:,:,iblk,metricp)

	   nEdgesOnCell => imem_block(:,iblk, nedgep)
	   cellVerticesAtVertex => imem2d_block(:,:,iblk, cellvvp)
	   cellsOnVertex => imem2d_block(:,:,iblk, cellsvp)
	   verticesOnCell => imem2d_block(:,:,iblk, voncellsp)
	   	   							
  end subroutine set_velocity_arrays

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_alloc_device
!
!-----------------------------------------------------------------------
  subroutine seaice_stress_alloc_device(domain)!{{{
  
    type(domain_type), intent(inout) :: &
         domain

    type(block_type), pointer :: &
         block

    type (MPAS_pool_type), pointer :: &
         mesh, &
         velocityVariationalPool, &
         velocitySolverPool

    real(kind=RKIND), dimension(:), pointer :: &
         areaTriangle, &
         areaCell, &
         tanLatVertexRotatedOverRadius

    real(kind=RKIND), dimension(:,:), pointer :: &
         replacementPressure, &
         strain11, &
         strain22, &
         strain12, &
         stress11, &
         stress22, &
         stress12

    real(kind=RKIND), dimension(:,:,:), pointer :: &
         basisGradientU, &
         basisGradientV, &
         basisIntegralsU, &
         basisIntegralsV, &
         basisIntegralsMetric

    integer, dimension(:), pointer :: &
         nEdgesOnCell

    integer, dimension(:,:), pointer :: &
         cellVerticesAtVertex, &
         cellsOnVertex, &
         verticesOnCell

    integer :: &
    	 dim1, nblocks, iblk, &
    	 r2ddim1, r2ddim2, &
    	 r3ddim1, r3ddim2, r3ddim3, &
    	 idim1, i2ddim1, i2ddim2

	nblocks = 0
	dim1 = 1
	r2ddim1 = 1
	r2ddim2 = 1

	r3ddim1 = 1
	r3ddim2 = 1
	r3ddim3 = 1
	i2ddim1 = 1
	i2ddim2 = 1
	idim1 = 1
	
	block => domain % blocklist
    do while (associated(block))
       call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
       call MPAS_pool_get_subpool(block % structs, "velocity_variational", velocityVariationalPool)
       call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsU", basisIntegralsU)
       call MPAS_pool_get_array(velocityVariationalPool, "strain11", strain11)
	   call MPAS_pool_get_array(mesh, "areaCell", areaCell)
       call MPAS_pool_get_array(mesh, "areaTriangle", areaTriangle)
       call MPAS_pool_get_array(mesh, "nEdgesOnCell", nEdgesOnCell)
       call MPAS_pool_get_array(mesh, "verticesOnCell", verticesOnCell)
       call MPAS_pool_get_array(mesh, "cellsOnVertex", cellsOnVertex)

   	   r2ddim1 = max(size(strain11, 1), r2ddim1)
   	   r2ddim2 = max(size(strain11, 2), r2ddim2)

   	   r3ddim1 = max(size(basisIntegralsU, 1), r3ddim1)
   	   r3ddim2 = max(size(basisIntegralsU, 2), r3ddim2)
   	   r3ddim3 = max(size(basisIntegralsU, 3), r3ddim3)
   	   
   	   dim1 = max(size(areaCell, 1), dim1)
   	   dim1 = max(size(areaTriangle, 1), dim1)
	   idim1 = max(size(nEdgesOnCell, 1), idim1)
	   
	   i2ddim1 = max(size(verticesOnCell, 1), i2ddim1)
	   i2ddim1 = max(size(cellsOnVertex, 1), i2ddim1)
	   
	   i2ddim2 = max(size(verticesOnCell, 2), i2ddim2)
	   i2ddim2 = max(size(cellsOnVertex, 2), i2ddim2)
   	   
	   nblocks = nblocks + 1
       block => block % next
    end do

	!if ( 0 == domain % dminfo % my_proc_id ) &
	!	print *,'nblocks, dim1, r3ddim1, r3ddim2, r3ddim3, idim1, i2ddim1, i2ddim2 = ', &
	!		nblocks, dim1, r3ddim1, r3ddim2, r3ddim3, idim1, i2ddim1, i2ddim2
			
	if ( nblocks > 0 ) then
	   allocate(rmem_block(dim1,nblocks,endflistp))
	   allocate(rmem2d_block(r2ddim1,r2ddim2,nblocks,endflist2dp))
	   allocate(rmem3d_block(r3ddim1,r3ddim2,r3ddim3,nblocks,endflist3dp))
       allocate(imem_block(idim1,nblocks,endilistp))
       allocate(imem2d_block(i2ddim1,i2ddim2,nblocks,endilist2dp))
	
	   block => domain % blocklist
	   iblk = 1
       do while ( associated(block) )
          call MPAS_pool_get_subpool(block % structs, "mesh", mesh)
          call MPAS_pool_get_subpool(block % structs, "velocity_variational", velocityVariationalPool)
       
          call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsU", basisIntegralsU)
          call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsV", basisIntegralsV)
          call MPAS_pool_get_array(velocityVariationalPool, "basisGradientU", basisGradientU)
          call MPAS_pool_get_array(velocityVariationalPool, "basisGradientV", basisGradientV)
          call MPAS_pool_get_array(velocityVariationalPool, "basisIntegralsMetric", basisIntegralsMetric)
          rmem3d_block(:,:,:,iblk,bintup) = basisIntegralsU
          rmem3d_block(:,:,:,iblk,bintvp) = basisIntegralsV
          rmem3d_block(:,:,:,iblk,gradup) = basisGradientU
          rmem3d_block(:,:,:,iblk,gradvp) = basisGradientV
          rmem3d_block(:,:,:,iblk,metricp) = basisIntegralsMetric
          
	      call MPAS_pool_get_array(mesh, "areaCell", areaCell)
          call MPAS_pool_get_array(mesh, "areaTriangle", areaTriangle)
          call MPAS_pool_get_array(velocityVariationalPool, "tanLatVertexRotatedOverRadius", tanLatVertexRotatedOverRadius)
          rmem_block(:,iblk,areacellp) = areaCell
          rmem_block(:,iblk,areatrip)  = areaTriangle
          rmem_block(:,iblk,tanlatp)   = tanLatVertexRotatedOverRadius
          
          call MPAS_pool_get_array(mesh, "nEdgesOnCell", nEdgesOnCell)
          imem_block(:,iblk,nedgep) = nEdgesOnCell

          call MPAS_pool_get_array(mesh, "cellsOnVertex", cellsOnVertex)
          call MPAS_pool_get_array(mesh, "verticesOnCell", verticesOnCell)
          call MPAS_pool_get_array(velocityVariationalPool, "cellVerticesAtVertex", cellVerticesAtVertex)
          imem2d_block(:,:,iblk,cellsvp) = cellsOnVertex
          imem2d_block(:,:,iblk,voncellsp) = verticesOnCell
          imem2d_block(:,:,iblk,cellvvp) = cellVerticesAtVertex

          call MPAS_pool_get_array(velocityVariationalPool, "strain11", strain11)
          call MPAS_pool_get_array(velocityVariationalPool, "strain22", strain22)
          call MPAS_pool_get_array(velocityVariationalPool, "strain12", strain12)
          call MPAS_pool_get_array(velocityVariationalPool, "stress11", stress11)
          call MPAS_pool_get_array(velocityVariationalPool, "stress22", stress22)
          call MPAS_pool_get_array(velocityVariationalPool, "stress12", stress12)
          call MPAS_pool_get_array(velocityVariationalPool, "replacementPressure", replacementPressure)
          rmem2d_block(:,:,iblk,stra11p) = strain11
          rmem2d_block(:,:,iblk,stra12p) = strain12
          rmem2d_block(:,:,iblk,stra22p) = strain22
          rmem2d_block(:,:,iblk,stre11p) = stress11
          rmem2d_block(:,:,iblk,stre12p) = stress12
          rmem2d_block(:,:,iblk,stre22p) = stress22
          rmem2d_block(:,:,iblk,replp)   = replacementPressure
          
          block => block % next
   		  iblk = iblk + 1       
       enddo
       
       !$acc enter data copyin(rmem_block,rmem2d_block,rmem3d_block, &
       !$acc                   imem_block,imem2d_block) async(1)
    endif   
       	
  end subroutine seaice_stress_alloc_device
  
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_dealloc_device
!
!-----------------------------------------------------------------------
  subroutine seaice_stress_dealloc_device()!{{{

    !$acc exit data delete(rmem_block,rmem2d_block,rmem3d_block, &
    !$acc                   imem_block,imem2d_block)

    deallocate(rmem_block, rmem2d_block, rmem3d_block, imem_block, imem2d_block)

  end subroutine seaice_stress_dealloc_device
  
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_init_device
!
!-----------------------------------------------------------------------
  subroutine seaice_stress_init_device(domain)!{{{
  
    type(domain_type), intent(inout) :: &
         domain


    call seaice_stress_alloc_device(domain)
    
  end subroutine seaice_stress_init_device
  
#endif ! MPAS_OPENACC

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_internal_stress_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_internal_stress_variational(domain, iblk, block, solveStress, solveVelocity, &
          			uVelocity, vVelocity, stressDivergenceU, stressDivergenceV, icePressure)!{{{

    type(domain_type), intent(inout) :: &
         domain

	integer, intent(in) :: &
		 iblk
		 
    type(block_type), intent(in), pointer :: &
         block

    integer, dimension(:), intent(in) :: &
         solveStress, &
         solveVelocity

    real(kind=RKIND), dimension(:), intent(in) :: &
         icePressure, &
    	 uVelocity, &
    	 vVelocity

    real(kind=RKIND), dimension(:), intent(out) :: &
         stressDivergenceU, &
         stressDivergenceV

    type (MPAS_pool_type), pointer :: &
         meshPool, &
         velocitySolverPool

    real(kind=RKIND), pointer :: &
         elasticTimeStep

    logical, pointer :: &
         revisedEVPPtr

	logical :: revisedEVP
	
    integer, dimension(:,:), pointer :: &
         cellVerticesAtVertex, &
         verticesOnCell

    integer, dimension(:), pointer :: &
         nEdgesOnCell

    real(kind=RKIND), dimension(:), pointer :: &
         tanLatVertexRotatedOverRadius

    real(kind=RKIND), dimension(:,:), pointer :: &
         replacementPressure, &
         strain11, &
         strain22, &
         strain12, &
         stress11, &
         stress22, &
         stress12

    real(kind=RKIND), dimension(:,:,:), pointer :: &
         basisGradientU, &
         basisGradientV, &
         basisIntegralsU, &
         basisIntegralsV, &
         basisIntegralsMetric

    integer, dimension(:,:), pointer :: &
         cellsOnVertex

    real(kind=RKIND), dimension(:), pointer :: &
    	 areaCell, &
         areaTriangle

    call MPAS_pool_get_config(block % configs, "config_revised_evp", revisedEVPPtr)
	revisedEVP = revisedEVPPtr
		
    call MPAS_pool_get_subpool(block % structs, "mesh", meshPool)
    call MPAS_pool_get_subpool(block % structs, "velocity_solver", velocitySolverPool)

    call MPAS_pool_get_array(velocitySolverPool, "elasticTimeStep", elasticTimeStep)

	call set_velocity_arrays(iblk,block,strain11,strain12,strain22,stress11,stress12,stress22, &
							 cellVerticesAtVertex, tanLatVertexRotatedOverRadius, &
							 basisGradientU,basisGradientV,basisIntegralsU, basisIntegralsV, &
							 basisIntegralsMetric,replacementPressure,nEdgesOnCell, &
							 verticesOnCell,cellsOnVertex,areaCell,areaTriangle)
							 	   
	! out: stress11,stress12,stress22,strain11,strain12,strain22,replacementPressure
    call mpas_timer_start("Velocity solver stress-strain tensor")
    call seaice_stress_strain_tensor_variational(&
       	    domain % dminfo % my_proc_id, &
            meshPool, &
            nEdgesOnCell, &
            verticesOnCell, &
            stress11, &
            stress22, &
            stress12, &
            strain11, &
            strain22, &
            strain12, &
            uVelocity, &
            vVelocity, &
            basisGradientU, &
            basisGradientV, &
            tanLatVertexRotatedOverRadius, &
            icePressure, &
            replacementPressure, &
            areaCell, &
            solveStress, &
            elasticTimeStep, &
            revisedEVP)
            
    call mpas_timer_stop("Velocity solver stress-strain tensor")
		
	! in: stress11,stress12,stress22
	! out: stressDivergenceU,stressDivergenceV	
    call mpas_timer_start("Velocity solver stress divergence")
    call seaice_stress_divergence_variational(&
            meshPool, &
            nEdgesOnCell, &
            verticesOnCell, &
            stressDivergenceU, &
            stressDivergenceV, &
            stress11, &
            stress22, &
            stress12, &
            basisIntegralsU, &
            basisIntegralsV, &
            basisIntegralsMetric, &
            tanLatVertexRotatedOverRadius, &
            cellVerticesAtVertex, &
            cellsOnVertex, &
            areaTriangle, &
            solveVelocity)
    call mpas_timer_stop("Velocity solver stress divergence")
	
  end subroutine seaice_internal_stress_variational!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_strain_tensor_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_stress_strain_tensor_variational(&
  	   proc_id, &
       mesh, &
       nEdgesOnCell, &
       verticesOnCell, &
       stress11, &
       stress22, &
       stress12, &
       strain11, &
       strain22, &
       strain12, &
       uVelocity, &
       vVelocity, &
       basisGradientU, &
       basisGradientV, &
       tanLatVertexRotatedOverRadius, &
       icePressure, &
       replacementPressure, &
       areaCell, &
       solveStress, &
       dtElastic, &
       revisedEVP)!{{{

    use seaice_velocity_solver_constitutive_relation, only: &
    	 eccentricitySquared, &
         seaice_evp_constitutive_relation, &
         seaice_evp_constitutive_relation_revised

    integer, intent(in) :: proc_id
    
    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), intent(inout) :: &
         stress11, & !< Input/Output:
         stress22, & !< Input/Output:
         stress12    !< Input/Output:

    real(kind=RKIND), dimension(:,:), intent(out) :: &
         strain11, & !< Output:
         strain22, & !< Output:
         strain12    !< Output:

    real(kind=RKIND), dimension(:), intent(in) :: &
         uVelocity, & !< Input:
         vVelocity, & !< Input:
         tanLatVertexRotatedOverRadius !< Input:

    real(kind=RKIND), dimension(:,:,:), intent(in) :: &
         basisGradientU, & !< Input:
         basisGradientV    !< Input:

    real(kind=RKIND), dimension(:,:), intent(out) :: &
         replacementPressure !< Output:

    real(kind=RKIND), dimension(:), intent(in) :: &
         areaCell, &
         icePressure !< Input:

    real(kind=RKIND), intent(in) :: &
         dtElastic !< Input:

    logical, intent(in) :: &
         revisedEVP !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress !< Input:

    integer, dimension(:), intent(in) :: &
         nEdgesOnCell

    integer, dimension(:,:), intent(in) :: &
         verticesOnCell

    integer :: &
         iCell, &
         iGradientVertex, &
         iBasisVertex, &
         iVertex, &
         jVertex

    integer, pointer :: &
         nCellsPtr

    integer :: &
         nCells

    ! init variables
    call MPAS_pool_get_dimension(mesh, "nCells", nCellsPtr)
	nCells = nCellsPtr
	        
    ! loop over cells
#ifdef MPAS_OPENACC
	!$acc parallel loop present(solveStress, uVelocity, vVelocity, &
    !$acc                   icePressure, &
    !$acc                   strain11, strain22, strain12, stress11, stress22, stress12, &
    !$acc                   tanLatVertexRotatedOverRadius, &
    !$acc                   basisGradientU, basisGradientV, &
    !$acc                   replacementPressure, &
    !$acc                   nEdgesOnCell, verticesOnCell, areaCell) &
    !$acc               private(iVertex, jVertex) &
    !$acc async(1)
#else    
    !$omp parallel do default(shared) private(iCell, iGradientVertex, iBasisVertex, iVertex, jVertex)
#endif    
    do iCell = 1, nCells

       do iGradientVertex = 1, nEdgesOnCell(iCell)
          replacementPressure(iGradientVertex,iCell) = 0.0_RKIND
       enddo
       
       if (solveStress(iCell) == 1) then

          do iGradientVertex = 1, nEdgesOnCell(iCell)
             strain11(iGradientVertex,iCell) = 0.0_RKIND
             strain22(iGradientVertex,iCell) = 0.0_RKIND
             strain12(iGradientVertex,iCell) = 0.0_RKIND
          enddo

		  !if ( maxval(nEdgesOnCell) > maxv ) print *,'error: nEdgesOnCell overwritten'
		  
          ! loop over velocity points surrounding cell - location of stress and derivative
          do iGradientVertex = 1, nEdgesOnCell(iCell)

             ! loop over basis functions
             do iBasisVertex = 1, nEdgesOnCell(iCell)

                iVertex = verticesOnCell(iBasisVertex,iCell)

                strain11(iGradientVertex,iCell) = strain11(iGradientVertex,iCell) + &
                     uVelocity(iVertex) * basisGradientU(iBasisVertex,iGradientVertex,iCell)

                strain22(iGradientVertex,iCell) = strain22(iGradientVertex,iCell) + &
                     vVelocity(iVertex) * basisGradientV(iBasisVertex,iGradientVertex,iCell)

                strain12(iGradientVertex,iCell) = strain12(iGradientVertex,iCell) + 0.5_RKIND * (&
                     uVelocity(iVertex) * basisGradientV(iBasisVertex,iGradientVertex,iCell) + &
                     vVelocity(iVertex) * basisGradientU(iBasisVertex,iGradientVertex,iCell))

             enddo ! iVertexOnCell

             ! metric terms
             jVertex = verticesOnCell(iGradientVertex,iCell)

             strain11(iGradientVertex,iCell) = strain11(iGradientVertex,iCell) - &
                  vVelocity(jVertex) * tanLatVertexRotatedOverRadius(jVertex)

             strain12(iGradientVertex,iCell) = strain12(iGradientVertex,iCell) + &
                  uVelocity(jVertex) * tanLatVertexRotatedOverRadius(jVertex) * 0.5_RKIND

		     !if ( .not. associated(evpPointer) ) print *,'Error: evpPointer not set'
             call seaice_evp_constitutive_relation(&
                     stress11(iGradientVertex,iCell), &
                     stress22(iGradientVertex,iCell), &
                     stress12(iGradientVertex,iCell), &
                     strain11(iGradientVertex,iCell), &
                     strain22(iGradientVertex,iCell), &
                     strain12(iGradientVertex,iCell), &
                     icePressure(iCell), &
                     replacementPressure(iGradientVertex,iCell), &
                     areaCell(iCell), &
                     dtElastic)

          enddo ! jVertexOnCell

       endif ! solveStress

    enddo ! iCell
		
  end subroutine seaice_stress_strain_tensor_variational!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_strain_tensor_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_strain_tensor_variational(&
       mesh, &
       strain11, &
       strain22, &
       strain12, &
       uVelocity, &
       vVelocity, &
       basisGradientU, &
       basisGradientV, &
       tanLatVertexRotatedOverRadius, &
       solveStress)!{{{

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), intent(out) :: &
         strain11, & !< Output:
         strain22, & !< Output:
         strain12    !< Output:

    real(kind=RKIND), dimension(:), intent(in) :: &
         uVelocity, & !< Input:
         vVelocity, & !< Input:
         tanLatVertexRotatedOverRadius !< Input:

    real(kind=RKIND), dimension(:,:,:), intent(in) :: &
         basisGradientU, & !< Input:
         basisGradientV    !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress !< Input:

    integer :: &
         iCell, &
         iGradientVertex, &
         iBasisVertex, &
         iVertex, &
         jVertex

    integer, pointer :: &
         nCells

    integer, dimension(:), pointer :: &
         nEdgesOnCell

    integer, dimension(:,:), pointer :: &
         verticesOnCell

    ! init variables
    call MPAS_pool_get_dimension(mesh, "nCells", nCells)

    call MPAS_pool_get_array(mesh, "nEdgesOnCell", nEdgesOnCell)
    call MPAS_pool_get_array(mesh, "verticesOnCell", verticesOnCell)

    ! loop over cells
    !!$omp parallel do default(shared) private(iCell, iGradientVertex, iBasisVertex, iVertex, jVertex)
    do iCell = 1, nCells

       if (solveStress(iCell) == 1) then

          strain11(:,iCell) = 0.0_RKIND
          strain22(:,iCell) = 0.0_RKIND
          strain12(:,iCell) = 0.0_RKIND

          ! loop over velocity points surrounding cell - location of stress and derivative
          do iGradientVertex = 1, nEdgesOnCell(iCell)

             ! loop over basis functions
             do iBasisVertex = 1, nEdgesOnCell(iCell)

                iVertex = verticesOnCell(iBasisVertex,iCell)

                strain11(iGradientVertex,iCell) = strain11(iGradientVertex,iCell) + &
                     uVelocity(iVertex) * basisGradientU(iBasisVertex,iGradientVertex,iCell)

                strain22(iGradientVertex,iCell) = strain22(iGradientVertex,iCell) + &
                     vVelocity(iVertex) * basisGradientV(iBasisVertex,iGradientVertex,iCell)

                strain12(iGradientVertex,iCell) = strain12(iGradientVertex,iCell) + 0.5_RKIND * (&
                     uVelocity(iVertex) * basisGradientV(iBasisVertex,iGradientVertex,iCell) + &
                     vVelocity(iVertex) * basisGradientU(iBasisVertex,iGradientVertex,iCell))

             enddo ! iVertexOnCell

             ! metric terms
             jVertex = verticesOnCell(iGradientVertex,iCell)

             strain11(iGradientVertex,iCell) = strain11(iGradientVertex,iCell) - &
                  vVelocity(jVertex) * tanLatVertexRotatedOverRadius(jVertex)

             strain12(iGradientVertex,iCell) = strain12(iGradientVertex,iCell) + &
                  uVelocity(jVertex) * tanLatVertexRotatedOverRadius(jVertex) * 0.5_RKIND

          enddo ! jVertexOnCell

       endif ! solveStress

    enddo ! iCell

  end subroutine seaice_strain_tensor_variational!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_tensor_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_stress_tensor_variational(&
       mesh, &
       stress11, &
       stress22, &
       stress12, &
       strain11, &
       strain22, &
       strain12, &
       icePressure, &
       replacementPressure, &
       solveStress, &
       dtElastic, &
       revisedEVP)!{{{

    use seaice_velocity_solver_constitutive_relation, only: &
         seaice_evp_constitutive_relation, &
         seaice_evp_constitutive_relation_revised

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:,:), intent(inout) :: &
         stress11, & !< Input/Output:
         stress22, & !< Input/Output:
         stress12    !< Input/Output:

    real(kind=RKIND), dimension(:,:), intent(out) :: &
         replacementPressure !< Output:

    real(kind=RKIND), dimension(:,:), intent(in) :: &
         strain11, & !< Input:
         strain22, & !< Input:
         strain12    !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         icePressure !< Input:

    integer, dimension(:), intent(in) :: &
         solveStress !< Input:

    real(kind=RKIND), intent(in) :: &
         dtElastic !< Input:

    logical, intent(in) :: &
         revisedEVP !< Input:

    integer :: &
         iCell, &
         iVertexOnCell

    integer, pointer :: &
         nCells, &
         maxEdges

    integer, dimension(:), pointer :: &
         nEdgesOnCell

    real(kind=RKIND), dimension(:), pointer :: &
         areaCell

    ! init variables
    call MPAS_pool_get_dimension(mesh, "nCells", nCells)
    call MPAS_pool_get_dimension(mesh, "maxEdges", maxEdges)

    call MPAS_pool_get_array(mesh, "nEdgesOnCell", nEdgesOnCell)
    call MPAS_pool_get_array(mesh, "areaCell", areaCell)

    if (.not. revisedEVP) then

       !$omp parallel do default(shared) private(iCell, iVertexOnCell)
       do iCell = 1, nCells

          replacementPressure(:,iCell) = 0.0_RKIND

          if (solveStress(iCell) == 1) then

             do iVertexOnCell = 1, nEdgesOnCell(iCell)

                call seaice_evp_constitutive_relation(&
                     stress11(iVertexOnCell,iCell), &
                     stress22(iVertexOnCell,iCell), &
                     stress12(iVertexOnCell,iCell), &
                     strain11(iVertexOnCell,iCell), &
                     strain22(iVertexOnCell,iCell), &
                     strain12(iVertexOnCell,iCell), &
                     icePressure(iCell), &
                     replacementPressure(iVertexOnCell,iCell), &
                     areaCell(iCell), &
                     dtElastic)

             enddo ! iVertexOnCell

          endif ! solveStress

       enddo ! iCell

    else

       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             do iVertexOnCell = 1, nEdgesOnCell(iCell)

                call seaice_evp_constitutive_relation_revised(&
                     stress11(iVertexOnCell,iCell), &
                     stress22(iVertexOnCell,iCell), &
                     stress12(iVertexOnCell,iCell), &
                     strain11(iVertexOnCell,iCell), &
                     strain22(iVertexOnCell,iCell), &
                     strain12(iVertexOnCell,iCell), &
                     icePressure(iCell), &
                     replacementPressure(iVertexOnCell,iCell), &
                     areaCell(iCell))

             enddo ! iVertexOnCell

          endif ! solveStress

       enddo ! iCell

    endif

  end subroutine seaice_stress_tensor_variational!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  seaice_stress_divergence_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date 2013-2014
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_stress_divergence_variational(&
       mesh, &
       nEdgesOnCell, &
       verticesOnCell, &
       stressDivergenceU, &
       stressDivergenceV, &
       stress11, &
       stress22, &
       stress12, &
       basisIntegralsU, &
       basisIntegralsV, &
       basisIntegralsMetric, &
       tanLatVertexRotatedOverRadius, &
       cellVerticesAtVertex, &
       cellsOnVertex, &
       areaTriangle, &
       solveVelocity)!{{{

    type(MPAS_pool_type), pointer, intent(in) :: &
         mesh !< Input:

    real(kind=RKIND), dimension(:), intent(out) :: &
         stressDivergenceU, & !< Output:
         stressDivergenceV    !< Output:

    real(kind=RKIND), dimension(:,:), intent(in) :: &
         stress11, & !< Input:
         stress22, & !< Input:
         stress12    !< Input:

    real(kind=RKIND), dimension(:,:,:), intent(in) :: &
         basisIntegralsU, &   !< Input:
         basisIntegralsV, &   !< Input:
         basisIntegralsMetric !< Input:

    real(kind=RKIND), dimension(:), intent(in) :: &
         tanLatVertexRotatedOverRadius !< Input:

    integer, dimension(:,:), intent(in) :: &
         cellVerticesAtVertex !< Input:

    integer, dimension(:), intent(in) :: &
         solveVelocity !< Input:

    integer, dimension(:), intent(in) :: &
         nEdgesOnCell

    integer, dimension(:,:), intent(in) :: &
         verticesOnCell

    real(kind=RKIND) :: &
         stressDivergenceUCell, &
         stressDivergenceVCell

    integer :: &
         iVertex, &
         iSurroundingCell, &
         iCell, &
         iStressVertex, &
         iVelocityVertex

    integer, pointer :: &
         nVerticesPtr, &
         vertexDegreePtr

    integer :: &
         nVertices, &
         vertexDegree

    integer, dimension(:,:), intent(in), pointer :: &
         cellsOnVertex

    real(kind=RKIND), dimension(:), intent(in), pointer :: &
         areaTriangle

    ! init variables
    call MPAS_pool_get_dimension(mesh, "nVertices", nVerticesPtr)
    nVertices = nVerticesPtr
    
    call MPAS_pool_get_dimension(mesh, "vertexDegree", vertexDegreePtr)
	vertexDegree = vertexDegreePtr
	
    ! loop over velocity positions
#ifdef MPAS_OPENACC
	!$acc parallel loop present(solveVelocity, stressDivergenceU, stressDivergenceV, &
	!$acc                       cellsOnVertex, cellVerticesAtVertex, nEdgesOnCell, tanLatVertexRotatedOverRadius, &
	!$acc                       basisIntegralsU, basisIntegralsV, basisIntegralsMetric, &
	!$acc                       areaTriangle, stress11, stress22, stress12 ) &
	!$acc               private(iCell,iVelocityVertex,stressDivergenceUCell,stressDivergenceVCell) &
	!$acc async(1)
#else    
    !$omp parallel do default(shared) private(iVertex, iSurroundingCell, iCell, iVelocityVertex, stressDivergenceUCell, stressDivergenceVCell, iStressVertex)
#endif    
    do iVertex = 1, nVertices

       if (solveVelocity(iVertex) == 1) then

          stressDivergenceU(iVertex) = 0.0_RKIND
          stressDivergenceV(iVertex) = 0.0_RKIND

          ! loop over surrounding cells
          do iSurroundingCell = 1, vertexDegree

             ! get the cell number of this cell
             iCell = cellsOnVertex(iSurroundingCell, iVertex)

             ! get the vertexOnCell number of the iVertex velocity point from cell iCell
             iVelocityVertex = cellVerticesAtVertex(iSurroundingCell,iVertex)

             stressDivergenceUCell = 0.0_RKIND
             stressDivergenceVCell = 0.0_RKIND

             ! loop over the vertices of the surrounding cell
             do iStressVertex = 1, nEdgesOnCell(iCell)

                ! normal terms
                stressDivergenceUCell = stressDivergenceUCell - &
                     stress11(iStressVertex,iCell) * basisIntegralsU(iStressVertex,iVelocityVertex,iCell) - &
                     stress12(iStressVertex,iCell) * basisIntegralsV(iStressVertex,iVelocityVertex,iCell)

                stressDivergenceVCell = stressDivergenceVCell - &
                     stress22(iStressVertex,iCell) * basisIntegralsV(iStressVertex,iVelocityVertex,iCell) - &
                     stress12(iStressVertex,iCell) * basisIntegralsU(iStressVertex,iVelocityVertex,iCell)

                ! metric terms
                stressDivergenceUCell = stressDivergenceUCell - &
                     stress12(iStressVertex,iCell) * basisIntegralsMetric(iStressVertex,iVelocityVertex,iCell) * &
                     tanLatVertexRotatedOverRadius(iVertex)

                stressDivergenceVCell = stressDivergenceVCell + &
                     stress11(iStressVertex,iCell) * basisIntegralsMetric(iStressVertex,iVelocityVertex,iCell) * &
                     tanLatVertexRotatedOverRadius(iVertex)

             enddo ! iStressVertex

             stressDivergenceU(iVertex) = stressDivergenceU(iVertex) + stressDivergenceUCell
             stressDivergenceV(iVertex) = stressDivergenceV(iVertex) + stressDivergenceVCell

          enddo ! iSurroundingCell

          stressDivergenceU(iVertex) = stressDivergenceU(iVertex) / areaTriangle(iVertex)
          stressDivergenceV(iVertex) = stressDivergenceV(iVertex) / areaTriangle(iVertex)

       endif ! solveVelocity

    enddo ! iVertex
	
  end subroutine seaice_stress_divergence_variational!}}}

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  final_divergence_shear_variational
!
!> \brief
!> \author Adrian K. Turner, LANL
!> \date July 9th 2015
!> \details
!>
!
!-----------------------------------------------------------------------

  subroutine seaice_final_divergence_shear_variational(block)

    use seaice_velocity_solver_constitutive_relation, only: &
         eccentricitySquared

    type(block_type), intent(inout) :: &
         block

    type(MPAS_pool_type), pointer :: &
         meshPool, &
         velocityVariationalPool, &
         velocitySolverPool, &
         ridgingPool

    integer, pointer :: &
         nCells

    integer, dimension(:), pointer :: &
         nEdgesOnCell, &
         solveStress

    real(kind=RKIND), dimension(:,:), pointer :: &
         strain11, &
         strain22, &
         strain12

    real(kind=RKIND), dimension(:), pointer :: &
         divergence, &
         shear, &
         ridgeConvergence, &
         ridgeShear

    real(kind=RKIND), dimension(:), allocatable :: &
         DeltaAverage

    real(kind=RKIND) :: &
         strainDivergenceSum, &
         strainTensionSum, &
         strainShearingSum, &
         strainDivergence, &
         strainTension, &
         strainShearing, &
         Delta

    logical, pointer :: &
         config_use_column_package

    integer :: &
         iCell, &
         iVertexOnCell

    call MPAS_pool_get_subpool(block % structs, "mesh", meshPool)
    call MPAS_pool_get_subpool(block % structs, "velocity_variational", velocityVariationalPool)
    call MPAS_pool_get_subpool(block % structs, "velocity_solver", velocitySolverPool)

    call MPAS_pool_get_dimension(meshPool, "nCells", nCells)

    call MPAS_pool_get_array(meshPool, "nEdgesOnCell", nEdgesOnCell)

    call MPAS_pool_get_array(velocityVariationalPool, "strain11", strain11)
    call MPAS_pool_get_array(velocityVariationalPool, "strain22", strain22)
    call MPAS_pool_get_array(velocityVariationalPool, "strain12", strain12)

    call MPAS_pool_get_array(velocitySolverPool, "divergence", divergence)
    call MPAS_pool_get_array(velocitySolverPool, "shear", shear)
    call MPAS_pool_get_array(velocitySolverPool, "solveStress", solveStress)

    allocate(DeltaAverage(nCells))

    do iCell = 1, nCells

       if (solveStress(iCell) == 1) then

          strainDivergenceSum = 0.0_RKIND
          strainTensionSum    = 0.0_RKIND
          strainShearingSum   = 0.0_RKIND
          DeltaAverage(iCell) = 0.0_RKIND

          do iVertexOnCell = 1, nEdgesOnCell(iCell)

             strainDivergence = strain11(iVertexOnCell,iCell) + strain22(iVertexOnCell,iCell)
             strainTension    = strain11(iVertexOnCell,iCell) - strain22(iVertexOnCell,iCell)
             strainShearing   = strain12(iVertexOnCell,iCell) * 2.0_RKIND

             Delta = sqrt(strainDivergence**2 + (strainTension**2 + strainShearing**2) / eccentricitySquared)

             strainDivergenceSum = strainDivergenceSum + strainDivergence
             strainTensionSum    = strainTensionSum    + strainTension
             strainShearingSum   = strainShearingSum   + strainShearing
             DeltaAverage(iCell) = DeltaAverage(iCell) + Delta

          enddo ! iVertexOnCell

          divergence(iCell)   = strainDivergenceSum                              / real(nEdgesOnCell(iCell),RKIND)
          shear(iCell)        = sqrt(strainTensionSum**2 + strainShearingSum**2) / real(nEdgesOnCell(iCell),RKIND)
          DeltaAverage(iCell) = DeltaAverage(iCell)                              / real(nEdgesOnCell(iCell),RKIND)

       else

          divergence(iCell)   = 0.0_RKIND
          shear(iCell)        = 0.0_RKIND

       endif

    enddo ! iCell

    ! ridging parameters
    call MPAS_pool_get_config(block % configs, "config_use_column_package", config_use_column_package)

    if (config_use_column_package) then

       call MPAS_pool_get_subpool(block % structs, "ridging", ridgingPool)

       call MPAS_pool_get_array(ridgingPool, "ridgeConvergence", ridgeConvergence)
       call MPAS_pool_get_array(ridgingPool, "ridgeShear", ridgeShear)

       do iCell = 1, nCells

          if (solveStress(iCell) == 1) then

             ridgeConvergence(iCell) = -min(divergence(iCell),0.0_RKIND)
             ridgeShear(iCell)       = 0.5_RKIND * (DeltaAverage(iCell) - abs(divergence(iCell)))

          else

             ridgeConvergence(iCell) = 0.0_RKIND
             ridgeShear(iCell)       = 0.0_RKIND

          endif

       enddo ! iCell

    endif ! config_use_column_package

    ! units - for comparison to CICE
    divergence = divergence * 100.0_RKIND * 86400.0_RKIND
    shear      = shear      * 100.0_RKIND * 86400.0_RKIND

    ! cleanup
    deallocate(DeltaAverage)

  end subroutine seaice_final_divergence_shear_variational

!-----------------------------------------------------------------------

end module seaice_velocity_solver_variational
