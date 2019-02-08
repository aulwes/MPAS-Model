module ocn_gpu_support

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants

   implicit none
   private

  type mesh2id
      type (mpas_pool_type), pointer :: meshPool
	  integer :: blockID     
  end type mesh2id
  
  type block2id
      type (block_type), pointer :: block
	  integer :: blockID     
  end type block2id
  
  integer :: g_nblocks
  type(mesh2id), dimension(:), allocatable :: mesh_map
  type(block2id), dimension(:), allocatable :: block_map
    
  ! mem block format: (..., field, block)

  ! 1D int arrays
  ! nEdgesOnEdge
  integer, dimension(:,:,:), allocatable, target :: &
         i1dmem_block

  ! 1D int arrays for cells
  ! nEdgesOnCell
  integer, dimension(:,:,:), allocatable, target :: &
         i1dcell_block

  ! 1D real arrays
  ! fEdge, dcEdge
  real(kind=RKIND), dimension(:,:,:), allocatable, target :: &
         r1dmem_block

  ! 2D int arrays
  ! edgesOnEdge
  integer, dimension(:,:,:,:), allocatable, target :: &
         i2dmem_block

  ! 2D int arrays for cells
  ! edgesOnEdge
  integer, dimension(:,:,:,:), allocatable, target :: &
         i2dcell_block

  ! 2D real arrays
  ! weightsOnEdge
  real(kind=RKIND), dimension(:,:,:,:), allocatable, target :: &
         r2dmem_block

  ! (2, nedges, block)
  integer, dimension(:,:,:), allocatable, target :: &
         coe_device ! cellsOnEdge

  ! 1D int arrays
  integer, parameter :: &
  		neoep = 1,  & ! nEdgesOnEdge
  		endi1dp = 2
  		
  ! 1D int arrays for cells
  integer, parameter :: &
  		neocp = 1,  & ! nEdgesOnCell
  		endi1dcellp = 2
  		
  ! 2D int arrays
  integer, parameter :: &
  		eoep = 1, & ! edgesOnEdge
  		endi2dp = 2
  		
  ! 2D int arrays for cells
  integer, parameter :: &
  		cocp = 1, & ! cellsOnCells
  		eocp = 2, & ! edgesOnCell
  		endi2dcellp = 3
  		
  ! 1D real arrays
  integer, parameter :: &
  		fedgep = 1,  & ! fEdge
  		dcedgep = 2, & ! dcEdge
  		endr1dp = 3

  ! 2D real arrays
  integer, parameter :: &
  		woep = 1,  & ! weightsOnEdge
  		endr2dp = 2

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

   interface ocn_blockid
      module procedure ocn_blockid_mesh
      module procedure ocn_blockid_block
   end interface
   
   public :: &
   			ocn_gpu_init, &
   			ocn_gpu_get_cellsOnEdge, ocn_gpu_get_cellsOnCell, &
   			ocn_gpu_get_nEdgesOnCell, ocn_gpu_get_edgesOnCell, &
   			ocn_gpu_get_nEdgesOnEdge, ocn_gpu_get_edgesOnEdge, &
   			ocn_gpu_get_weightsOnEdge, ocn_gpu_get_fEdge, &
   			ocn_gpu_get_dcEdge

contains

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_init
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_init(domain)!{{{

      type(domain_type), intent(inout) :: &
         domain

      type(block_type), pointer :: &
         block

      type (mpas_pool_type), pointer :: meshPool

	  integer :: max_nedges, max_neoe, iblk, max_cells, max_neoc
      integer, dimension(:), pointer :: nEdgesArray, nCellsArray, nEdgesOnEdge, &
      									nEdgesOnCell
      integer, dimension(:,:), pointer :: edgesOnEdge, cellsOnEdge, cellsOnCell, &
      									edgesOnCell
      real (kind=RKIND), dimension(:), pointer :: fEdge, dcEdge
      real (kind=RKIND), dimension(:,:), pointer :: weightsOnEdge

 	  g_nblocks = 0
 	  max_nedges = 0
 	  max_cells = 0
 	  max_neoe = 0
 	  max_neoc = 0
	  block => domain % blocklist
      do while (associated(block))
         g_nblocks = g_nblocks + 1
         call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
         call mpas_pool_get_dimension(block % dimensions, 'nEdgesArray', nEdgesArray)
         call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
         call mpas_pool_get_array(meshPool, 'nEdgesOnEdge', nEdgesOnEdge)
         call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
         max_nedges = max(max_nedges, maxval(nEdgesArray))
         max_cells = max(max_cells, maxval(nCellsArray))
         max_neoe = max(max_neoe, maxval(nEdgesOnEdge))         
         max_neoc = max(max_neoc, maxval(nEdgesOnCell))         
         block => block % next
      end do
 	  
 	  if ( g_nblocks > 0 ) then
 	    print *,' # blocks = ', g_nblocks
 	    ! verify g_nblocks
 	    !if ( g_nblocks .ne. size() ) then
 	    !   print *,'Error: expected ', size(), ' nblocks, but found ',g_nblocks
 	    !endif
 	  	allocate( i1dmem_block(max_nedges, endi1dp, g_nblocks), &
 	  			  i1dcell_block(max_cells, endi1dcellp, g_nblocks), &
 	  			  r1dmem_block(max_nedges, endr1dp, g_nblocks) )
 	  	allocate( i2dmem_block(max_neoe, max_nedges, endi2dp, g_nblocks), &
 	  			  coe_device(2, max_nedges, g_nblocks), &
 	  			  i2dcell_block(max_neoc, max_cells, endi2dcellp, g_nblocks), &
 	  			  r2dmem_block(max_neoe, max_nedges, endr2dp, g_nblocks) )
	    allocate( mesh_map(g_nblocks), block_map(g_nblocks) )
	    
	    block => domain % blocklist
	    iblk = 1
        do while ( associated(block) )
           iblk = block%localBlockID + 1
           call mpas_pool_get_subpool(block % structs, 'mesh', meshPool)
           mesh_map(iblk)%meshPool => meshPool
           mesh_map(iblk)%blockID = iblk
           block_map(iblk)%block => block
           block_map(iblk)%blockID = iblk
           
           call mpas_pool_get_array(meshPool, 'nEdgesOnEdge', nEdgesOnEdge)
           call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
		   call mpas_pool_get_array(meshPool, 'edgesOnEdge', edgesOnEdge)
		   call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
           call mpas_pool_get_array(meshPool, 'cellsOnCell', cellsOnCell)
		   call mpas_pool_get_array(meshPool, 'weightsOnEdge', weightsOnEdge)
		   call mpas_pool_get_array(meshPool, 'fEdge', fEdge)
		   call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
           call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
           
           i1dmem_block(:, neoep, iblk) = nEdgesOnEdge
           i1dcell_block(:, neocp, iblk) = nEdgesOnCell
           
           r1dmem_block(:, fedgep, iblk) = fEdge
           r1dmem_block(:, dcedgep, iblk) = dcEdge

           i2dmem_block(:,:, eoep, iblk) = edgesOnEdge
           r2dmem_block(:,:, woep, iblk) = weightsOnEdge
           i2dcell_block(:,:,cocp, iblk) = cellsOnCell
           i2dcell_block(:,:,eocp, iblk) = edgesOnCell
           
           coe_device(:,:,iblk) = cellsOnEdge
           
           block => block % next
    	   !iblk = iblk + 1       
        end do

        !$acc enter data copyin(i1dmem_block, r1dmem_block, i2dmem_block, &
        !$acc                   i1dcell_block, i2dcell_block, r2dmem_block, coe_device)
 	  	
 	  endif
 	  
   end subroutine ocn_gpu_init

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_blockid
!
!
!-----------------------------------------------------------------------
	integer function ocn_blockid_mesh(mesh)
	   integer :: ocn_blockid
       type (mpas_pool_type), pointer :: mesh
	   
	   integer :: iblk, i
	   
	   iblk = 0
	   do i = 1, g_nblocks
	      if ( associated(mesh_map(i)%meshPool, mesh) ) then
	         iblk = i	         
	      endif
	   end do
	   
	   ocn_blockid_mesh = iblk
	   
	end function ocn_blockid_mesh


	integer function ocn_blockid_block(block)
	   integer :: ocn_blockid

       type (block_type), pointer :: block
	   
	   integer :: iblk, i
	   
	   iblk = 0
	   do i = 1, g_nblocks
	      if ( associated(block_map(i)%block, block) ) then
	         iblk = i	         
	      endif
	   end do
	   
	   ocn_blockid_block = iblk
	   
	   
	end function ocn_blockid_block
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_cellsOnEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_cellsOnEdge(iblk, cellsOnEdge)!{{{

	  integer, intent(in) :: iblk
      integer, dimension(:,:), pointer :: cellsOnEdge
   
      cellsOnEdge => coe_device(:,:,iblk)
      
   end subroutine ocn_gpu_get_cellsOnEdge 

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_edgesOnCell
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_edgesOnCell(iblk, edgesOnCell)!{{{

	  integer, intent(in) :: iblk
      integer, dimension(:,:), pointer :: edgesOnCell
   
      edgesOnCell => i2dcell_block(:,:,eocp,iblk)
      
   end subroutine ocn_gpu_get_edgesOnCell 

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_cellsOnCell
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_cellsOnCell(iblk, cellsOnCell)!{{{

	  integer, intent(in) :: iblk
      integer, dimension(:,:), pointer :: cellsOnCell
   
      cellsOnCell => i2dcell_block(:,:,cocp,iblk)
      
   end subroutine ocn_gpu_get_cellsOnCell

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_dcEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_dcEdge(iblk, dcEdge)!{{{

	  integer, intent(in) :: iblk
      real (kind=RKIND), dimension(:), pointer :: dcEdge
   
      dcEdge => r1dmem_block(:,dcedgep,iblk)
      
   end subroutine ocn_gpu_get_dcEdge

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_fEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_fEdge(iblk, fEdge)!{{{

	  integer, intent(in) :: iblk
      real (kind=RKIND), dimension(:), pointer :: fEdge
   
      fEdge => r1dmem_block(:,fedgep,iblk)
      
   end subroutine ocn_gpu_get_fEdge

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_cellsOnEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_nEdgesOnCell(iblk, nEdgesOnCell)!{{{

	  integer, intent(in) :: iblk
      integer, dimension(:), pointer :: nEdgesOnCell
   
      nEdgesOnCell => i1dcell_block(:,neocp,iblk)
      
   end subroutine ocn_gpu_get_nEdgesOnCell

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_weightsOnEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_weightsOnEdge(iblk, weightsOnEdge)!{{{

	  integer, intent(in) :: iblk
      real (kind=RKIND), dimension(:,:), pointer :: weightsOnEdge
   
      weightsOnEdge => r2dmem_block(:,:,woep,iblk)
      
   end subroutine ocn_gpu_get_weightsOnEdge

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_cellsOnEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_edgesOnEdge(iblk, edgesOnEdge)!{{{

	  integer, intent(in) :: iblk
      integer, dimension(:,:), pointer :: edgesOnEdge
   
      edgesOnEdge => i2dmem_block(:,:,eoep,iblk)
      
   end subroutine ocn_gpu_get_edgesOnEdge 

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_gpu_get_nEdgesOnEdge
!
!
!-----------------------------------------------------------------------
   subroutine ocn_gpu_get_nEdgesOnEdge(iblk, nEdgesOnEdge)!{{{

	  integer, intent(in) :: iblk
      integer, dimension(:), pointer :: nEdgesOnEdge
   
      nEdgesOnEdge => i1dmem_block(:,neoep,iblk)
      
   end subroutine ocn_gpu_get_nEdgesOnEdge 

end module ocn_gpu_support
