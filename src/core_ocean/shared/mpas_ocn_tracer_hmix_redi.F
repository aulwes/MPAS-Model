! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_tracer_hmix_redi
!
!> \brief MPAS ocean horizontal tracer mixing driver
!> \author Doug Jacobsen, Mark Petersen, Todd Ringler
!> \date   September 2011
!> \details
!>  This module contains the main driver routine for computing
!>  horizontal mixing tendencies.
!>
!>  It provides an init and a tend function. Each are described below.
!
!-----------------------------------------------------------------------

module ocn_tracer_hmix_redi

   use mpas_timer
   use mpas_derived_types
   use mpas_pool_routines
   use mpas_threading

   use ocn_constants
#ifdef MPAS_OPENACC
   use ocn_gpu_support
#endif

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

   public :: ocn_tracer_hmix_redi_tend, &
             ocn_tracer_hmix_redi_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

   logical :: rediOn
   logical, pointer :: config_disable_redi_horizontal_term1
   logical, pointer :: config_disable_redi_horizontal_term2
   logical, pointer :: config_disable_redi_horizontal_term3
   real (kind=RKIND), pointer :: config_Redi_kappaPtr
   real (kind=RKIND) :: config_Redi_kappa


!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_tracer_hmix_redi_tend
!
!> \brief   Computes Laplacian tendency term for horizontal tracer mixing
!> \author  Doug Jacobsen, Mark Petersen, Todd Ringler
!> \date    September 2011
!> \details
!>  This routine computes the horizontal mixing tendency for tracers
!>  based on current state using a Laplacian parameterization.
!
!-----------------------------------------------------------------------

   subroutine ocn_tracer_hmix_redi_tend(meshPool, scratchPool, layerThicknessEdge, zMid, tracers, &
                                        relativeSlopeTopOfEdge, relativeSlopeTapering, relativeSlopeTaperingCell, tend, err)!{{{


      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in), pointer :: meshPool !< Input: Mesh information
      type (mpas_pool_type), intent(in) :: scratchPool !< Input: Scratch information

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         layerThicknessEdge, &!< Input: thickness at edge
         zMid,               &!< Input: Z coordinate at the center of a cell
         relativeSlopeTopOfEdge,    &!< Input: slope of coordinate relative to neutral surface at edges
         relativeSlopeTapering,     &!< Input: tapering of slope of coordinate relative to neutral surface at edges
         relativeSlopeTaperingCell   !< Input: tapering of slope of coordinate relative to neutral surface at cells

      real (kind=RKIND), dimension(:,:,:), intent(in) :: &
        tracers !< Input: tracer quantities

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:,:), intent(inout) :: &
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

      integer :: iCell, iEdge, cell1, cell2, iblk, mxlvlEdge, mxlvlCell, mxlvl
      integer :: i, k, iTracer, num_tracers, nCells, nEdges
      integer, pointer :: nVertLevels
      integer, dimension(:), pointer :: nCellsArray, nEdgesArray

      integer, dimension(:,:), allocatable :: boundaryMask

      integer, dimension(:), pointer :: maxLevelEdgeTop, nEdgesOnCell, maxLevelCell
      integer, dimension(:,:), pointer :: cellsOnEdge, edgesOnCell, edgeSignOnCell

      real (kind=RKIND) :: invAreaCell1, invAreaCell2, invAreaCell, areaEdge
      real (kind=RKIND) :: tracer_turb_flux, flux, s_tmp, r_tmp, h1, h2, s_tmpU, s_tmpD

      real (kind=RKIND), dimension(:), pointer :: areaCell, dvEdge, dcEdge

      real (kind=RKIND), dimension(:,:), pointer :: gradTracerEdge, gradTracerTopOfEdge, gradHTracerSlopedTopOfCell, &
         dTracerdZTopOfCell, dTracerdZTopOfEdge, areaCellSum

      type (field2DReal), pointer :: gradTracerEdgeField, gradTracerTopOfEdgeField, gradHTracerSlopedTopOfCellField, &
                                     dTracerdZTopOfCellField, dTracerdZTopOfEdgeField, areaCellSumField

      err = 0

      if (.not.rediOn) return

      call mpas_timer_start("tracer redi")

      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
      num_tracers = size(tracers, dim=1)

      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)

      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)

#ifdef MPAS_OPENACC
	  iblk = ocn_blockid(meshPool)
	  call ocn_gpu_get_maxLevelEdgeTop(iblk, maxLevelEdgeTop)
	  call ocn_gpu_get_maxLevelCell(iblk, maxLevelCell)
      call ocn_gpu_get_cellsOnEdge(iblk, cellsOnEdge)
	  call ocn_gpu_get_areaCell(iblk, areaCell)
      call ocn_gpu_get_dvEdge(iblk, dvEdge)
      call ocn_gpu_get_dcEdge(iblk, dcEdge)
	  call ocn_gpu_get_nEdgesOnCell(iblk,nEdgesOnCell)
	  call ocn_gpu_get_edgesOnCell(iblk,edgesOnCell)
      call ocn_gpu_get_edgeSignOnCell(iblk, edgeSignOnCell)
#else
      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)

      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgeSignOnCell', edgeSignOnCell)
#endif

      call mpas_pool_get_config(ocnConfigs, 'config_Redi_kappa',config_Redi_kappaPtr)
      config_Redi_kappa = config_Redi_kappaPtr
      
      call mpas_pool_get_config(ocnConfigs, 'config_disable_redi_horizontal_term1',config_disable_redi_horizontal_term1)
      call mpas_pool_get_config(ocnConfigs, 'config_disable_redi_horizontal_term2',config_disable_redi_horizontal_term2)
      call mpas_pool_get_config(ocnConfigs, 'config_disable_redi_horizontal_term3',config_disable_redi_horizontal_term3)

      !
      ! COMPUTE the extra terms arising due to mismatch between the constant coordinate surfaces and the
      ! isopycnal surfaces.
      !

      call mpas_pool_get_field(scratchPool, 'gradTracerEdge', gradTracerEdgeField)
      call mpas_pool_get_field(scratchPool, 'gradTracerTopOfEdge', gradTracerTopOfEdgeField)
      call mpas_pool_get_field(scratchPool, 'gradHTracerSlopedTopOfCell', gradHTracerSlopedTopOfCellField)
      call mpas_pool_get_field(scratchPool, 'dTracerdZTopOfCell', dTracerdZTopOfCellField)
      call mpas_pool_get_field(scratchPool, 'dTracerdZTopOfEdge', dTracerdZTopOfEdgeField)
      call mpas_pool_get_field(scratchPool, 'areaCellSum', areaCellSumField)

      call mpas_allocate_scratch_field(gradTracerEdgeField, .true., .false.)
      call mpas_allocate_scratch_field(gradTracerTopOfEdgeField, .true., .false.)
      call mpas_allocate_scratch_field(gradHTracerSlopedTopOfCellField, .true., .false.)
      call mpas_allocate_scratch_field(dTracerdZTopOfCellField, .true., .false.)
      call mpas_allocate_scratch_field(dTracerdZTopOfEdgeField, .true., .false.)
      call mpas_allocate_scratch_field(areaCellSumField, .true., .false.)
      call mpas_threading_barrier()

      gradTracerEdge => gradTracerEdgeField % array
      gradTracerTopOfEdge => gradTracerTopOfEdgeField % array
      gradHTracerSlopedTopOfCell => gradHTracerSlopedTopOfCellField % array
      dTracerdZTopOfCell => dTracerdZTopOfCellField % array
      dTracerdZTopOfEdge => dTracerdZTopOfEdgeField % array
      areaCellSum => areaCellSumField % array
#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc enter data copyin(tracers,tend,layerThicknessEdge,zMid,relativeSlopeTopOfEdge, &
	  !$acc            relativeSlopeTapering,relativeSlopeTaperingCell, &
	  !$acc            gradTracerEdge,gradTracerTopOfEdge,gradHTracerSlopedTopOfCell, &
	  !$acc            dTracerdZTopOfCell,dTracerdZTopOfEdge,areaCellSum) async(1)
	  !$omp end master
#endif
      nCells = nCellsArray( size(nCellsArray) )
      nEdges = nEdgesArray( size(nEdgesArray) )

	  mxlvlEdge = maxval(maxLevelEdgeTop)
	  mxlvlCell = maxval(maxLevelCell)
	  
#ifdef MPAS_OPENACC
	  !$omp master
      !$acc parallel loop collapse(2) &
      !$acc present(dTracerdZTopOfCell) &
      !$acc async(2) wait(1)
#else
      !$omp do schedule(runtime)
#endif      
      do iCell = 1, nCells
        do k = 1,mxlvlCell+1
           dTracerdZTopOfCell(k, iCell) = 0.0_RKIND
        end do
      end do
#ifdef MPAS_OPENACC
	  !$omp end master
#else
      !$omp end do
#endif      

#ifdef MPAS_OPENACC
	  !$omp master
      !$acc parallel loop collapse(2) &
      !$acc present(areaCellSum,gradHTracerSlopedTopOfCell,gradTracerEdge,gradTracerTopOfEdge,dTracerdZTopOfEdge) &
      !$acc async(1)
#else
      !$omp do schedule(runtime)
#endif      
      do iEdge = 1, nEdges
        do k = 1,mxlvlEdge+1
           areaCellSum(k, iCell) = 1.0e-34_RKIND
           gradHTracerSlopedTopOfCell(k, iCell) = 0.0_RKIND
           gradTracerEdge(k, iEdge) = 0.0_RKIND
           gradTracerTopOfEdge(k, iEdge) = 0.0_RKIND
           dTracerdZTopOfEdge(k, iEdge) = 0.0_RKIND
        end do
      end do
#ifdef MPAS_OPENACC
	  !$omp end master
#else
      !$omp end do
#endif      

      ! this is the "standard" del2 term, but forced to use config_redi_kappa
      if(.not.config_disable_redi_horizontal_term1) then

         nCells = nCellsArray( 1 )
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop &
         !$acc present(areaCell,nEdgesOnCell,edgesOnCell,cellsOnEdge,dvEdge,dcEdge, &
         !$acc         maxLevelEdgeTop,layerThicknessEdge,tracers,tend,edgeSignOnCell) &
         !$acc private(invAreaCell, i, iEdge, cell1, cell2, r_tmp, k, &
         !$acc         s_tmp, iTracer, tracer_turb_flux, flux) &
         !$acc async(2)
#else
         !$omp do schedule(runtime) private(invAreaCell, i, iEdge, cell1, cell2, r_tmp, k, s_tmp, iTracer, tracer_turb_flux, flux)
#endif         
         do iCell = 1, nCells
            invAreaCell = 1.0_RKIND / areaCell(iCell)
            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)
               cell1 = cellsOnEdge(1,iEdge)
               cell2 = cellsOnEdge(2,iEdge)

               r_tmp = config_redi_kappa * dvEdge(iEdge) / dcEdge(iEdge)

               do k = 1, maxLevelEdgeTop(iEdge)

                  flux = layerThicknessEdge(k, iEdge) * r_tmp
                  do iTracer = 1, num_tracers
                     ! \kappa_2 \nabla \phi on edge
                     tracer_turb_flux = tracers(iTracer, k, cell2) - tracers(iTracer, k, cell1)

                     ! div(h \kappa_2 \nabla \phi) at cell center
                     flux = flux * tracer_turb_flux

                     tend(iTracer, k, iCell) = tend(iTracer, k, iCell) - edgeSignOnCell(i, iCell) * flux * invAreaCell

                  end do
               end do

            end do
         end do
#ifdef MPAS_OPENACC
	  !$acc update host(tend) async(2)
	  !$omp end master
#else
      !$omp end do
#endif      

      endif

      ! Compute vertical derivative of tracers at cell center and top of layer
      do iTracer = 1, num_tracers
         ! Sync threads before starting on tracers
         call mpas_threading_barrier()

         nCells = nCellsArray( 2 )
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop collapse(2) &
         !$acc present(maxLevelCell,dTracerdZTopOfCell,tracers,zMid) &
         !$acc async(2)
#else
         !$omp do schedule(runtime) private(k)
#endif         
         do iCell = 1, nCells
            do k = 2,mxlvlCell
               if ( k <= maxLevelCell(iCell) ) then
                  dTracerdZTopOfCell(k,iCell) = (tracers(iTracer,k-1,iCell) - tracers(iTracer,k,iCell)) &
                                           / (zMid(k-1,iCell) - zMid(k,iCell))
			   endif                                           
            end do

            ! Approximation of dTracerdZTopOfCell on the top and bottom interfaces through the idea of having
            ! ghost cells above the top and below the bottom layers of the same depths and tracer density.
            ! Essentially, this enforces the boundary condition (d tracer)/dz = 0 at the top and bottom.
            !dTracerdZTopOfCell(1,iCell) = 0.0_RKIND
            !dTracerdZTopOfCell(maxLevelCell(iCell)+1,iCell) = 0.0_RKIND
         end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif      

         nEdges = nEdgesArray( 2 )
         ! Compute tracer gradient (gradTracerEdge) along the constant coordinate surface.
         ! The computed variables lives at edge and mid-layer depth
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop collapse(2) private(cell1, cell2) &
         !$acc present(maxLevelEdgeTop,gradTracerEdge,tracers,dcEdge) &
         !$acc async(1)
#else
         !$omp do schedule(runtime) private(cell1, cell2, k)
#endif         
         do iEdge = 1, nEdges
            do k=1,mxlvlEdge
               if ( k <= maxLevelEdgeTop(iEdge) ) then
                  cell1 = cellsOnEdge(1,iEdge)
                  cell2 = cellsOnEdge(2,iEdge)

                  gradTracerEdge(k,iEdge) = (tracers(iTracer,k,cell2) - tracers(iTracer,k,cell1)) / dcEdge(iEdge)
               endif   
            end do
         end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif      

         nEdges = nEdgesArray( 2 )
         ! Interpolate dTracerdZTopOfCell to edge and top of layer
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop collapse(2) private(cell1, cell2) &
         !$acc present(maxLevelEdgeTop,gradTracerEdge,tracers,dcEdge) &
         !$acc async(2)
#else
         !$omp do schedule(runtime) private(cell1, cell2, k)
#endif         
         do iEdge = 1, nEdges
            do k=1,mxlvlEdge
               if ( k <= maxLevelEdgeTop(iEdge) ) then
                  cell1 = cellsOnEdge(1,iEdge)
                  cell2 = cellsOnEdge(2,iEdge)
                  dTracerdZTopOfEdge(k,iEdge) = 0.5_RKIND * (dTracerdZTopOfCell(k,cell1) + dTracerdZTopOfCell(k,cell2))
               endif
            end do
            !dTracerdZTopOfEdge(maxLevelEdgeTop(iEdge)+1,iEdge) = 0.0_RKIND
         end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif      

         nEdges = nEdgesArray( 2 )
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop &
         !$acc present(maxLevelEdgeTop,gradTracerEdge,gradTracerTopOfEdge) &
         !$acc async(2)
#else
         !$omp do schedule(runtime)
#endif         
         do iEdge = 1, nEdges
            ! Approximation of values on the top and bottom interfaces through the idea of having ghost cells above
            ! the top and below the bottom layers of the same depths and tracer concentration.
            gradTracerTopOfEdge(1,iEdge) = gradTracerEdge(2,iEdge)
            gradTracerTopOfEdge(maxLevelEdgeTop(iEdge)+1,iEdge) = gradTracerEdge(max(maxLevelEdgeTop(iEdge),1),iEdge)
		 end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif      
		 
         ! Interpolate gradTracerEdge to edge and top of layer
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop collapse(2) private(k, h1, h2) &
         !$acc present(maxLevelEdgeTop,gradTracerEdge,gradTracerTopOfEdge) &
         !$acc async(1)
#else
         !$omp do schedule(runtime) private(k, h1, h2)
#endif         
         do iEdge = 1, nEdges
            do k=1,mxlvlEdge
               if ( k <= maxLevelEdgeTop(iEdge) ) then
                  h1 = layerThicknessEdge(k-1,iEdge)
                  h2 = layerThicknessEdge(k,iEdge)

                  ! Using second-order interpolation below
                  gradTracerTopOfEdge(k,iEdge) = (h2 * gradTracerEdge(k-1,iEdge) + h1 * gradTracerEdge(k,iEdge)) / (h1 + h2)
               endif
            end do

            ! Approximation of values on the top and bottom interfaces through the idea of having ghost cells above
            ! the top and below the bottom layers of the same depths and tracer concentration.
            !gradTracerTopOfEdge(1,iEdge) = gradTracerEdge(2,iEdge)
            !gradTracerTopOfEdge(maxLevelEdgeTop(iEdge)+1,iEdge) = gradTracerEdge(max(maxLevelEdgeTop(iEdge),1),iEdge)
         end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif      

         ! Compute \nabla\cdot(relativeSlope d\phi/dz)
         if(.not.config_disable_redi_horizontal_term2) then
            nCells = nCellsArray( 1 )
#ifdef MPAS_OPENACC
	        !$omp master
            !$acc parallel loop private(invAreaCell, i, iEdge, k, s_tmpU, s_tmpD, flux) &
            !$acc present(nEdgesOnCell,edgesOnCell,edgeSignOnCell,maxLevelEdgeTop, &
            !$acc         relativeSlopeTapering,relativeSlopeTopOfEdge,dTracerdZTopOfEdge, &
            !$acc         dvEdge,layerThicknessEdge,areaCell,tend) &
            !$acc async(1) wait(2)
#else
            !$omp do schedule(runtime) private(invAreaCell, i, iEdge, k, s_tmpU, s_tmpD, flux)
#endif            
            do iCell = 1, nCells
               invAreaCell = 1.0_RKIND / areaCell(iCell)
               do i = 1, nEdgesOnCell(iCell)
                  iEdge = edgesOnCell(i, iCell)
                  flux = edgeSignOnCell(i, iCell) * config_Redi_kappa * invAreaCell
                  do k = 1, maxLevelEdgeTop(iEdge)
                     s_tmpU = relativeSlopeTapering(k, iEdge) * relativeSlopeTopOfEdge(k, iEdge) * dTracerdZTopOfEdge(k, iEdge)
                     s_tmpD = relativeSlopeTapering(k+1, iEdge) * relativeSlopeTopOfEdge(k+1, iEdge) &
                            * dTracerdZTopOfEdge(k+1, iEdge)

                     flux = 0.5 * dvEdge(iEdge) * ( s_tmpU + s_tmpD ) * flux
                     flux = flux * layerThicknessEdge(k, iEdge)
                     tend(iTracer, k, iCell) = tend(iTracer, k, iCell) + flux
                  end do
               end do
            end do
#ifdef MPAS_OPENACC
	        !$acc update host(tend) async(1) wait(2)
	        !$omp end master
#else
            !$omp end do
#endif      
         endif

         ! Compute dz * d(relativeSlope\cdot\nabla\phi)/dz  (so the dz cancel out)

         ! Compute relativeSlope\cdot\nabla\phi (variable gradHTracerSlopedTopOfCell) at non-boundary edges

         nCells = nCellsArray( 1 )
         mxlvl = size(areaCellSum,1)
         
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop &
         !$acc present(areaCellSum,gradHTracerSlopedTopOfCell) &
         !$acc async(1)
#else
         !$omp do schedule(runtime)
#endif         
         do iCell = 1, nCells
            do k = 1, mxlvl
               areaCellSum(k, iCell) = 1.0e-34_RKIND
               gradHTracerSlopedTopOfCell(k, iCell) = 0.0_RKIND
            end do
         end do      
#ifdef MPAS_OPENACC
	    !$omp end master
#else
        !$omp end do
#endif      

#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop private(i, iedge, areaEdge, k, r_tmp) &
         !$acc present(areaCellSum,gradHTracerSlopedTopOfCell,maxLevelEdgeTop, &
         !$acc         nEdgesOnCell,edgesOnCell,dcEdge,dvEdge,&
         !$acc         relativeSlopeTapering,gradTracerTopOfEdge,relativeSlopeTopOfEdge) &
         !$acc async(1)
#else
         !$omp do schedule(runtime) private(i, iedge, areaEdge, k, r_tmp)
#endif         
         do iCell = 1, nCells
            !areaCellSum(:, iCell) = 1.0e-34_RKIND
            !gradHTracerSlopedTopOfCell(:, iCell) = 0.0_RKIND
            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)
               areaEdge = 0.5_RKIND * dcEdge(iEdge) * dvEdge(iEdge)
               do k = 1, maxLevelEdgeTop(iEdge)
                  r_tmp = areaEdge * relativeSlopeTapering(k, iEdge) * relativeSlopeTopOfEdge(k,iEdge) &
                        * gradTracerTopOfEdge(k,iEdge)
                  gradHTracerSlopedTopOfCell(k, iCell) = gradHTracerSlopedTopOfCell(k, iCell) + r_tmp
                  areaCellSum(k, iCell) = areaCellSum(k, iCell) + areaEdge
               end do
            end do
         end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif

         nCells = nCellsArray( 1 )
#ifdef MPAS_OPENACC
	     !$omp master
         !$acc parallel loop collapse(2) &
         !$acc present(areaCellSum,gradHTracerSlopedTopOfCell,maxLevelCell) &
         !$acc async(1)
#else
         !$omp do schedule(runtime) private(k)
#endif         
         do iCell=1,nCells
            do k = 1, mxlvlCell
               if ( k <= maxLevelCell(iCell) ) then
                  gradHTracerSlopedTopOfCell(k,iCell) = gradHTracerSlopedTopOfCell(k,iCell)/areaCellSum(k,iCell)
               endif   
            end do
         end do
#ifdef MPAS_OPENACC
	     !$omp end master
#else
         !$omp end do
#endif

         if(.not.config_disable_redi_horizontal_term3) then
            nCells = nCellsArray( 1 )
#ifdef MPAS_OPENACC
	        !$omp master
            !$acc parallel loop  &
            !$acc present(gradHTracerSlopedTopOfCell,maxLevelCell) &
            !$acc async(1)
#else
            !$omp do schedule(runtime) private(iCell)
#endif            
            do iCell = 1, nCells
               gradHTracerSlopedTopOfCell(1,iCell) = 0.0_RKIND
               gradHTracerSlopedTopOfCell(maxLevelCell(iCell)+1,iCell) = 0.0_RKIND
            end do
#ifdef MPAS_OPENACC
	        !$omp end master
#else
            !$omp end do
#endif         
            
#ifdef MPAS_OPENACC
	        !$omp master
            !$acc parallel loop collapse(2) &
            !$acc present(tend,gradHTracerSlopedTopOfCell,maxLevelCell) &
            !$acc async(1)
#else
            !$omp do schedule(runtime) private(k, s_tmp)
#endif            
            do iCell = 1, nCells
               ! impose no-flux boundary conditions at top and bottom of column
               !gradHTracerSlopedTopOfCell(1,iCell) = 0.0_RKIND
               !gradHTracerSlopedTopOfCell(maxLevelCell(iCell)+1,iCell) = 0.0_RKIND
               do k = 1, mxlvlCell
                  if ( k <= maxLevelCell(iCell) ) then
                     tend(iTracer,k,iCell) = tend(iTracer,k,iCell) + config_Redi_kappa * &
                         (gradHTracerSlopedTopOfCell(k,iCell) - gradHTracerSlopedTopOfCell(k+1,iCell))
                  endif       
               end do
            end do
#ifdef MPAS_OPENACC
			!$acc update host(tend) async(1)
	        !$omp end master
#else
            !$omp end do
#endif         
         endif

      end do  ! iTracer

#ifdef MPAS_OPENACC
	  !$omp master
	  !$acc wait
	  !$acc exit data delete(tracers,tend,layerThicknessEdge,zMid,relativeSlopeTopOfEdge, &
	  !$acc            relativeSlopeTapering,relativeSlopeTaperingCell, &
	  !$acc            gradTracerEdge,gradTracerTopOfEdge,gradHTracerSlopedTopOfCell, &
	  !$acc            dTracerdZTopOfCell,dTracerdZTopOfEdge,areaCellSum)
	  !$omp end master
#endif

      call mpas_threading_barrier()
      call mpas_deallocate_scratch_field(gradTracerEdgeField, .true.)
      call mpas_deallocate_scratch_field(gradTracerTopOfEdgeField, .true.)
      call mpas_deallocate_scratch_field(gradHTracerSlopedTopOfCellField, .true.)
      call mpas_deallocate_scratch_field(dTracerdZTopOfCellField, .true.)
      call mpas_deallocate_scratch_field(dTracerdZTopOfEdgeField, .true.)

      call mpas_timer_stop("tracer redi")

   !--------------------------------------------------------------------

   end subroutine ocn_tracer_hmix_redi_tend!}}}

!***********************************************************************
!
!  routine ocn_tracer_hmix_redi_init
!
!> \brief   Initializes ocean tracer horizontal mixing quantities
!> \author  Doug Jacobsen, Mark Petersen, Todd Ringler
!> \date    September 2011
!> \details
!>  This routine initializes a variety of quantities related to
!>  Laplacian horizontal velocity mixing in the ocean.
!
!-----------------------------------------------------------------------

   subroutine ocn_tracer_hmix_redi_init(err)!{{{

   !--------------------------------------------------------------------

      !-----------------------------------------------------------------
      !
      ! call individual init routines for each parameterization
      !
      !-----------------------------------------------------------------

      integer, intent(out) :: err !< Output: error flag

      logical, pointer :: config_use_standardGM

      err = 0

      call mpas_pool_get_config(ocnConfigs, 'config_use_standardGM', config_use_standardGM)

      rediOn = .false.

      if ( config_use_standardGM ) then
          rediOn = .true.
      endif


   !--------------------------------------------------------------------

   end subroutine ocn_tracer_hmix_redi_init!}}}

!***********************************************************************

end module ocn_tracer_hmix_redi

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
