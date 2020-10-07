! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!
!  ocn_diagnostics_routines
!
!> \brief MPAS ocean diagnostics driver
!> \author Mark Petersen
!> \date   November 2013
!> \details
!>  This module contains the routines for computing individual diagnostic variables
!
!-----------------------------------------------------------------------
!#undef MPAS_OPENACC

module ocn_diagnostics_routines

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_constants
   use mpas_timer
   use ocn_mesh, only: edgeSignOnVertex, edgesOnVertex

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

   public :: ocn_relativeVorticity_circulation

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------

!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_relativeVorticity_circulation
!
!> \brief   Computes relative vorticity and circulation
!> \author  Mark Petersen, Doug Jacobsen, Todd Ringler
!> \date    November 2013
!> \details
!>  Computes relative vorticity and circulation
!
!-----------------------------------------------------------------------

   subroutine ocn_relativeVorticity_circulation(relativeVorticity, circulation, meshPool, normalVelocity, err)!{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type (mpas_pool_type), intent(in) :: &
         meshPool

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(out) :: &
         relativeVorticity

      real (kind=RKIND), dimension(:,:), intent(out) :: &
         circulation

      integer, intent(out) :: err !< Output: error flag

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      integer :: iVertex, iEdge, i, k, vertexDegree, nVertices, mxlevel
      integer, pointer :: nVerticesPtr, vertexDegreePtr
      integer, dimension(:), pointer :: maxLevelVertexBot
      !integer, dimension(:,:), pointer :: edgesOnVertex

      real (kind=RKIND) :: &
              invAreaTri,       &! 1/area of triangles
              rSign,            &! sign of vector at edge
              rTmp               ! temp for common factors
      real (kind=RKIND), dimension(:), pointer :: &
              dcEdge, areaTriangle

      call mpas_pool_get_dimension(meshPool, 'nVertices', nVerticesPtr)
      nVertices = nVerticesPtr
      
      call mpas_pool_get_dimension(meshPool, 'vertexDegree', vertexDegreePtr)
      vertexDegree = vertexDegreePtr
      
      call mpas_pool_get_array(meshPool, 'maxLevelVertexBot', maxLevelVertexBot)
      !call mpas_pool_get_array(meshPool, 'edgesOnVertex', edgesOnVertex)

      call mpas_pool_get_array(meshPool, 'areaTriangle', areaTriangle)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)

      err = 0
	  mxlevel = size(circulation,1)
	  
	  #ifdef MPAS_OPENACC
      !$acc parallel loop gang vector collapse(2) &
      !$acc    present(circulation, relativeVorticity) async(1)
      #else
      !$omp parallel do schedule(runtime) 
      #endif
      do iVertex = 1, nVertices
         do k = 1, mxlevel
            circulation (k,iVertex) = 0.0_RKIND
            relativeVorticity(k,iVertex) = 0.0_RKIND
         end do
      end do
      #ifndef MPAS_OPENACC
      !$omp end parallel do
      #endif
      
      #ifdef MPAS_OPENACC
      !$acc parallel loop gang vector collapse(2) &
      !$acc    present(circulation, relativeVorticity, areaTriangle, &
      !$acc            edgesOnVertex, edgeSignOnVertex, dcEdge, &
      !$acc            maxLevelVertexBot, normalVelocity) &
      !$acc    private(i, k, iEdge, rSign, rTmp, invAreaTri) async(1)
      #else
      !$omp parallel do schedule(runtime) &
      !$omp    private(i, k, iEdge, rSign, rTmp, invAreaTri)
      #endif
      do iVertex = 1, nVertices
         do k = 1, mxlevel
            invAreaTri = 1.0_RKIND / areaTriangle(iVertex)
            if ( k <= maxLevelVertexBot(iVertex) ) then
               rTmp = 0.0_RKIND
               do i = 1, vertexDegree
                  iEdge = edgesOnVertex(i,iVertex)
                  rSign = edgeSignOnVertex(i,iVertex)*dcEdge(iEdge)
                  rTmp = rTmp + rSign*normalVelocity(k, iEdge)
               end do
               circulation (k,iVertex) = circulation(k, iVertex) + rTmp
               relativeVorticity(k,iVertex) = relativeVorticity(k,iVertex) + &
                                              rTmp*invAreaTri
            endif
         end do ! k
      end do ! vertex
!
!      do iVertex = 1, nVertices
!         invAreaTri = 1.0_RKIND / areaTriangle(iVertex)
!         do i = 1, vertexDegree
!            iEdge = edgesOnVertex(i,iVertex)
!            rSign = edgeSignOnVertex(i,iVertex)*dcEdge(iEdge)
!            do k = 1, maxLevelVertexBot(iVertex)
!               rTmp = rSign*normalVelocity(k, iEdge)
!
!               circulation (k,iVertex) = circulation(k, iVertex) + rTmp
!               relativeVorticity(k,iVertex) = relativeVorticity(k,iVertex) + &
!                                         rTmp*invAreaTri
!            end do ! vert levels
!         end do ! vertexDegree
!      end do ! vertex
      #ifndef MPAS_OPENACC
      !$omp end parallel do
      #endif

   !--------------------------------------------------------------------

   end subroutine ocn_relativeVorticity_circulation!}}}

end module ocn_diagnostics_routines

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
