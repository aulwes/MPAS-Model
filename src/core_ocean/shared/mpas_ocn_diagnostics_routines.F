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
#undef MPAS_OPENACC
module ocn_diagnostics_routines

   use mpas_derived_types
   use mpas_log

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

   public :: ocn_relativeVorticity_circulation

   !--------------------------------------------------------------------
   ! Private module variables
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

   subroutine ocn_relativeVorticity_circulation(relVorticity, &
                                  circulation, normalVelocity, err)!{{{

      !-----------------------------------------------------------------
      ! input variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(in) :: &
         normalVelocity           !< [in] velocity normal to edge

      !-----------------------------------------------------------------
      ! output variables
      !-----------------------------------------------------------------

      real (kind=RKIND), dimension(:,:), intent(out) :: &
         relVorticity,      &!< [out] relative vorticity
         circulation         !< [out] circulation of flow

      integer, intent(out) :: err !< [out] error flag

      !-----------------------------------------------------------------
      ! local variables
      !-----------------------------------------------------------------

      integer :: &
         iVertex, iEdge, i, k  ! loop iterators

      real (kind=RKIND) :: &
         invAreaTri,       &! 1/area of triangles
         rSign,            &! sign of vector at edge
         rTmp               ! temp for common factors

      ! End preamble
      !-------------
      ! Begin code

      err = 0

      #ifdef MPAS_OPENACC
      !$acc parallel loop gang vector &
      !$acc    present(circulation, relVorticity, areaTriangle, &
      !$acc            edgesOnVertex, edgeSignOnVertex, dcEdge, &
      !$acc            maxLevelVertexBot, normalVelocity) &
      !$acc    private(i, k, iEdge, rSign, rTmp, invAreaTri) async(1)
      #else
      !$omp parallel do schedule(runtime) &
      !$omp    private(i, k, iEdge, rSign, rTmp, invAreaTri)
      #endif
      do iVertex = 1, nVerticesAll
         circulation (:,iVertex) = 0.0_RKIND
         relVorticity(:,iVertex) = 0.0_RKIND
         invAreaTri = 1.0_RKIND / areaTriangle(iVertex)
         do i = 1, vertexDegree
            iEdge = edgesOnVertex(i,iVertex)
            rSign = edgeSignOnVertex(i,iVertex)*dcEdge(iEdge)
            do k = 1, maxLevelVertexBot(iVertex)
               rTmp = rSign*normalVelocity(k, iEdge)

               circulation (k,iVertex) = circulation(k, iVertex) + rTmp
               relVorticity(k,iVertex) = relVorticity(k,iVertex) + &
                                         rTmp*invAreaTri
            end do ! vert levels
         end do ! vertexDegree
      end do ! vertex
      #ifndef MPAS_OPENACC
      !$omp end parallel do
      #endif

   !--------------------------------------------------------------------

   end subroutine ocn_relativeVorticity_circulation!}}}

end module ocn_diagnostics_routines

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
