! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!
module ocn_gm

   use mpas_derived_types
   use mpas_pool_routines
   use mpas_timer
   use mpas_constants
   use mpas_threading

   use ocn_constants
   use ocn_config

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

   public :: ocn_GM_compute_Bolus_velocity, &
             ocn_GM_init

   !--------------------------------------------------------------------
   !
   ! Private module variables
   !
   !--------------------------------------------------------------------
   private :: tridiagonal_solve

   ! Config options
   real(kind=RKIND), dimension(:), allocatable :: tridiagA, tridiagB, tridiagC, rightHandSide
   real(kind=RKIND), dimension(:), allocatable :: bTemp,rTemp

   real (kind=RKIND), parameter :: p_epsGM = 1.0e-12_RKIND

   logical :: local_config_GM_lat_variable_c2
   logical :: local_config_GM_kappa_lat_depth_variable
   real(kind=RKIND), parameter :: local_config_GM_min_phase_speed = 0.1_RKIND
   real(kind=RKIND) :: slopeTaperFactor, sfcTaperFactor, rediGMinitValue
!***********************************************************************

contains

!***********************************************************************
!
!  routine ocn_GM_compute_Bolus_velocity
!
!> \brief   Computes GM Bolus velocity
!> \details
!>  This routine is the main driver for the Gent-McWilliams (GM) parameterization.
!>  It computes horizontal and vertical density gradients, the slope
!>  of isopycnal surfaces, and solves a boundary value problem in each column
!>  for the stream function, which is used to compute the Bolus velocity.
!
!-----------------------------------------------------------------------

   subroutine ocn_GM_compute_Bolus_velocity(statePool, diagnosticsPool, &
                                            meshPool, scratchPool, timeLevelIn)
      !{{{

      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      type(mpas_pool_type), intent(in) :: &
         statePool, &! pool containing state variables
         meshPool              ! pool containing mesh quantities

      integer, intent(in), optional :: &
         timeLevelIn          ! time level for state variables

      !-----------------------------------------------------------------
      !
      ! input/output variables
      !
      !-----------------------------------------------------------------

      type(mpas_pool_type), intent(inout) :: &
         diagnosticsPool, &! pool containing some diagnostics
         scratchPool           ! pool containing some scratch space

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      real(kind=RKIND), dimension(:, :), pointer :: density, displacedDensity, zMid, normalGMBolusVelocity, &
                                                    layerThicknessEdge, gradDensityEdge, &
                                                    k33, gmStreamFuncTopOfEdge, BruntVaisalaFreqTop, gmStreamFuncTopOfCell, &
                                                    layerThickness, inSituThermalExpansionCoeff, &
                                                    inSituSalineContractionCoeff, kappaGMCell, kappaRediSfcTaper

      real(kind=RKIND), dimension(:), pointer   :: gmBolusKappa, cGMphaseSpeed, ssh
      real(kind=RKIND), dimension(:, :, :), pointer :: slopeTriadUp, slopeTriadDown
      real(kind=RKIND), dimension(:), pointer   :: areaCell, dcEdge, dvEdge
      integer, dimension(:), pointer   :: maxLevelEdgeTop, maxLevelCell, nEdgesOnCell, indMLD
      integer, dimension(:, :), pointer :: cellsOnEdge, edgesOnCell
      integer                          :: i, k, iEdge, cell1, cell2, iCell, N, iter, iCellSelf, maxLocation
      real(kind=RKIND)                 :: h1, h2, areaEdge, c, BruntVaisalaFreqTopEdge, rtmp
      real(kind=RKIND)                 :: kappaGMEdge, sumN2, countN2, maxN, kappaSum, ltSum
      real(kind=RKIND)                 :: zMLD, sshEdge, sfcTaper
      real(kind=RKIND) :: gradDensityTopOfEdge, dcEdgeInv, drhoDx, drhoDT, drhoDS, dTdx, dSdx, BVFcent
      real(kind=RKIND) :: slopeTaperUp, slopeTaperDown, sfcTaperUp, sfcTaperDown, invAreaCell
      real(kind=RKIND), dimension(:), allocatable :: dzTop, dTdzTop, dSdzTop, k33Norm
      real (kind=RKIND)                :: epsGM,ltem,ltem1,coeff,ltdenom,inv_slope, &
                                          cgm_min_strat_ratio, credi_max_slope, cgm_const_gwspeed

      ! Dimensions
      integer :: nsmooth, nCells, nEdges, lo, hi, mxlvl, n2d
      integer, pointer :: nVertLevelsPtr
      integer :: nVertLevels
      integer, dimension(:), pointer :: nCellsArray, nEdgesArray

      type(field2DReal), pointer :: gradDensityEdgeField

      type(mpas_pool_type), pointer :: tracersPool
      real(kind=RKIND), dimension(:, :, :), pointer :: activeTracers
      integer, pointer :: indexTemperaturePtr, indexSalinityPtr
      integer :: timeLevel, indexTemperature, indexSalinity

      if (present(timeLevelIn)) then
         timeLevel = timeLevelIn
      else
         timeLevel = 1
      end if

      call mpas_timer_start('gm bolus velocity')

      epsGM = p_epsGM
      cgm_min_strat_ratio = config_GM_min_stratification_ratio
      credi_max_slope = config_Redi_maximum_slope
      cgm_const_gwspeed = config_gm_constant_gravwavespeed
      
      call mpas_pool_get_array(statePool, 'layerThickness', layerThickness, timeLevel)
      call mpas_pool_get_subpool(statePool, 'tracers', tracersPool)
      call mpas_pool_get_array(statePool, 'ssh', ssh, timeLevel)
      call mpas_pool_get_array(tracersPool, 'activeTracers', activeTracers, timeLevel)
      call mpas_pool_get_dimension(tracersPool, 'index_temperature', indexTemperaturePtr)
      indexTemperature = indexTemperaturePtr
      
      call mpas_pool_get_dimension(tracersPool, 'index_salinity', indexSalinityPtr)
      indexSalinity = indexSalinityPtr

      call mpas_pool_get_array(diagnosticsPool, 'density', density)
      call mpas_pool_get_array(diagnosticsPool, 'displacedDensity', displacedDensity)
      call mpas_pool_get_array(diagnosticsPool, 'inSituThermalExpansionCoeff', inSituThermalExpansionCoeff)
      call mpas_pool_get_array(diagnosticsPool, 'inSituSalineContractionCoeff', inSituSalineContractionCoeff)
      call mpas_pool_get_array(diagnosticsPool, 'zMid', zMid)
      call mpas_pool_get_array(diagnosticsPool, 'indMLD', indMLD)

      call mpas_pool_get_array(diagnosticsPool, 'cGMphaseSpeed', cGMphaseSpeed)
      call mpas_pool_get_array(diagnosticsPool, 'kappaGMCell', kappaGMCell)
      call mpas_pool_get_array(diagnosticsPool, 'kappaRediSfcTaper', kappaRediSfcTaper)
      call mpas_pool_get_array(diagnosticsPool, 'normalGMBolusVelocity', normalGMBolusVelocity)
      call mpas_pool_get_array(diagnosticsPool, 'slopeTriadUp', slopeTriadUp)
      call mpas_pool_get_array(diagnosticsPool, 'slopeTriadDown', slopeTriadDown)
      call mpas_pool_get_array(diagnosticsPool, 'k33', k33)
      call mpas_pool_get_array(diagnosticsPool, 'layerThicknessEdge', layerThicknessEdge)
      call mpas_pool_get_array(diagnosticsPool, 'zMid', zMid)
      call mpas_pool_get_array(diagnosticsPool, 'BruntVaisalaFreqTop', BruntVaisalaFreqTop)
      call mpas_pool_get_array(diagnosticsPool, 'gmStreamFuncTopOfEdge', gmStreamFuncTopOfEdge)
      call mpas_pool_get_array(diagnosticsPool, 'gmStreamFuncTopOfCell', gmStreamFuncTopOfCell)
      call mpas_pool_get_array(diagnosticsPool, 'gmBolusKappa', gmBolusKappa)

      call mpas_pool_get_array(meshPool, 'maxLevelEdgeTop', maxLevelEdgeTop)
      call mpas_pool_get_array(meshPool, 'maxLevelCell', maxLevelCell)
      call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
      call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
      call mpas_pool_get_array(meshPool, 'dcEdge', dcEdge)
      call mpas_pool_get_array(meshPool, 'dvEdge', dvEdge)
      call mpas_pool_get_array(meshPool, 'nEdgesOnCell', nEdgesOnCell)
      call mpas_pool_get_array(meshPool, 'edgesOnCell', edgesOnCell)

      call mpas_pool_get_dimension(meshPool, 'nEdgesArray', nEdgesArray)
      call mpas_pool_get_dimension(meshPool, 'nCellsArray', nCellsArray)
      call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevelsPtr)
      nVertLevels = nVertLevelsPtr
      n2d = size(slopeTriadUp, 2)
      
      call mpas_pool_get_field(scratchPool, 'gradDensityEdge', gradDensityEdgeField)
      call mpas_allocate_scratch_field(gradDensityEdgeField, .True., .false.)

      gradDensityEdge => gradDensityEdgeField%array

#ifdef MPAS_OPENACC
      !$omp master
      !$acc enter data copyin(normalGMBolusVelocity, &
      !$acc       gradDensityEdge, slopeTriadUp,slopeTriadDown, &
      !$acc       cGMphaseSpeed, kappaGMCell, kappaRediSfcTaper, &
      !$acc       gmStreamFuncTopOfEdge,gmStreamFuncTopOfCell,gmBolusKappa, &
      !$acc       BruntVaisalaFreqTop, k33, displacedDensity, density, zMid, &
      !$acc       inSituThermalExpansionCoeff, inSituSalineContractionCoeff, indMLD, &
      !$acc       layerThicknessEdge) &
      !$acc       async(1)
      !$omp end master
#endif

      allocate (rightHandSide(nVertLevels))
      allocate (tridiagA(nVertLevels))
      allocate (tridiagB(nVertLevels))
      allocate (tridiagC(nVertLevels))

      nCells = nCellsArray(size(nCellsArray))
      nEdges = nEdgesArray(size(nEdgesArray))

      mxlvl = maxval(maxLevelEdgeTop)
      allocate(bTemp(mxlvl), rTemp(mxlvl))

#ifdef MPAS_OPENACC
      !$omp master
      !$acc parallel loop collapse(2) &
      !$acc present(gradDensityEdge,normalGMBolusVelocity) &
      !$acc async(1)
#else
      !$omp do schedule(runtime) private(k)
#endif      
      do iEdge = 1, nEdges
         do k = 1, nVertLevels
            gradDensityEdge(k, iEdge) = 0.0_RKIND
            normalGMBolusVelocity(k, iEdge) = 0.0_RKIND
         end do
      end do
#ifdef MPAS_OPENACC
      !$omp end master
#else      
      !$omp end do
#endif      

#ifdef MPAS_OPENACC
      !$omp master
      !$acc parallel loop collapse(3) &
      !$acc present(slopeTriadUp,slopeTriadDown) &
      !$acc async(1)
#else
      !$omp do schedule(runtime) private(k)
#endif      
      do iEdge = 1, nEdges
         do i = 1,n2d
            do k = 1, nVertLevels
               slopeTriadUp(k, i, iEdge) = 0.0_RKIND
               slopeTriadDown(k, i, iEdge) = 0.0_RKIND
            end do
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
      !$acc present(k33) &
      !$acc async(1)
#else
      !$omp do schedule(runtime)
#endif      
      do iCell = 1, nCells + 1
         do k = 1, nVertLevels
            k33(k, iCell) = 0.0_RKIND
         end do
      end do
#ifdef MPAS_OPENACC
      !$omp end master
#else      
      !$omp end do
#endif      
      allocate (dzTop(nVertLevels + 1))
      allocate (dTdzTop(nVertLevels + 1))
      allocate (dSdzTop(nVertLevels + 1))
      allocate (k33Norm(nVertLevels + 1))

#ifdef MPAS_OPENACC
      !$omp master
      !$acc parallel loop gang vector collapse(2) &
      !$acc present(kappaGMCell,kappaRediSfcTaper) &
      !$acc async(1)
#else
      !$omp do schedule(runtime)
#endif      
      do iCell = 1, nCells
         do k = 1,nVertLevels
            kappaGMCell(k, iCell) = 1.0_RKIND
            kappaRediSfcTaper(k, iCell) = 1.0_RKIND
         end do
      end do
#ifdef MPAS_OPENACC
      !$omp end master
#else      
      !$omp end do
#endif      

      if (local_config_GM_kappa_lat_depth_variable) then

#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop &
         !$acc present(BruntVaisalaFreqTop,maxLevelCell,kappaGMCell) &
         !$acc private(maxLocation, cell1, cell2, k, BruntVaisalaFreqTopEdge, maxN) &
         !$acc async(1)
#else
         !$omp do schedule(runtime) private(maxLocation, cell1, cell2, k, BruntVaisalaFreqTopEdge, maxN)
#endif
         do iCell = 1, nCells
            k = 1
            maxN = max(BruntVaisalaFreqTop(k, iCell), 0.0_RKIND)
            do while (BruntVaisalaFreqTop(k + 1, iCell) > maxN .and. k < maxLevelCell(iCell) - 1)
               k = k + 1
               maxN = max(BruntVaisalaFreqTop(k, iCell), 0.0_RKIND)
            end do

            maxLocation = k
            maxN = 1.0 / (maxN + 1.0E-10_RKIND)
            do k = maxLocation, maxLevelCell(iCell)
               BruntVaisalaFreqTopEdge = max(BruntVaisalaFreqTop(k, iCell), 0.0_RKIND)

               kappaGMCell(k, iCell) = min(max(cgm_min_strat_ratio, &
                                               BruntVaisalaFreqTopEdge * maxN), &
                                           1.0_RKIND)
            end do
         end do
#ifdef MPAS_OPENACC
         !$acc update host(kappaGMCell) async(1)
         !$omp end master
#else      
         !$omp end do
#endif      
      end if

      if (config_Redi_use_surface_taper) then
#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop &
         !$acc present(indMLD,ssh,zMid,kappaRediSfcTaper) &
         !$acc private(zMLD, k) &
         !$acc async(1)
#else
         !$omp do schedule (runtime) private(zMLD, k)
#endif         
         do iCell = 1, nCells
            k = max(1, indMLD(iCell))
            zMLD = 1.0 / (ssh(iCell) - zMid(k, iCell))

            do k = 1, indMLD(iCell)
               kappaRediSfcTaper(k, iCell) = abs((ssh(iCell) - zMid(k, iCell)) * zMLD)
            end do
         end do
#ifdef MPAS_OPENACC
         !$acc update host(kappaRediSfcTaper) async(1)
         !$omp end master
#else      
         !$omp end do
#endif      
      end if

      mxlvl = size(k33, 1)
      inv_slope = 1.0 / credi_max_slope
      
      nCells = nCellsArray(3)
#ifdef MPAS_OPENACC
      !$omp master
      !$acc parallel loop &
      !$acc present(areaCell,layerThickness,activeTracers,maxLevelCell,nEdgesOnCell, &
      !$acc         edgesOnCell,cellsOnEdge,dcEdge,dvEdge,inSituThermalExpansionCoeff,&
      !$acc         inSituSalineContractionCoeff,slopeTriadUp,slopeTriadDown,kappaRediSfcTaper, &
      !$acc         k33) &
      !$acc private(invAreaCell,dzTop,dTdzTop,dSdzTop,k33Norm,cell1,cell2,hi, &
      !$acc         drhoDT, drhoDS, dTdx, dSdx, drhoDx, dcEdgeInv, iCellSelf, &
      !$acc         sfcTaper, sfcTaperUp, sfcTaperDown, slopeTaperDown, slopeTaperUp) &
      !$acc async(1)
#else
      !$omp do schedule(runtime) private(i, k, iEdge, cell1, cell2, iCellSelf, &
      !$omp& invAreaCell, dcEdgeInv, areaEdge, drhoDT, drhoDS, dTdx, dSdx, drhoDx, &
      !$omp& sfcTaper, sfcTaperUp, sfcTaperDown, slopeTaperDown, slopeTaperUp)
#endif      
      do iCell = 1, nCells
         invAreaCell = 1.0_RKIND/areaCell(iCell)
         hi = maxLevelCell(iCell)
         !$acc loop seq
         do k = 1,hi + 1
            k33(k,iCell) = 0.0_RKIND
            k33Norm(k) = epsGM
            if ( (k >= 2) .and. (k <= maxLevelCell(iCell)) ) then
               dzTop(k) = 0.5_RKIND*(layerThickness(k - 1, iCell) + layerThickness(k, iCell))
               dTdzTop(k) = (activeTracers(indexTemperature, k - 1, iCell) &
                          - activeTracers(indexTemperature, k, iCell)) &
                         /dzTop(k)
               dSdzTop(k) = (activeTracers(indexSalinity, k - 1, iCell) &
                          - activeTracers(indexSalinity, k, iCell)) &
                         /dzTop(k)
            endif
         end do
         dzTop(1) = -1e-15_RKIND
         dTdzTop(1) = -1e-15_RKIND
         dSdzTop(1) = -1e-15_RKIND
         dzTop(maxLevelCell(iCell) + 1) = -1e-15_RKIND
         dTdzTop(maxLevelCell(iCell) + 1) = -1e-15_RKIND
         dSdzTop(maxLevelCell(iCell) + 1) = -1e-15_RKIND

         do i = 1, nEdgesOnCell(iCell)
            iEdge = edgesOnCell(i, iCell)
            cell1 = cellsOnEdge(1, iEdge)
            cell2 = cellsOnEdge(2, iEdge)
            if (cell1 == iCell) then
               iCellSelf = 1
            else  ! cell2 == iCell
               iCellSelf = 2
            end if
            dcEdgeInv = 1.0_RKIND/dcEdge(iEdge)
            areaEdge = dcEdge(iEdge)*dvEdge(iEdge)

            do k = 1, maxLevelEdgeTop(iEdge)
               drhoDT = -inSituThermalExpansionCoeff(k, iCell)
               drhoDS = inSituSalineContractionCoeff(k, iCell)
               dTdx = (activeTracers(indexTemperature, k, cell2) &
                       - activeTracers(indexTemperature, k, cell1)) &
                      *dcEdgeInv
               dSdx = (activeTracers(indexSalinity, k, cell2) &
                       - activeTracers(indexSalinity, k, cell1)) &
                      *dcEdgeInv
               drhoDx = drhoDT*dTdx + drhoDS*dSdx

               ! Always compute *Up on the top cell and *Down on the bottom
               ! cell, even though they are never used. This avoids an if
               ! statement or separate computation for top and bottom.
               slopeTriadUp(k, iCellSelf, iEdge) = &
                  -drhoDx/ &
                  (drhoDT*dTdzTop(k) &
                   + drhoDS*dSdzTop(k) + 1E-15_RKIND)
               slopeTriadDown(k, iCellSelf, iEdge) = &
                  -drhoDx/ &
                  (drhoDT*dTdzTop(k + 1) &
                   + drhoDS*dSdzTop(k + 1) + 1E-15_RKIND)

               ! set taper of slope ('F' function from Danabasoglu and McWilliams 95)
               if (abs(slopeTriadDown(k, iCellSelf, iEdge)) > 0.6_RKIND*credi_max_slope) then
                  slopeTaperDown = 0.0_RKIND
               else if (abs(slopeTriadDown(k, iCellSelf, iEdge)) < 0.2_RKIND*credi_max_slope) then
                  slopeTaperDown = 1.0_RKIND
               else
                  slopeTaperDown = 0.5_RKIND*(1.0_RKIND - ((2.5_RKIND*abs(slopeTriadDown(k, iCellSelf, iEdge))) * &
                                   inv_slope - 1.0_RKIND)*(4.0_RKIND - abs(10.0_RKIND* &
                                   abs(slopeTriadDown(k, iCellSelf, iEdge)) * inv_slope - 4.0_RKIND)))
               end if
               if (abs(slopeTriadUp(k, iCellSelf, iEdge)) > 0.6_RKIND*credi_max_slope) then
                  slopeTaperUp = 0.0_RKIND
               else if (abs(slopeTriadUp(k, iCellSelf, iEdge)) < 0.2_RKIND*credi_max_slope) then
                  slopeTaperUp = 1.0_RKIND
               else
                  slopeTaperUp = 0.5_RKIND*(1.0_RKIND - ((2.5_RKIND*abs(slopeTriadUp(k, iCellSelf, iEdge))) * &
                                 inv_slope - 1.0_RKIND)*(4.0_RKIND - abs(10.0_RKIND* &
                                 abs(slopeTriadUp(k, iCellSelf, iEdge)) * inv_slope - 4.0_RKIND)))
               end if

               slopeTaperUp = 1.0_RKIND + slopeTaperFactor*(slopeTaperUp - 1.0_RKIND)
               slopeTaperDown = 1.0_RKIND + slopeTaperFactor*(slopeTaperDown - 1.0_RKIND)

               sfcTaper = min(kappaRediSfcTaper(k, cell1), kappaRediSfcTaper(k, cell2))
               sfcTaperUp = 1.0_RKIND + sfcTaperFactor*(sfcTaper - 1.0_RKIND)
               sfcTaperDown = 1.0_RKIND + sfcTaperFactor*(sfcTaper - 1.0_RKIND)

               ! Griffies 1998 eqn 34
               if (k > 1) then
                  k33(k, iCell) = k33(k, iCell) + slopeTaperUp*sfcTaperUp* &
                                  areaEdge*dzTop(k)*slopeTriadUp(k, iCellSelf, iEdge)**2
                  k33Norm(k) = k33Norm(k) + areaEdge*dzTop(k)
               end if

               k33(k + 1, iCell) = k33(k + 1, iCell) + slopeTaperDown*sfcTaperDown* &
                                   areaEdge*dzTop(k + 1)*slopeTriadDown(k, iCellSelf, iEdge)**2
               k33Norm(k + 1) = k33Norm(k + 1) + areaEdge*dzTop(k + 1)

               slopeTriadUp(k, iCellSelf, iEdge) = &
                  slopeTaperUp*sfcTaperUp*slopeTriadUp(k, iCellSelf, iEdge)
               slopeTriadDown(k, iCellSelf, iEdge) = &
                  slopeTaperDown*sfcTaperDown*slopeTriadDown(k, iCellSelf, iEdge)

            end do ! maxLevelEdgeTop(iEdge)
         end do ! nEdgesOnCell(iCell)

         ! Normalize k33
         do k = 2, maxLevelCell(iCell)
            k33(k, iCell) = k33(k, iCell)/k33Norm(k)*kappaRediSfcTaper(k, iCell)* &
                            kappaGMCell(k, iCell)
         end do
         k33(1, iCell) = 0.0_RKIND
         k33(maxLevelCell(iCell) + 1, iCell) = 0.0_RKIND
      end do ! iCell
#ifdef MPAS_OPENACC
      !$acc update host(k33,slopeTriadUp,slopeTriadDown) async(1)
      !$omp end master
#else      
      !$omp end do
#endif

      deallocate (dzTop)
      deallocate (dTdzTop)
      deallocate (dSdzTop)
      deallocate (k33Norm)

      ! allow disabling of K33 for testing
      if (config_disable_redi_k33) then
         nCells = nCellsArray(size(nCellsArray))
#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop gang vector collapse(2) &
         !$acc present(k33) async(1)
#else         
         !$omp do schedule(runtime)
#endif         
         do iCell = 1, nCells
            do k = 1,mxlvl
               k33(k, iCell) = 0.0_RKIND
            end do
         end do
#ifdef MPAS_OPENACC
         !$acc update host(k33) async(1)
         !$omp end master
#else      
         !$omp end do
#endif         
      end if

      !--------------------------------------------------------------------
      !
      ! Compute stream function and Bolus velocity for Bolus part of GM
      !
      !--------------------------------------------------------------------

      if (config_use_GM) then
         nEdges = nEdgesArray(3)
#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop &
         !$acc present(areaCell,layerThickness,activeTracers,maxLevelCell,nEdgesOnCell, &
         !$acc         edgesOnCell,cellsOnEdge,dcEdge,dvEdge,inSituThermalExpansionCoeff,&
         !$acc         inSituSalineContractionCoeff,kappaRediSfcTaper, &
         !$acc         k33) &
         !$acc private(cell1, cell2, dcEdgeInv, drhoDT, drhoDS, dTdx, dSdx, drhoDx) &
         !$acc async(1)
#else
         !$omp do schedule(runtime) private(cell1, cell2, dcEdgeInv, &
         !$omp& drhoDT, drhoDS, dTdx, dSdx, drhoDx)
#endif         
         do iEdge = 1, nEdges
            cell1 = cellsOnEdge(1, iEdge)
            cell2 = cellsOnEdge(2, iEdge)
            dcEdgeInv = 1.0_RKIND/dcEdge(iEdge)

            do k = 1, maxLevelEdgeTop(iEdge)
               drhoDT = -0.5_RKIND*(inSituThermalExpansionCoeff(k, cell1) + &
                                    inSituThermalExpansionCoeff(k, cell2))
               drhoDS = 0.5_RKIND*(inSituSalineContractionCoeff(k, cell1) + &
                                   inSituSalineContractionCoeff(k, cell2))
               dTdx = (activeTracers(indexTemperature, k, cell2) &
                       - activeTracers(indexTemperature, k, cell1)) &
                      *dcEdgeInv
               dSdx = (activeTracers(indexSalinity, k, cell2) &
                       - activeTracers(indexSalinity, k, cell1)) &
                      *dcEdgeInv
               drhoDx = drhoDT*dTdx + drhoDS*dSdx

               gradDensityEdge(k, iEdge) = drhoDx*rho_sw
            end do
         end do
#ifdef MPAS_OPENACC
         !$acc update host(gradDensityEdge) async(1)
         !$omp end master
#else      
         !$omp end do
#endif         

         nEdges = nEdgesArray(3)

         if (local_config_GM_lat_variable_c2) then
#ifdef MPAS_OPENACC
            !$omp master
            !$acc parallel loop &
            !$acc present(cellsOnEdge,maxLevelEdgeTop,BruntVaisalaFreqTop,layerThicknessEdge, &
            !$acc         cGMphaseSpeed) &
            !$acc private(cell1, cell2, sumN2, ltSum, countN2, BruntVaisalaFreqTopEdge) &
            !$acc async(1)
#else
            !$omp do schedule(runtime) private(cell1, cell2, sumN2, ltSum, countN2, BruntVaisalaFreqTopEdge)
#endif
            do iEdge = 1, nEdges
               cell1 = cellsOnEdge(1, iEdge)
               cell2 = cellsOnEdge(2, iEdge)
               sumN2 = 0.0
               ltSum = 0.0
               countN2 = 0

               do k = 2, maxLevelEdgeTop(iEdge)

                  BruntVaisalaFreqTopEdge = 0.5_RKIND*(BruntVaisalaFreqTop(k, cell1) + BruntVaisalaFreqTop(k, cell2))
                  BruntVaisalaFreqTopEdge = max(BruntVaisalaFreqTopEdge, 0.0_RKIND)

                  sumN2 = sumN2 + BruntVaisalaFreqTopEdge*layerThicknessEdge(k, iEdge)
                  ltSum = ltSum + layerThicknessEdge(k, iEdge)
                  countN2 = countN2 + 1

               end do

               if (countN2 > 0) cGMphaseSpeed(iEdge) = max(local_config_GM_min_phase_speed, sqrt(sumN2/ltSum)*ltSum/3.141592_RKIND)

            end do
#ifdef MPAS_OPENACC
            !$acc update host(cGMphaseSpeed) async(1)
            !$omp end master
#else      
            !$omp end do
#endif         
         else
#ifdef MPAS_OPENACC
            !$omp master
            !$acc parallel loop &
            !$acc present(cGMphaseSpeed) &
            !$acc async(1)
#else
            !$omp do schedule(runtime)
#endif            
            do iEdge = 1, nEdges
               cGMphaseSpeed(iEdge) = cgm_const_gwspeed
            end do
#ifdef MPAS_OPENACC
            !$acc update host(cGMphaseSpeed) async(1)
            !$omp end master
#else      
            !$omp end do
#endif         
         end if

         coeff = gravity / rho_sw
         nEdges = nEdgesArray( 3 )
         mxlvl = maxval(maxLevelEdgeTop)
#ifdef MPAS_OPENACC
	     !$omp master
	     !$acc parallel loop &
	     !$acc          present(gmStreamFuncTopOfEdge,maxLevelEdgeTop,cellsOnEdge,layerThicknessEdge, &
	     !$acc                  gradDensityEdge,kappaGMCell,kappaRediSfcTaper,BruntVaisalaFreqTop,cGMphaseSpeed, &
	     !$acc                  gmBolusKappa) &
	     !$acc          private(tridiagB,tridiagC,tridiagA,rightHandSide,cell1, cell2, k, BruntVaisalaFreqTopEdge, &
   	     !$acc                  kappaGMEdge, bTemp,rTemp, N, hi, ltdenom, ltem, ltem1, c) async(2) wait(1)
#else
         !$omp do schedule(runtime) private(cell1, cell2, k,  gradDensityTopOfEdge, &
         !$omp& kappaGMEdge, BruntVaisalaFreqTopEdge, N)
#endif         
         do iEdge = 1, nEdges
            hi = maxLevelEdgeTop(iEdge)
            cell1 = cellsOnEdge(1, iEdge)
            cell2 = cellsOnEdge(2, iEdge)

            ! Construct the tridiagonal matrix
            if (hi .GE. 3) then

               ! First row
               do k = 2, hi
                  gmStreamFuncTopOfEdge(k,iEdge) = 0.0_RKIND
                  ltem = layerThicknessEdge(k, iEdge)
                  ltem1 = layerThicknessEdge(k-1, iEdge)
                  ltdenom = 1.0 / (ltem1 + ltem)
                  c = 2.0_RKIND * cGMphaseSpeed(iEdge)**2
                  gradDensityTopOfEdge = (ltem1*gradDensityEdge(k - 1, iEdge) + &
                                       ltem*gradDensityEdge(k, iEdge))/(ltem1 + ltem)
                  kappaGMEdge = 0.5_RKIND*(kappaGMCell(k, cell1) + kappaGMCell(k, cell2))* &
                             min(kappaRediSfcTaper(k, cell1), kappaRediSfcTaper(k, cell2))
                  BruntVaisalaFreqTopEdge = 0.5_RKIND*(BruntVaisalaFreqTop(k, cell1) + BruntVaisalaFreqTop(k, cell2))
                  BruntVaisalaFreqTopEdge = max(BruntVaisalaFreqTopEdge, 0.0_RKIND)
               
                  if ( k > 2 ) then
                     tridiagA(k-2) = (c / ltem1) * ltdenom
                  endif
                  tridiagB(k-1) = - c / (ltem1 * ltem ) - BruntVaisalaFreqTopEdge
                  if ( k < maxLevelEdgeTop(iEdge) ) then              
                     tridiagC(k-1) = (c / ltem) * ltdenom
                  endif              
                  rightHandSide(k - 1) = gmBolusKappa(iEdge)*kappaGMEdge*coeff*gradDensityTopOfEdge
               end do
               
               ! Total number of rows
               N = hi - 1

               ! Call the tridiagonal solver
               call tridiagonal_solve(tridiagA, tridiagB, tridiagC, rightHandSide, bTemp,rTemp, &
                                      gmStreamFuncTopOfEdge(:, iEdge), N)  
            endif                
         end do
#ifdef MPAS_OPENACC
	     !$acc update host(gmStreamFuncTopOfEdge) async(2)
         !$omp end master
#else
         !$omp end do
#endif         

         nEdges = nEdgesArray(3)
         mxlvl = maxval(maxLevelEdgeTop)
         ! Compute normalGMBolusVelocity from the stream function
#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop collapse(2) &
         !$acc          present(maxLevelEdgeTop,normalGMBolusVelocity,gmStreamFuncTopOfEdge,layerThicknessEdge) &
         !$acc          async(2)
#else
         !$omp do schedule(runtime) private(k)
#endif          
         do iEdge = 1, nEdges
            do k = 1,mxlvl
               if ( k <=  maxLevelEdgeTop(iEdge) ) then
                  normalGMBolusVelocity(k, iEdge) = (gmStreamFuncTopOfEdge(k, iEdge) - gmStreamFuncTopOfEdge(k + 1, iEdge)) &
                                                 /layerThicknessEdge(k, iEdge)
               endif
            end do
         end do
#ifdef MPAS_OPENACC
         !$acc update host(normalGMBolusVelocity) async(2)
         !$omp end master
#else
         !$omp end do
#endif

         nCells = nCellsArray( 1 )
         mxlvl = size(gmStreamFuncTopOfCell,1)
  
#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop collapse(2) &
         !$acc          present(gmStreamFuncTopOfCell) async(1)
#else
         !$omp do schedule(runtime) private(k)
#endif      
         do iCell = 1, nCells
            do k = 1,mxlvl
               gmStreamFuncTopOfCell(k, iCell) = 0.0_RKIND
            end do
         end do   
#ifdef MPAS_OPENACC
         !$omp end master
#else
         !$omp end do
#endif

         ! Interpolate gmStreamFuncTopOfEdge to cell centers for visualization
#ifdef MPAS_OPENACC
         !$omp master
         !$acc parallel loop &
         !$acc          present(nEdgesOnCell,edgesOnCell,dcEdge,dvEdge, &
         !$acc                  maxLevelEdgeTop,gmStreamFuncTopOfEdge,gmStreamFuncTopOfCell) &
         !$acc          private(i, iEdge, areaEdge, k, rtmp) &
         !$acc          async(1)
#else
         !$omp do schedule(runtime) private(i, iEdge, areaEdge, k, rtmp)
#endif      
         do iCell = 1, nCells
            do i = 1, nEdgesOnCell(iCell)
               iEdge = edgesOnCell(i, iCell)

               areaEdge = 0.25_RKIND*dcEdge(iEdge)*dvEdge(iEdge)

               do k = 1, maxLevelEdgeTop(iEdge)
                  rtmp = 0.5_RKIND*(gmStreamFuncTopOfEdge(k, iEdge) + gmStreamFuncTopOfEdge(k + 1, iEdge))*areaEdge
                  gmStreamFuncTopOfCell(k, iCell) = gmStreamFuncTopOfCell(k, iCell) + rtmp
               end do
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
         !$acc          present(gmStreamFuncTopOfCell,areaCell) async(1)
#else
         !$omp do schedule(runtime)
#endif      
         do iCell = 1, nCells
            do k = 1,mxlvl
               gmStreamFuncTopOfCell(k, iCell) = gmStreamFuncTopOfCell(k,iCell) / areaCell(iCell)
            end do   
         end do
#ifdef MPAS_OPENACC
         !$acc update host(gmStreamFuncTopOfCell) async(1)
         !$omp end master
#else
         !$omp end do
#endif

         deallocate (rightHandSide)
         deallocate (tridiagA)
         deallocate (tridiagB)
         deallocate (tridiagC)
         deallocate(bTemp)
         deallocate(rTemp)

      end if !end config_use_GM

#ifdef MPAS_OPENACC
      !$omp master
      !$acc wait
  
      !$acc exit data delete(normalGMBolusVelocity, &
      !$acc       gradDensityEdge, slopeTriadUp,slopeTriadDown, &
      !$acc       cGMphaseSpeed, kappaGMCell, kappaRediSfcTaper, &
      !$acc       gmStreamFuncTopOfEdge,gmStreamFuncTopOfCell,gmBolusKappa, &
      !$acc       BruntVaisalaFreqTop, k33, displacedDensity, density, zMid, &
      !$acc       inSituThermalExpansionCoeff, inSituSalineContractionCoeff, indMLD, &
      !$acc       layerThicknessEdge)

      !$omp end master	  	  
#endif

      ! Deallocate scratch variables
      call mpas_deallocate_scratch_field(gradDensityEdgeField, .true.)
      call mpas_timer_stop('gm bolus velocity')

   end subroutine ocn_GM_compute_Bolus_velocity!}}}

!***********************************************************************
!
!  routine tridiagonal_solve
!
!> \brief   Solve the matrix equation Ax=r for x, where A is tridiagonal.
!> \details
!>  Solve the matrix equation Ax=r for x, where A is tridiagonal.
!>  A is an nxn matrix, with:
!>  a sub-diagonal, filled from 1:n-1 (a(1) appears on row 2)
!>  b diagonal, filled from 1:n
!>  c sup-diagonal, filled from 1:n-1  (c(1) apears on row 1)
!
!-----------------------------------------------------------------------
! mrp note:  This subroutine also appears in vmix and should really be put in the framework.
   !subroutine tridiagonal_solve(a,b,c,r,lo,hi,x,n) !{{{
   subroutine tridiagonal_solve(a,b,c,r,bTemp,rTemp,x,n) !{{{
#ifdef MPAS_OPENACC
   !$acc routine seq
#endif
      !-----------------------------------------------------------------
      !
      ! input variables
      !
      !-----------------------------------------------------------------

      integer, intent(in) :: n
      real(KIND=RKIND), dimension(n), intent(in) :: a, b, c, r
      real(KIND=RKIND), dimension(n), intent(inout) :: bTemp,rTemp

      !-----------------------------------------------------------------
      !
      ! output variables
      !
      !-----------------------------------------------------------------

      real(KIND=RKIND), dimension(n), intent(out) :: x

      !-----------------------------------------------------------------
      !
      ! local variables
      !
      !-----------------------------------------------------------------

      real (KIND=RKIND) :: m
      integer i

      ! Use work variables for b and r
      bTemp(1) = b(1)
      rTemp(1) = r(1)

      ! First pass: set the coefficients
      do i = 2, n
         m = a(i - 1)/bTemp(i - 1)
         bTemp(i) = b(i) - m*c(i - 1)
         rTemp(i) = r(i) - m*rTemp(i - 1)
      end do

      x(n+1) = rTemp(n)/bTemp(n)
      ! Second pass: back-substition
      do i = n - 1, 1, -1
         x(i+1) = (rTemp(i) - c(i)*x(i + 2))/bTemp(i)
      end do

   end subroutine tridiagonal_solve !}}}

!***********************************************************************
!
!  routine ocn_GM_init
!
!> \brief   Initializes ocean momentum horizontal pressure gradient
!> \details
!>  This routine initializes parameters required for the computation of the
!>  horizontal pressure gradient.
!
!-----------------------------------------------------------------------

   subroutine ocn_GM_init(domain, err)!{{{

      type(domain_type), intent(in) :: domain

      integer, intent(out) :: err !< Output: error flag

      real(kind=RKIND) :: avgCellDiameter, sqrtPiInv

      type(block_type), pointer :: block
      type(mpas_pool_type), pointer :: meshPool
      type(mpas_pool_type), pointer :: diagnosticsPool
      real(kind=RKIND), dimension(:), pointer :: areaCell

      integer :: iEdge
      integer, pointer :: nEdges
      integer, pointer :: nVertLevels
      real(kind=RKIND), dimension(:), pointer   :: fEdge, gmBolusKappa
      integer, dimension(:, :), pointer :: cellsOnEdge, edgesOncell

      err = 0

      if ((.not. config_use_GM) .and. (.not. config_use_Redi)) return

      block => domain%blocklist
      do while (associated(block))
         call mpas_pool_get_subpool(block%structs, 'mesh', meshPool)
         call mpas_pool_get_subpool(block%structs, 'diagnostics', diagnosticsPool)
         call mpas_pool_get_dimension(meshPool, 'nVertLevels', nVertLevels)
         call mpas_pool_get_dimension(meshPool, 'nEdges', nEdges)
         call mpas_pool_get_array(meshPool, 'cellsOnEdge', cellsOnEdge)
         call mpas_pool_get_array(diagnosticsPool, 'gmBolusKappa', gmBolusKappa)
         call mpas_pool_get_array(meshPool, 'areaCell', areaCell)
         call mpas_pool_get_array(meshPool, 'fEdge', fEdge)

         if (config_Redi_use_slope_taper) then
            slopeTaperFactor = 1.0_RKIND
         else
            slopeTaperFactor = 0.0_RKIND
         end if

         if (config_Redi_use_surface_taper) then
            if (config_AM_mixedLayerDepths_enable .and. config_AM_mixedLayerDepths_Dthreshold) then
               sfcTaperFactor = 1.0_RKIND
            else
               call mpas_log_write('Redi Surface tapering requires MLD AM enabled with dThresh option selected.', &
                                   MPAS_LOG_CRIT)
               err = 1
               call mpas_dmpar_finalize(domain%dminfo)
            end if
         else
            sfcTaperFactor = 0.0_RKIND
         end if

         RediGMinitValue = 1.0_RKIND
         if (config_GM_closure == 'constant') then
            !$omp do schedule(runtime)
            do iEdge = 1, nEdges
               gmBolusKappa(iEdge) = config_GM_kappa
            end do
            !$omp end do
            RediGMinitValue = 1.0_RKIND
         else if (config_GM_closure == 'N2_dependent') then
            local_config_GM_lat_variable_c2 = .true.
            local_config_GM_kappa_lat_depth_variable = .true.

            RediGMinitValue = 0.0_RKIND
            ! for N2 dependence, we still assign Kappa as a constant.
            !$omp do schedule(runtime)
            do iEdge = 1, nEdges
               gmBolusKappa(iEdge) = config_GM_kappa
            end do
            !$omp end do
         else
            call mpas_log_write('Invalid choice of config_GM_closure.', MPAS_LOG_CRIT)
            err = 1
            call mpas_dmpar_finalize(domain%dminfo)
         end if

         ! Add resolution taper
         if (config_eddying_resolution_taper == 'none') then
            ! Nothing to do, as we just keep the same assignment as above.
         else if (config_eddying_resolution_taper == 'ramp') then
            sqrtPiInv = 1.0_RKIND/sqrt(pii)
            !$omp do schedule(runtime) private( avgCellDiameter)
            do iEdge = 1, nEdges
               avgCellDiameter = sqrtPiInv*(sqrt(areaCell(cellsOnEdge(1, iEdge))) + sqrt(areaCell(cellsOnEdge(2, iEdge))))
               if (avgCellDiameter <= config_eddying_resolution_ramp_min) then
                  gmBolusKappa(iEdge) = 0.0_RKIND
               else if (avgCellDiameter >= config_eddying_resolution_ramp_max) then
                  ! do nothing, i.e. gmBolusKappa(iEdge) remains unchanged.
               else
                  gmBolusKappa(iEdge) = gmBolusKappa(iEdge) &
                                        /(config_eddying_resolution_ramp_max - config_eddying_resolution_ramp_min) &
                                        *(avgCellDiameter - config_eddying_resolution_ramp_min)
               end if
            end do
            !$omp end do
         else
            call mpas_log_write('Invalid choice of config_eddying_resolution_taper.', MPAS_LOG_CRIT)
            err = 1
            call mpas_dmpar_finalize(domain%dminfo)
         end if

         block => block%next
      end do
   end subroutine ocn_GM_init!}}}

!***********************************************************************

end module ocn_gm

!|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
! vim: foldmethod=marker
