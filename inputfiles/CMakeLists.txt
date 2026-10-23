set(iPIC_resources
  turbulence/DebyeScaleTurbulence.inp
  turbulence/DebyeScaleTurbulenceRestart.inp
  turbulence/TaylorGreen.inp
  magnetosphere/Magnetosphere2D.inp
  magnetosphere/testMagnetosphere2Dsmall.inp
  magnetosphere/testMagnetosphere2DsmallNBCIO.inp
  magnetosphere/testMagnetosphere3Dsmall.inp
  magneticReconnection/MagnetotailReconEngParticle.inp
  magneticReconnection/ScalingTestGEM3D.inp
  magneticReconnection/testGEM2D_NoHDF5.inp
  magneticReconnection/testGEM2Dsmall.inp
  magneticReconnection/testGEM3Dsmall.inp
  storePersistenceDiagram.py
  NullPoints.inp
  Ram1TestParticle.inp
  )
  
install(FILES ${iPIC_resources}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/ipic3d/inputfiles
  )
