set(iPIC_resources
  DebyeScaleTurbulence.inp
  DebyeScaleTurbulenceRestart.inp
  Magnetosphere2D.inp
  MagnetotailReconEngParticle.inp
  NullPoints.inp
  Ram1TestParticle.inp
  ScalingTestGEM3D.inp
  storePersistenceDiagram.py
  TaylorGreen.inp
  testGEM2D_NoHDF5.inp
  testGEM2Dsmall.inp
  testGEM3Dsmall.inp
  testMagnetosphere2Dsmall.inp
  testMagnetosphere2DsmallNBCIO.inp
  testMagnetosphere3Dsmall.inp
  )

install(FILES ${iPIC_resources}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/ipic3d/inputfiles
  )
