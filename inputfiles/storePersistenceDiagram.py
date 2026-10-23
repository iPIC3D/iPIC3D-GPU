# pylint: disable=invalid-name, missing-docstring, import-error
# --------------------------------------------------------------

from paraview import coprocessing, simple
from paraview.simple import servermanager as sm
import os
#pv_case    = os.environ['PV_CASE'] if 'PV_CASE' in os.environ else 'viz'
pv_address = os.environ['PV_ADDRESS'] if 'PV_ADDRESS' in os.environ else 'localhost'
pv_port    = os.environ['PV_PORT'] if 'PV_PORT' in os.environ else 22222
print('pv_address', pv_address, pv_port)

# Global timestep output options
timeStepToStartOutputAt = 0
forceOutputAtFirstCall = False

# Global screenshot output options
imageFileNamePadding = 0
rescale_lookuptable = False

# Whether or not to request specific arrays from the adaptor.
requestSpecificArrays = False

# a root directory under which all Catalyst output goes
rootDirectory = ""

# makes a cinema D index table
make_cinema_table = False

# --------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.6.0
# --------------------------------------------------------------

# ----------------------- CoProcessor definition -----------------------


def CreateCoProcessor():
    def _CreatePipeline(coprocessor, datadescription):
        particles = coprocessor.CreateProducer(datadescription, "particles")
        ppdwriter0 = sm.writers.XMLPImageDataWriter(Input=particles)
        ## create a producer from a simulation input
        #particles = coprocessor.CreateProducer(datadescription, "particles")

        ## compute the magnitude of the magnetic field
        #calc = simple.Calculator(Input=particles)
        #calc.Function = "mag(B)"

        ## store a compressed version inside a Cinema Database
        #cineWriter0 = simple.TTKCinemaWriter(Input=calc, DatabasePath="data/tcomp.cdb")
        #cineWriter0.ForwardInput = False
        #cineWriter0.Storeas = 2
        #cineWriter0.ScalarField = "Result"

        ## trigger this branch of the pipeline
        #ppdwriter0 = sm.writers.XMLPImageDataWriter(Input=cineWriter0)
        #coprocessor.RegisterWriter(
        #    ppdwriter0, filename="data/catalyst/tmp.pvtp", freq=10, paddingamount=0,
        #)

        ## annotate data
        #arrEd0 = simple.TTKArrayEditor(Target=calc, Source=None)
        #arrEd0.TargetAttributeType = "Field Data"
        #arrEd0.DataString = "ScalarField,mag(B)"

        ## normalize scalar field
        #sfn0 = simple.TTKScalarFieldNormalizer(Input=arrEd0)
        #sfn0.ScalarField = "Result"

        ## generate a persistence diagram
        #pDiag = simple.TTKPersistenceDiagram(Input=sfn0)
        #pDiag.ScalarField = "Result"
        #pDiag.EmbedinDomain = 0

        ## store inside a Cinema Database
        #cineWriter1 = simple.TTKCinemaWriter(
        #    Input=pDiag, DatabasePath="data/pdiags.cdb"
        #)
        #cineWriter1.ForwardInput = False

        ## trigger this branch of the pipeline
        #ppdwriter1 = sm.writers.XMLPUnstructuredGridWriter(Input=cineWriter1)
        #coprocessor.RegisterWriter(
        #    ppdwriter1, filename="data/catalyst/tmp.pvtp", freq=1, paddingamount=0,
        #)

        ## extract the z component of B (Bz)
        #extrComp = simple.ExtractComponent(Input=particles)
        #extrComp.InputArray = ["POINTS", "B"]
        #extrComp.Component = "Z"

        ## annotate data
        #arrEd1 = simple.TTKArrayEditor(Target=extrComp, Source=None)
        #arrEd1.TargetAttributeType = "Field Data"
        #arrEd1.DataString = "ScalarField,Bz"

        ## normalize scalar field
        #sfn1 = simple.TTKScalarFieldNormalizer(Input=arrEd1)
        #sfn1.ScalarField = "Result"

        ## generate the persistence diagram
        #pDiag1 = simple.TTKPersistenceDiagram(Input=sfn1)
        #pDiag1.ScalarField = "Result"
        #pDiag1.EmbedinDomain = 0

        ## store inside a Cinema Database
        #cineWriter2 = simple.TTKCinemaWriter(
        #    Input=pDiag1, DatabasePath="data/pdiags.cdb"
        #)
        #cineWriter2.ForwardInput = False

        ## trigger this branch of the pipeline
        #ppdwriter2 = sm.writers.XMLPUnstructuredGridWriter(Input=cineWriter2)
        #coprocessor.RegisterWriter(
        #    ppdwriter2, filename="data/catalyst/tmp.pvtp", freq=1, paddingamount=0,
        #)

    class CoProcessor(coprocessing.CoProcessor):
        def CreatePipeline(self, datadescription):
            self.Pipeline = _CreatePipeline(self, datadescription)

    coprocessor = CoProcessor()
    # these are the frequencies at which the coprocessor updates.
    freqs = {"particles": [1, 10]}
    coprocessor.SetUpdateFrequencies(freqs)
    if requestSpecificArrays:
        arrays = []
        coprocessor.SetRequestedArrays("particles", arrays)
    coprocessor.SetInitialOutputOptions(timeStepToStartOutputAt, forceOutputAtFirstCall)

    if rootDirectory:
        coprocessor.SetRootDirectory(rootDirectory)

    if make_cinema_table:
        coprocessor.EnableCinemaDTable()

    return coprocessor


# --------------------------------------------------------------
# Global variable that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

# --------------------------------------------------------------
# Enable Live-Visualizaton with ParaView and the update frequency
coprocessor.EnableLiveVisualization(True, 1)

# ---------------------- Data Selection method ----------------------


def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)


# ------------------------ Processing method ------------------------


def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription)

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(
        datadescription,
        rescale_lookuptable=rescale_lookuptable,
        image_quality=0,
        padding_amount=imageFileNamePadding,
    )

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, pv_address, pv_port)
