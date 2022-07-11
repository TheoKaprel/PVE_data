import click
from itk import RTK as rtk
import numpy as np

sigma0pve_default = 0.9008418065898374
alphapve_default = 0.025745123547513887



CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings = CONTEXT_SETTINGS)
@click.option('--geom')
def testRTKforward(geom):

    xmlReader = rtk.ThreeDCircularProjectionGeometryXMLFileReader.New()
    xmlReader.SetFilename(geom)
    xmlReader.GenerateOutputInformation()
    geometry = xmlReader.GetOutputObject()

    angles = geometry.GetGantryAngles()

    print(len(angles))


if __name__=='__main__':
    testRTKforward()