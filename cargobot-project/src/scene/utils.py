from manipulation.utils import AddPackagePaths
import os

def ConfigureParser(parser):
    """Add the manipulation/package.xml index to the given Parser."""
    package_xml = os.path.join(os.path.dirname(__file__), "models/package.xml")
    parser.package_map().AddPackageXml(filename=package_xml)
    AddPackagePaths(parser)