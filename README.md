# pybioimage

## Installation

To install ``pybioimage`` run the following command.

````
pip install pybioimage
````

## Usage

The package is divided into submodules for FRAP, vertex enrichment, and cell 
aggregation analysis. Each of these submodules exposes an ``Analyzer`` class,
which constitutes the main class to use. These classes are instantiated with a
path pointing to the respective image. In order to make 
that as simple as possible, there is a convenience function in the ``utils`` 
module called ``find_files()``. This function searches for and returns all 
files from a given path fulfilling certain requirements. You can use this to 
find all files that are supposed to be analyzed.

````python
from pybioimage.utils import find_files


# Find all TIFF files in the 'data' folder but ignores files and folders 
# starting with an underscore '_'. 
find_files("data", pattern=".*\\.tiff$", ignore="^_.*")
````

Using regular expression, you can further finetune which files to return and 
then use those to instantiate your analyzer.
