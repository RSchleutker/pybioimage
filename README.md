# Pybioimage

## Installation

### From this repository

Make sure to meet the following requirements:

* You have *git* installed on your local computer in order to clone this 
  repository.
* You have a Python installation with Python 3.11 or higher.

To install ``pybioimage`` run the following command.

````
pip install pybioimage@git+https://zivgitlab.uni-muenster.de/r_schl17/pybioimage
````

## Usage

The package is divided into submodules for FRAP, vertex enrichment, and cell 
aggregation analysis. Each of these submodules exposes an ``Analyzer`` class,
which constitutes the main class for you to use. These classes are 
instantiated with a path pointing to the respective image. In order to make 
that as simple as possible, there is a convenience function in the ``utils`` 
module called ``find_files()``. This function searches for and return all 
files from a given path fulfilling certain requirements. You can use this to 
find all files that are supposed to be analyzed.

````python
from pybioimage.utils import find_files


# Find all TIFF files in the 'data' folder but ignores files and folders 
# starting with an underscore '_'. 
find_files("data", pattern=".*\\.tiff$", ignore="^_.*")
````

Using regular expression, you can further finetune which files to return and 
then use those to instantiate your analyzer. Doing so will also create a 
dictionary of metadata from the parent folder name of the file using the 
``str2dict()`` function. This function breaks up the folder name, which is 
commonly used by biologists to store important data.

````python
from pybioimage.utils import str2dict


str2dict("date-20220215_prot-m6_allele-wt_expr-homo_emb-1")
# {'Date': '20220215', 'Prot': 'M6', 'Allele': 'Wt', 'Expr': 'Homo', 'Emb': '1'}
````
