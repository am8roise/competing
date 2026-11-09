from pathlib import Path
import os

__all__ = [
	'DATA_FOLDER',
	'GRAPHS_FOLDER',
	'COMPUTE_FOLDER'
]

# Define the root folder as the third parent directory of the current file
ROOT_FOLDER = Path(__file__).resolve().parents[4]

# Define the path to the data folder where input data files are stored
DATA_FOLDER = os.path.join(ROOT_FOLDER, "DATA")

# Define the path to the graphs folder where output graph files will be saved
GRAPHS_FOLDER = os.path.join(ROOT_FOLDER, "GRAPHS")

# Define the path to the graphs folder where output analyzed data files will be saved
COMPUTE_FOLDER = "ANALYZED"
