import os
import re
import pickle
import math

__all__ = [
	'load_file',
	'folder_names',
	'create_folder_if_missing',
	'dict_to_str',
	'save_graph',
	'format_value',
	'format_sign'
]

def load_file(folder, expression):
	"""
	Loads files from a specified folder that match a given regular expression pattern.
	
	Args:
		folder (str): The path to the folder containing the files.
		expression (str): The regular expression pattern to match file names against.
		
	Returns:
		tuple: A tuple containing two lists:
			- vals (list): A list of dictionaries where each dictionary contains the matched groups 
			  from the regular expression with their values converted to floats.
			- names (list): A list of file names that matched the regular expression.
	"""
	vals = []
	names = []

	for file in sorted(os.scandir(folder), key=lambda e: e.name):
		try:
			m = re.match(expression, file.name)
			d = m.groupdict()
		except:
			pass
		else:
			for key in d.keys():
				d[key] = float(d[key])

			vals.append(d)
			names.append(file.name)

	return vals, names

def folder_names(folder):
	"""
	Retrieves a sorted list of directory names within a specified folder.
	
	Args:
		folder (str): The path to the folder to search for directories.
		
	Returns:
		list: A sorted list of directory names within the specified folder.
	"""
	return sorted([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def create_folder_if_missing(path):
	"""
	Creates a folder at the specified path if it does not already exist.
	
	Args:
		path (str): The path where the folder should be created.
	"""
	if not os.path.exists(path):
		os.makedirs(path)

def dict_to_str(d):
	"""
	Converts a dictionary to a string representation where each key-value pair is 
	formatted as 'key=value' and pairs are separated by underscores.
	
	Args:
		d (dict): The dictionary to convert.
		
	Returns:
		str: The string representation of the dictionary.
	"""
	string = ""
	for item in d:
		string += item + "=" + str(d[item]) + "_"

	return string[:-1]

def save_graph(fig, path, name):
	"""
	Saves a graph figure to a specified path with a given name, ensuring the directory exists.
	
	Args:
		fig (matplotlib.figure.Figure): The figure object to save.
		path (str): The directory path where the figure should be saved.
		name (str): The name to use for the saved figure file.
	"""
	create_folder_if_missing(path)
	fig.savefig(os.path.join(path, name) + ".")

def format_value(val: float, unc: float, par=False) -> str:

	if unc == 0.0:
		return f"{val:.4f}"

	digits = -int(math.floor(math.log10(round(unc, -int(math.floor(math.log10(abs(unc))))))))
	
	rounded_val = round(val, digits)
	rounded_unc = round(unc, digits)

	if abs(rounded_val) < 1e-10:
		return "0"

	order_val = int(math.floor(math.log10(abs(rounded_val)))) if rounded_val != 0 else 0
	order_unc = -digits
	scale = 10.0 ** (-order_unc)

	cond_1 = order_unc < -3
	cond_2 = order_val < -3
	cond_3 = order_val <= order_unc
	cond_4 = order_val >= order_unc + 3

	if cond_1 and (cond_2 or cond_3):
		val_scaled = int(round(rounded_val * scale))
		unc_scaled = int(round(rounded_unc * scale))
		return f"({val_scaled} \\pm {unc_scaled}) \\cdot 10^{{{order_unc}}}"

	elif cond_1 and (not cond_2 or cond_4):
		return f"{rounded_val:.{digits}f} \\pm {int(round(rounded_unc * scale))} \\cdot 10^{{{order_unc}}}"

	if digits <= 0:
		return f"{int(round(rounded_val))} \\pm {int(round(rounded_unc))}"
	else:
		return f"{rounded_val:.{digits}f} \\pm {rounded_unc:.{digits}f}"

def format_sign(val: float) -> str:
	return "-" if val < 0 else "+"
