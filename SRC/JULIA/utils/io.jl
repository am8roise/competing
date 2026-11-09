using PyCall
using PyPlot
using Statistics

py"""
def create_folder_if_missing(path):
	import os

	if not os.path.exists(path):
		os.makedirs(path)
"""

py"""
def dict_to_str(d):

	string = ""
	for item in d:
		string += item + "=" + str(d[item]) + "_"

	return string[:-1]
"""

py"""
def load_file(folder, expression):
	import os
	import re

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
"""

py"""
def folder_names(folder):
	import os

	return sorted([name+"/" for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])
"""

using Printf

function format_value(val::Float64, unc::Float64)

	digits = -floor(Int, log10(round(unc, digits=-floor(Int, log10(unc)))))
	
	rounded_val = round(val, digits=digits)
	rounded_unc = round(unc, digits=digits)

	if (abs(rounded_val) < 1E-10)
		return @sprintf("0")
	end

	order_val = floor(Int, log10(abs(rounded_val)))
	order_unc = -digits
	scale = 10. ^ (-order_unc)

	cond_1 = order_unc < -3
	cond_2 = order_val < -3
	cond_3 = order_val ≤ order_unc + 1
	cond_4 = order_val ≥ order_unc + 3

	if cond_1 & (cond_2 | cond_3)
		return @sprintf("(%d \\pm %d) \\cdot 10^{%d}", round(rounded_val * scale), round(rounded_unc * scale), order_unc)

	elseif cond_1 & (!cond_2 | cond_4)
		return @sprintf("%.*f \\pm %d \\cdot 10^{%d}", digits, rounded_val, round(rounded_unc * scale), order_unc)

	end

	if digits ≤ 0
		return @sprintf("%d \\pm %d", round(rounded_val), round(rounded_unc))

	else
		return @sprintf("%.*f \\pm %.*f", digits, rounded_val, digits, rounded_unc)

	end
end

format_sign(val::Float64) = val < 0 ? "-" : "+"