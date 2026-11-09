import matplotlib
import matplotlib.pyplot

__all__ = [
	'COLORS'
]

COLORS = ["#5b2e10", "#b7a294", "#bd7984", "#e8cbcb", "#f2b361", "#fae4c6", "#90423b", "#ffffff"]

matplotlib.pyplot.rcParams.update({
	# -------------------------------
	# Text & fonts
	# -------------------------------
	'text.usetex': True,
	'font.family': 'serif',

	'text.latex.preamble': """\\usepackage{amsmath}
							  \\usepackage{mathtools}
							  \\usepackage{physics}
							  \\usepackage{amsfonts}""",

	'font.size': 12,
	'legend.fontsize': 10,

	# -------------------------------
	# Axes appearance
	# -------------------------------
	'axes.xmargin': 0.1,
	'axes.ymargin': 0.1,
	'axes.linewidth': 0.5,

	# -------------------------------
	# Ticks
	# -------------------------------
	'xtick.major.size': 5,
	'xtick.major.width': 0.5,

	'xtick.minor.size': 2,
	'xtick.minor.width': 0.5,
	'xtick.minor.visible': True,

	'ytick.major.size': 5,
	'ytick.major.width': 0.5,
	
	'ytick.minor.size': 2,
	'ytick.minor.width': 0.5,
	'ytick.minor.visible': True,

	# -------------------------------
	# Lines & markers
	# -------------------------------
	'lines.markeredgecolor': COLORS[-1],
	'lines.markeredgewidth': 0.2,
	'lines.dash_capstyle': 'round',
	'lines.solid_capstyle': 'round',

	# -------------------------------
	# Legend
	# -------------------------------
	'legend.frameon': True,
	'legend.fancybox': False,
	'legend.framealpha': 'none',
	'legend.facecolor': '1',
	'legend.edgecolor': '0',
	'legend.borderpad': 0.5,
	'patch.linewidth': 0.5,

	'legend.labelspacing': 0.8,
	'legend.columnspacing': 1,
	'legend.handlelength': 1,
	'legend.handletextpad': 0.5,
	'legend.handleheight': 0,
	'legend.borderaxespad': 0,

	# -------------------------------
	# Figure & saving
	# -------------------------------
	'savefig.dpi': 300,
	'savefig.format': 'pdf',
	'savefig.bbox': 'tight',
	'savefig.pad_inches': 0.05,

	# -------------------------------
	# Paths (PDF robustness)
	# -------------------------------
	'path.simplify': True,
	'path.simplify_threshold': 0.6

})

