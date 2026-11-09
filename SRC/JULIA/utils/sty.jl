using PyPlot

COLORS = ["#5b2e10", "#b7a294", "#bd7984", "#e8cbcb", "#f2b361", "#fae4c6", "#90423b", "#ffffff"]

# -------------------------------
# Text & fonts
# -------------------------------
rc("text", usetex=true)
rc("font", family="serif")

rc("text.latex", preamble=
	"\\usepackage{amsmath}
	 \\usepackage{amsfonts}"
)

rc("font", size=12)
rc("legend", fontsize=10)
rc("axes", titlesize=12)

# -------------------------------
# Axes appearance
# -------------------------------
rc("axes", xmargin=0.1)
rc("axes", ymargin=0.1)
rc("axes", linewidth=0.5)

rc("axes.formatter", limits=[-3, 3]) 

# -------------------------------
# Ticks
# -------------------------------
rc("xtick.major", size=5, width=0.5)
rc("xtick.minor", size=2, width=0.5, visible=true)

rc("ytick.major", size=5, width=0.5)
rc("ytick.minor", size=2, width=0.5, visible=true)

# -------------------------------
# Lines & markers
# -------------------------------
rc("lines", markeredgecolor=COLORS[end])
rc("lines", markeredgewidth=0.2)
rc("lines", dash_capstyle="round")
rc("lines", solid_capstyle="round")

# -------------------------------
# Legend
# -------------------------------
rc("legend", frameon=true)
rc("legend", fancybox=false)
rc("legend", framealpha="none")
rc("legend", facecolor="1")
rc("legend", edgecolor="0")
rc("legend", borderpad=0.5)
rc("patch", linewidth=0.5)

rc("legend", labelspacing=0.8)
rc("legend", columnspacing=1)
rc("legend", handlelength=1)
rc("legend", handletextpad=0.5)
rc("legend", handleheight=0)
rc("legend", borderaxespad=0)

# -------------------------------
# Figure & saving
# -------------------------------
rc("savefig", dpi=300)
rc("savefig", format="pdf")
rc("savefig", bbox="tight")
rc("savefig", pad_inches=0.05)

# -------------------------------
# Paths (PDF robustness)
# -------------------------------
rc("path", simplify=true)
rc("path", simplify_threshold=0.6)

