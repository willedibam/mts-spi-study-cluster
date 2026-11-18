import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False, "lines.linewidth": 1.7, "lines.markersize": 2.7,
    "figure.constrained_layout.use": True, "figure.dpi": 180, "savefig.dpi": 300, "savefig.bbox": "tight"})
