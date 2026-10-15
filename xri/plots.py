"""Static diagnostic projections; no surface extraction or data modification."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def diagnostics_plot(density, axes, sensitivity, models, destination):
    x, y, z = axes
    fig, panels = plt.subplots(2, 3, figsize=(14, 8))
    projections = (
        (density.sum(axis=0).T, (y[0], y[-1], z[0], z[-1]), "Y", "Z"),
        (density.sum(axis=2), (y[0], y[-1], x[0], x[-1]), "Y", "X"),
        (density.sum(axis=1), (z[0], z[-1], x[0], x[-1]), "Z", "X"),
    )
    for ax, (image, extent, horizontal, vertical) in zip(panels[0], projections):
        im = ax.imshow(image, extent=extent, origin="lower", aspect="auto", cmap="magma")
        ax.set(xlabel=horizontal+" (mm)", ylabel=vertical+" (mm)", title="Provisional sum projection")
        fig.colorbar(im, ax=ax)
    panels[1, 0].plot(x, sensitivity)
    panels[1, 0].set(xlabel="Sample X (mm)", ylabel="Operator sensitivity", title="Full X grid; no support cropping")
    for key in list(models)[:4]:
        model = models[key]
        obs = model["observations"]
        xx = np.array([o["x"] for o in obs])
        panels[1, 1].plot(xx, [o["row"] for o in obs], "o", markersize=3)
        panels[1, 1].plot(xx, np.polyval(model["row_line"], xx), label=key)
    panels[1, 1].legend(fontsize=7)
    panels[1, 1].set(xlabel="Foil X (mm)", ylabel="Detector row (pixels)", title="Selected fitted trajectories")
    panels[1, 2].plot(x, density.sum(axis=(1, 2)))
    panels[1, 2].set(xlabel="Sample X (mm)", ylabel="Relative intensity sum", title="Depth profile")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
