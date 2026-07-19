#!/usr/bin/env python3
# Generic smoke plot template.
# Customize x/y to the project's actual quantities.

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("docs/harness/plots")
OUT.mkdir(parents=True, exist_ok=True)

# Replace with real project data.
x = np.linspace(0.0, 1.0, 200)
y = np.exp(-x) * np.sin(8 * np.pi * x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("x [project units]")
ax.set_ylabel("diagnostic quantity [project units]")
ax.set_title("SMOKE diagnostic: replace with project-specific quantity")
ax.grid(True, alpha=0.3)

png = OUT / "smoke_plot.png"
fig.savefig(png, dpi=160, bbox_inches="tight")

meta = {
    "category": "SMOKE",
    "script": str(Path(__file__)),
    "output": str(png),
    "note": "Template plot only. Replace with project-specific diagnostic before using as evidence.",
}
(OUT / "smoke_plot_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print(f"Wrote {png}")
