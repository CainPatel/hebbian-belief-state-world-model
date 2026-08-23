"""Generate notebooks/belief_heatmaps.ipynb and notebooks/sigma_decay.ipynb (thin wrappers over hbwm.viz / results)."""

from pathlib import Path

import nbformat as nbf

HEAT = [
    "# Belief heatmaps\nWatch `belief_map` for each object over an episode. Set RUN_DIR to a headline run.",
    "RUN_DIR = 'runs/study1/bdh_g100_lr0.001/seed0'\nDATA = 'data/grid9'\nEPISODE = 0",
    "from hbwm.viz.heatmaps import render_episode\nframes = render_episode(RUN_DIR, DATA, episode=EPISODE)\nprint(len(frames), frames[0].parent)",
    "from IPython.display import Image, display\nfor p in frames[::8]:\n    display(Image(filename=str(p)))",
]
DECAY = [
    "# Sigma decay curves and top-k edges\nReads runs/study1/results/*.json written by hbwm.probes.evaluate.",
    "import json\nfrom pathlib import Path\nR = Path('runs/study1/results')\nh2 = json.loads((R/'h2.json').read_text())\nh4 = json.loads((R/'h4.json').read_text())",
    "import matplotlib.pyplot as plt\nfrom hbwm.probes.eligibility import BUCKET_NAMES\nfor stem, r in h2.items():\n    plt.plot([r['values'].get(b) for b in BUCKET_NAMES], marker='o', label=stem)\nplt.xticks(range(6), BUCKET_NAMES); plt.legend(); plt.ylabel('probe acc'); plt.show()",
    "for stem, r in h4.items():\n    print(stem, 'median k90 =', r['median_k90'], 'strong', r['strong'], 'weak', r['weak'])",
]


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c) if c.startswith("#") else nbf.v4.new_code_cell(c) for c in cells]
    Path(path).write_text(nbf.writes(nb))


if __name__ == "__main__":
    build(HEAT, "notebooks/belief_heatmaps.ipynb")
    build(DECAY, "notebooks/sigma_decay.ipynb")
    print("wrote notebooks/")
