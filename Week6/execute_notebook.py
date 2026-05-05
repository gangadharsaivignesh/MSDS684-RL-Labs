"""Execute Lab_PolicyGradient.ipynb in-place to populate cell outputs."""
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent / 'Lab_PolicyGradient.ipynb'
nb = nbformat.read(NB, as_version=4)

t0 = time.time()
print(f'Executing {NB.name} — {len(nb.cells)} cells…')
client = NotebookClient(
    nb,
    timeout=3600,                     # 60 min cell timeout (sweep cells need it)
    kernel_name='python3',
    resources={'metadata': {'path': str(NB.parent)}},
)
client.execute()

nbformat.write(nb, NB)
print(f'Done in {time.time() - t0:.1f}s. Saved to {NB}')
