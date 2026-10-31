"""Export saved training sources from the mounted Kaggle artifact dataset. No training."""
from pathlib import Path
import hashlib
import json
import zipfile

BASE = Path("/kaggle/input/datasets/angelisnguele/azfgrvv/cmi_hybrid102_complete_folds1_to5")
DEST = Path("/kaggle/working/cmi_training_sources.zip")
required = {
    "src/spatial/train_cmi.py": BASE / "initial_ensemble/spatial/train_cmi.py",
    "src/phases/train_cmi.py": BASE / "initial_ensemble/phases/train_cmi.py",
    "src/heading/train_cmi.py": BASE / "initial_ensemble/heading/train_cmi.py",
    "src/composite102/train_cmi.py": BASE / "composite102/train_cmi.py",
}
for source in required.values():
    if not source.is_file():
        raise FileNotFoundError(source)
manifest = []
with zipfile.ZipFile(DEST, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for destination, source in required.items():
        archive.write(source, destination)
        manifest.append({
            "path": destination,
            "source_path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    for family in ["spatial", "phases", "heading"]:
        source = BASE / "initial_ensemble" / family / "configuration.json"
        if source.is_file():
            archive.write(source, f"src/{family}/configuration.json")
    for filename in ["configuration.json", "classes102.csv"]:
        source = BASE / "composite102" / filename
        if source.is_file():
            archive.write(source, f"src/composite102/{filename}")
    archive.writestr("source_manifest.json", json.dumps(manifest, indent=2))
print(f"Exported {len(required)} saved training scripts: {DEST}")
print("No training. No weights or competition data exported.")
from IPython.display import FileLink, display
display(FileLink(str(DEST)))
