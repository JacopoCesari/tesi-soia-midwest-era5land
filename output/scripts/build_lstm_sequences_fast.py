"""Development entry point for the canonical complete-window sequence builder.

Sequence implementation is maintained exclusively in scripts/06_train_lstm_model.py.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("recurrent", ROOT / "scripts/06_train_lstm_model.py")
recurrent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recurrent)

if __name__ == "__main__":
    for width in [30, 10, 5]:
        recurrent.build_sequence_caches(width)
