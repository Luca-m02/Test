import os
import json
import atexit
import numpy as np
from datetime import datetime


def _to_py(o):
    """Rende serializzabili in JSON numpy, torch, Path, ecc."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "shape") and hasattr(o, "dtype"):   # tensori torch
        return o.detach().cpu().tolist() if hasattr(o, "detach") else str(o)
    return str(o)


class Summary:
    def __init__(self, out_dir, autosave=True):
        self.out_dir = out_dir
        self.data = {"start": datetime.now().isoformat(timespec="seconds")}
        os.makedirs(out_dir, exist_ok=True)
        if autosave:
            atexit.register(self.save)   # salva anche se lo script si interrompe

    def add(self, key, value):
        """Aggiunge o sovrascrive un valore. Con '/' crei sezioni annidate."""
        node = self.data
        *parents, last = key.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = value

    def add_many(self, prefix="", **kwargs):
        for k, v in kwargs.items():
            self.add(f"{prefix}/{k}" if prefix else k, v)

    def append(self, key, value):
        """Aggiunge a una lista (utile per metriche per epoca)."""
        node = self.data
        *parents, last = key.split("/")
        for p in parents:
            node = node.setdefault(p, {})
        node.setdefault(last, []).append(value)

    def add_array_info(self, key, arr):
        """Salva shape, dtype, min, max, media di un array o tensore."""
        a = np.asarray(arr)
        self.add(key, {"shape": list(a.shape), "dtype": str(a.dtype),
                       "min": a.min(), "max": a.max(), "mean": a.mean()})

    def save(self, filename="summary.json"):
        self.data["end"] = datetime.now().isoformat(timespec="seconds")
        path = os.path.join(self.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, default=_to_py)
        return path