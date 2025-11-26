import pickle
from pathlib import Path
p = Path('data/processed/feature_names.pkl')
if not p.exists():
    print('MISSING')
    raise SystemExit(1)
with open(p,'rb') as f:
    feat = pickle.load(f)
try:
    print(len(feat))
except Exception:
    print('CANNOT_COUNT')
