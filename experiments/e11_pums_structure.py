"""E11. ACS PUMS structure check (one state), for choosing the evaluation population.

Downloads the 2024 1-year person file for a state (default Rhode Island, 44), and reports
the number of adults 25-64 per household (the cluster size r) and between-PUMA spread of the
employment rate (ESR in {1,2}). Rhode Island: 92% of households have 1-2 such adults;
PUMA employment-rate sd = 0.021.
  python experiments/e11_pums_structure.py [state_abbrev_lower]
Writes results/pums_structure_<state>.json.
"""
import io
import json
import sys
import urllib.request
import zipfile

import pandas as pd

from _common import DATA, RESULTS

st = sys.argv[1] if len(sys.argv) > 1 else 'ri'
path = DATA / f'csv_p{st}.zip'
if not path.exists():
    DATA.mkdir(exist_ok=True)
    url = f'https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_p{st}.zip'
    path.write_bytes(urllib.request.urlopen(url).read())
with zipfile.ZipFile(path) as z:
    name = [n for n in z.namelist() if n.endswith('.csv')][0]
    df = pd.read_csv(io.BytesIO(z.read(name)), usecols=['SERIALNO', 'AGEP', 'ESR', 'PUMA'])
a = df[(df.AGEP >= 25) & (df.AGEP <= 64)].copy()
a['emp'] = a.ESR.isin([1, 2]).astype(int)
hs = a.groupby('SERIALNO').size()
g = a.groupby('SERIALNO').emp.agg(['size', 'mean']); g2 = g[g['size'] == 2]
p = a.emp.mean()
out = dict(state=st, persons=len(df), adults_25_64=len(a), households=int(len(hs)), pumas=int(df.PUMA.nunique()),
           household_adult_count_share=hs.value_counts(normalize=True).sort_index().round(4).to_dict(),
           employment_rate=p, two_adult_concordance=float(((g2['mean'] == 0) | (g2['mean'] == 1)).mean()),
           independence_concordance=p * p + (1 - p)**2,
           puma_employment_rate=a.groupby('PUMA').emp.mean().describe().to_dict())
json.dump(out, open(RESULTS / f'pums_structure_{st}.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
