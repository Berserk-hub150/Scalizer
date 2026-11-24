"""Create a small synthetic CSV for the HTTP demo; never downloads trace data."""
from pathlib import Path

import numpy as np
import pandas as pd

minute = np.arange(300)
load = 100 + minute * 0.5
cpu = 0.015 * (load / 10) + 0.05
series = pd.DataFrame(dict(service="MS_EXAMPLE", minute=minute, load=load,
                           containers=10, cpu=cpu, mem=0.25,
                           rt_ms=20 + 3 * cpu / (1 - cpu)))
output = Path("data/api-demo-synthetic.csv")
output.parent.mkdir(parents=True, exist_ok=True)
series.to_csv(output, index=False)
print(f"Synthetic demo data: {output.resolve()}")
print("CSV upload metadata: load_unit=requests per second; cpu_basis=fraction_of_request")
print("Job: services=[MS_EXAMPLE], start=0d0, end=0d5, version=v1, source=csv")
