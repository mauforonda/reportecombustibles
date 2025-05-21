#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt
import warnings

warnings.filterwarnings("ignore")

SALDOS = "saldos/data"
COMBUSTIBLES = [3, 2, 7, 10]
REPORT = "reporte/timeline.csv"

# Consolidate all events

df = pd.concat(
    [
        pd.read_csv(f"{SALDOS}/{f}", parse_dates=["fecha_actualizacion"])
        for f in os.listdir(SALDOS)
    ]
).sort_values("fecha_actualizacion")

# Filter in events since April 2025

df = df[
    (df.fecha_actualizacion.dt.date >= dt.date(2025, 4, 1))
    & (df.id_producto_bsa.isin(COMBUSTIBLES))
]

# Filter out events with negative balance

df = df[df.saldo_bsa > 0]

# Filter out balance outliers

df = pd.concat(
    [
        df.groupby(["id_eess", "id_producto_bsa"], group_keys=False).apply(
            lambda dfi: dfi[(np.abs(stats.zscore(dfi.saldo_bsa)) < 3)]
        )
    ]
)

# Sum balance per date and product across all stations

df.groupby(
    ["fecha_actualizacion", "id_producto_bsa"], as_index=False
).saldo_bsa.sum().pivot_table(
    index="fecha_actualizacion", columns="id_producto_bsa", values="saldo_bsa"
).to_csv(REPORT, date_format="%Y-%m-%dT%H:%M:%S")
