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
REPORTE_SALDO = "reporte/saldo.csv"
REPORTE_CONSUMO = "reporte/consumo.csv"


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
).to_csv(REPORTE_SALDO, date_format="%Y-%m-%dT%H:%M:%S")


def tendencia_consumo(df, producto):
    dfp = df[df.id_producto_bsa == producto].copy()
    dfp = (
        dfp.groupby(["fecha_actualizacion", "id_eess"])
        .saldo_bsa.sum()
        .unstack(level=1)
        .ffill()
    )
    dfp = np.log1p(dfp).diff()
    dfp = dfp.rolling(window=2).mean()
    dfp = dfp[(dfp > -2.2) & (dfp < 6)]
    dfp = (
        1 - np.exp(dfp[dfp < 0].resample("D").sum(min_count=1).mean(axis=1)).iloc[:-1]
    ).rename(producto)

    return dfp


pd.concat([tendencia_consumo(df, p) for p in COMBUSTIBLES], axis=1).to_csv(
    REPORTE_CONSUMO
)
