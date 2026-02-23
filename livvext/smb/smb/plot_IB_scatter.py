#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smb.preproc as preproc
from livvkit import elements as el

describe = """
Modeled annual surface mass balance versus estimates derived from IceBridge
altimetry data detailed in Lewis et al. (2017). SMB estimates from these were
taken in the accumulation zone (SMB>0), and comparisons are given in kg m-2
yr-1. Colors correspond to the eight major drainage basins denoted by Zwally et
al., 2012 A 1:1 line is drawn in blue. Note that because this plot compares
nearest neighbor values without any spatial interpolation, one coarse model
cell may be compared to multiple (if not many) observed SMB estimates, as the
isochrones dataset is of very high spatial resolution.
"""


def main(args, config):
    img_list = []
    plt.style.use = "default"

    _, _, ib_file = preproc.ib_outfile(config)
    ice_bridge = pd.read_csv(ib_file, index_col=False)

    majorbasins = np.floor(ice_bridge.zwallyBasin).astype("str")

    accplot = ice_bridge[
        ice_bridge.thk_flag == 1
    ]  # Remove data outside of modeled ice sheet for comparisons

    colors = {
        "0.0": "moccasin",
        "1.0": "steelblue",
        "2.0": "darkturquoise",
        "3.0": "green",
        "4.0": "lightsalmon",
        "5.0": "mediumpurple",
        "6.0": "grey",
        "7.0": "purple",
        "8.0": "firebrick",
    }
    if accplot.shape[0] > 0:
        testplot = accplot.plot(
            kind="scatter",
            x="b",
            y="mod_b",
            s=30,
            c=majorbasins.apply(lambda x: colors[x]),
            legend=True,
            alpha=0.1,
            figsize=(8, 8),
        )
    else:
        return []

    testplot.set_xlabel("Radar SMB estimate (kg m$^{-2}$ a$^{-1}$)")
    testplot.set_ylabel("Modeled SMB (kg m$^{-2}$ a$^{-1}$)")
    testplot.set_ylim(-200, 1200)
    testplot.set_xlim(-200, 1200)
    testplot.plot([0, 1], [0, 1], transform=testplot.transAxes, zorder=0)
    testplot.set_title("Accumulation")

    plt.tight_layout()
    img_file = os.path.join(args.out, "IB_AccCompare.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Modeled SMB vs. IceBridge",
        " ".join(describe.split()),
        img_link,
        height=args.img_height,
        group="Statstical",
        relative_to="",
    )
    img_list.append(img_elem)

    return img_list
