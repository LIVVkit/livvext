#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smb.preproc as preproc
from livvkit import elements as el

describe_felev = """
Field estimates of annual surface mass balance in kg m-2 yr-1 as a function of
observed elevation in meters. Data were compiled from Cogley et al. (2004),
Bales et al. (2009), and the PROMICE database (Machguth et al., 2016), from
both ablation and accumulation zones. Colors correspond to the eight major
drainage basins denoted by Zwally et al., 2012 Size of the point represents the
number of years in the field record, with larger points containing
comparatively more temporal information than smaller points. Each point
represents an annual surface mass balance estimate averaged across at least one
year of data.
"""

describe_melev = """
Model elevation in meters versus modeled annual surface mass balance in kg m-2
yr-1. Each point represents a cell location at which a core or firn estimate is
available. Colors correspond to the eight major drainage basins denoted by
Zwally et al., 2012.
"""

describe_transelev = """
Annual surface mass balance in kg m-2 yr-1 as a function of observed elevation
in meters at four different transect locations in the ablation zone. Red dots
are modeled SMB and elevation, while blue dots are observed SMB and elevation
from firn and core field estimates. Observation data were compiled from the
PROMICE database (Machguth et al., 2016).
"""

describe_facc = """
Modeled annual surface mass balance versus estimates derived from firn and core
field data compiled from Cogley et al. (2004) and Bales et al. (2009). Ice
sheet and glacier SMB estimates from these sources are all taken in the
accumulation zone (SMB>0), and comparisons are given in kg m-2 yr-1. Colors
correspond to the eight major drainage basins denoted by Zwally et al., 2012 A
1:1 line is drawn in blue.
"""

describe_fabl = """
Modeled annual surface mass balance versus estimates derived from firn and core
field data compiled from PROMICE (Machguth et al., 2016). Ice sheet and glacier
SMB estimates from PROMICE are all taken in the ablation zone (SMB<0), and
comparisons are given in kg m-2 yr-1. Colors correspond to the eight major
drainage basins denoted by Zwally et al., 2012 A 1:1 line is drawn in blue.
"""
IMG_GROUP = "Statstical"


def main(args, config):
    img_list = []
    plt.style.use = "default"
    _, _, _, smb_mo_file = preproc.core_file(config)
    smb_avg = pd.read_csv(smb_mo_file)
    smb_avg = smb_avg[smb_avg.thk_flag == 1]

    majorbasins = np.floor(smb_avg.zwallyBasin).astype("str")

    grps = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 100, 500]
    size = np.digitize(smb_avg.nyears, grps)
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

    testplot = smb_avg.plot(
        kind="scatter",
        x="Z",
        y="b",
        s=size * 10 + 1,
        c=majorbasins.apply(lambda x: colors[x]),
        legend=True,
        alpha=0.6,
    )
    testplot.set_ylabel("Field SMB estimate (kg m$^{-2}$ a$^{-1}$) ")
    testplot.set_xlabel("Elevation (m)")
    testplot.axhline(y=0.0, c="darkgrey", linewidth=1, zorder=0)
    testplot.set_ylim(-5500, 1500)

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_smbelev_obs.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Obs. SMB vs. obs. elevation",
        " ".join(describe_felev.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    testplot2 = smb_avg.plot(
        kind="scatter",
        x="mod_Z",
        y="mod_b",
        s=30,
        c=majorbasins.apply(lambda x: colors[x]),
        legend=True,
        alpha=0.6,
    )
    testplot2.set_ylabel("Modeled SMB (kg m$^{-2}$ a$^{-1}$)")
    testplot2.set_xlabel("Elevation (m)")
    testplot2.axhline(y=0.0, c="darkgrey", linewidth=1, zorder=0)
    testplot2.set_ylim(-5500, 1500)

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_smbelev_mod.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Modeled SMB vs. model elevation",
        " ".join(describe_melev.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    working = smb_avg[smb_avg.thk_flag == 1]
    glacier_list = ["414", "180", "215", "454"]
    # glac_longnames = ['Qamanârssêp Sermia', 'Nioghalfvjerdsfjorden', 'Storstrømmen', 'K-transect']
    glac_longnames = [
        "Qaman\u00e2rss\u00eap Sermia",
        "Nioghalfvjerdsfjorden",
        "Storstr\u00f8mmen",
        "K-transect",
    ]

    fig = plt.figure(figsize=(12, 12))
    for i in range(0, len(glacier_list)):
        glacplot = fig.add_subplot(i + 221)
        glac_name = glacier_list[i]
        glac_data = working[working.glacier_ID == glac_name]
        glacplot.scatter(
            x=glac_data.Z, y=glac_data.b, color="slateblue", label="Observed"
        )
        glacplot.scatter(
            x=glac_data.mod_Z,
            y=glac_data.mod_b,
            marker="d",
            color="lightcoral",
            label="Modeled",
        )
        if i == 0:
            glacplot.legend(loc="upper left")
        glacplot.set_facecolor("white")
        glacplot.grid(color="lightgrey")
        glacplot.set_ylabel("Surface mass balance (kg m$^{-2}$ a$^{-1}$)")
        glacplot.set_xlabel("Transect elevation (m)")
        if i % 2 == 1:
            glacplot.set_ylabel(" ")
        if i < 2:
            glacplot.set_xlabel(" ")
        glacplot.set_title(glac_longnames[i], size=12)

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_transects.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Ablation transect SMB vs. obs. elevation",
        " ".join(describe_transelev.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    working = smb_avg[
        smb_avg.thk_flag == 1
    ]  # Remove data outside of modeled ice sheet for comparisons

    accplot = working[working.source != "promice"]
    majorbasins = np.floor(accplot.zwallyBasin).astype("str")
    testplot = accplot.plot(
        kind="scatter",
        x="b",
        y="mod_b",
        s=30,
        c=majorbasins.apply(lambda x: colors[x]),
        legend=True,
        alpha=0.6,
        figsize=(8, 8),
    )
    testplot.set_xlabel("Field SMB estimate (kg m$^{-2}$ a$^{-1}$)")
    testplot.set_ylabel("Modeled SMB (kg m$^{-2}$ a$^{-1}$)")
    testplot.set_ylim(-200, 1200)
    testplot.set_xlim(-200, 1200)
    testplot.plot([0, 1], [0, 1], transform=testplot.transAxes, zorder=0)
    testplot.set_title("Accumulation")

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_AccCompare.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Modeled SMB vs. field accumulation",
        " ".join(describe_facc.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    ablplot = working[working.source == "promice"]
    majorbasins = np.floor(ablplot.zwallyBasin).astype("str")
    testplot = ablplot.plot(
        kind="scatter",
        x="b",
        y="mod_b",
        s=30,
        c=majorbasins.apply(lambda x: colors[x]),
        legend=True,
        alpha=0.6,
        figsize=(8, 8),
    )
    testplot.set_xlabel("Field SMB estimate (kg m$^{-2}$ a$^{-1}$)")
    testplot.set_ylabel("Modeled SMB (kg m$^{-2}$ a$^{-1}$)")
    testplot.set_ylim(-6000, 1000)
    testplot.set_xlim(-6000, 1000)
    testplot.plot([0, 1], [0, 1], transform=testplot.transAxes, zorder=0)
    testplot.set_title("Ablation")

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_AblCompare.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Modeled SMB vs field ablation",
        " ".join(describe_fabl.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    return img_list
