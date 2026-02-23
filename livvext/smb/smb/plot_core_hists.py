#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import smb.preproc as preproc
from livvkit import elements as el

describe_hist = """
Histogram of differences between modeled and observed annual surface mass
balance in kg m-2 yr-1, at all core and firn locations in the Greenland Ice
Sheet. High frequencies near zero imply greater model agreement with core data.
Data are compiled from Cogley et al. (2004), Bales et al. (2009), and the
PROMICE database (Machguth et al., 2016), from both ablation and accumulation
zones. The light blue line highlights 0 difference between model and observed;
values above (below) this line indicate that the model overestimates
(underestimates) SMB in comparison to core observations.
"""

describe_basins = """
Histogram of differences between modeled and observed annual surface mass
balance in kg m-2 yr-1, at core and firn locations in the Greenland Ice Sheet.
Colors correspond to the eight major drainage basins denoted by Zwally et al.,
2012 High frequencies near zero imply greater model agreement with core data.
Data were compiled from Cogley et al. (2004), Bales et al., 2009, and the
PROMICE database (Machguth et al., 2016), from both ablation and accumulation
zones. The light blue line highlights 0 difference between model and observed;
values above (below) this line indicate that the model overestimates
(underestimates) SMB in comparison to core observations. Axes have been
normalized for the eight drainage histograms in order to show differences
between number of available comparison points per basin.
"""
IMG_GROUP = "Statstical"


def main(args, config):
    img_list = []
    plt.style.use = "default"

    _, _, _, smb_mo_file = preproc.core_file(config)
    smb_avg = pd.read_csv(smb_mo_file)
    working = smb_avg[smb_avg.thk_flag == 1]

    diff_data = pd.DataFrame(
        data=working.mod_b - working.b, index=working.index, columns=["difference"]
    )
    diff_data["zwallyBasin"] = working.zwallyBasin
    diff_data["majorbasins"] = np.floor(working.zwallyBasin).astype("str")

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

    list_regions = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0"]
    zwally_regions = [2, 3, 4, 1, 5, 6, 7, 8]
    pltcrds = [(0, 1), (1, 0), (1, 1), (0, 0), (2, 0), (2, 1), (3, 0), (3, 1)]

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
    for i in range(0, len(list_regions)):
        regname = list_regions[i]
        inde = pltcrds[i]
        reg_data = diff_data[diff_data.majorbasins == regname]
        subs = reg_data["difference"].hist(ax=axes[inde], color=colors[regname])
        subs.set_xlabel("Model - observed SMB difference\n(kg m$^{-2}$ a$^{-1}$)")
        subs.set_ylabel("Cell frequency")
        subs.set_title("Region {}".format(zwally_regions[i]), size=12)
        subs.set_facecolor("white")
        subs.axvline(x=0.0, c="lightskyblue", linewidth=2)
        subs.set_xlim(-2000, 2000)
        subs.set_ylim(0, 65)

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_diffhist_basins.png")
    plt.savefig(img_file)
    plt.close()
    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Core difference histogram by basin",
        " ".join(describe_basins.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)
    difftest = pd.DataFrame(
        data=smb_avg.mod_b - smb_avg.b, index=smb_avg.index, columns=["difference"]
    )
    _ = plt.figure(figsize=(8, 8))
    hist1 = difftest["difference"].hist(bins=30, edgecolor="k")
    hist1.set_xlabel("Model - observed SMB difference\n(kg m$^{-2}$ a$^{-1}$)")
    hist1.set_ylabel("Cell frequency")
    hist1.set_xlim(-4000, 4000)
    hist1.axvline(x=0.0, c="lightskyblue", linewidth=2)

    plt.tight_layout()
    img_file = os.path.join(args.out, "core_diffhist_all.png")
    plt.savefig(img_file)
    plt.close()

    img_link = os.path.join(
        "imgs", os.path.basename(args.out), os.path.basename(img_file)
    )
    img_elem = el.Image(
        "Core difference histogram",
        " ".join(describe_hist.split()),
        img_link,
        height=args.img_height,
        group=IMG_GROUP,
        relative_to="",
    )
    img_list.append(img_elem)

    return img_list[::-1]
