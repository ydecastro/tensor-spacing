#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:37:11 2023

@author: Yohann
"""
#%%
import numpy as np


#%%
def cdf_dist_null(z):
    angle = np.arccos(z)
    primitive = (1-np.cos(angle))/2
    return primitive

def scalar_prod(u,v):
    return np.dot(np.transpose(u),v)
