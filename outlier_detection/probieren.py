#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:31:20 2025

@author: rainer
"""

import numpy as np

from outlier_detection.simple.one_dimension_numeric.z_score import Z_SCORE

d = np.random.normal(size=10_000)

z = Z_SCORE(list(d))
z.z_score()

print(z.data)
