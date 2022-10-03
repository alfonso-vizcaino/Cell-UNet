#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:05:23 2021

@author: alfonso
"""


class BoundingBox:
    def __init__(self, x1, y1, x2, y2, unmatched=True):
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.unmatched = unmatched

    def contains(self, x, y):
        x_in_range = self.x1 <= x <= self.x2
        y_in_range = self.y1 <= y <= self.y2

        return x_in_range and y_in_range
