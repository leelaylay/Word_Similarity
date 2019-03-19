#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math

import google
from google import google


class GoogleSearch:
    def __init__(self, word1, word2):
        temp = google.search(word1, 1)
        if len(temp) > 0:
            self.x = temp[0].number_of_results*1.0
        else:
            self.x = 0.0
        temp = google.search(word2, 1)
        if len(temp) > 0:
            self.y = temp[0].number_of_results*1.0
        else:
            self.y = 0.0
        temp = google.search(word1+" "+word2, 1)
        if len(temp) > 0:
            self.xy = temp[0].number_of_results*1.0
        else:
            self.xy = 0.0

    @property
    def WebJaccard(self):
        return (self.xy)/(self.x+self.y-self.xy+1)

    @property
    def WebOverlap(self):
        return (self.xy)/(min(self.x)+1)

    @property
    def WebDice(self):
        return (2*self.xy)/(self.x+self.y+1)

    @property
    def WebPMI(self):
        return (self.xy)/(self.x+self.y-self.xy+1)

    @property
    def NGD(self):
        n = 42000000000
        ngdnumerator = max(math.log10(self.x), math.log10(
            self.y))-math.log10(self.xy)
        ngddenominator = math.log10(
            n)-min(math.log10(self.x), math.log10(self.y))+1
        return ngdnumerator/ngddenominator
