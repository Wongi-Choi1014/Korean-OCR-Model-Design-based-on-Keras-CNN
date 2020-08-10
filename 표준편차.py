# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:06:51 2020

@author: 원기
"""
syllable_to_index = {syllable: index for index, syllable in enumerate(syllable)}
index_to_syllable = {index: syllable for index, syllable in enumerate(syllable)}

import math

def mean(values):
  if len(values) == 0:
    return None
  return sum(values, 0.0) / len(values)

def standardDeviation(values, option):
  if len(values) < 2:
    return None
  sd = 0.0
  sum = 0.0
  meanValue = mean(values)
  for i in range(0, len(values)):
    diff = values[i] - meanValue
    sum += diff * diff
  sd = math.sqrt(sum / (len(values) - option))
  return sd
lst = [0 for _ in range(2349)]
for data in Korean_text:
    lst[syllable_to_index[data]]+=1
    
print("표준편차: ",standardDeviation(lst,0))
