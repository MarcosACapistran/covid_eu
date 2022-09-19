#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download, organize and store Europe COVID data
"""

import numpy as np
import urllib
outfilename = "data/europe_covid_data.csv"
url_of_file = "https://opendata.ecdc.europa.eu/covid19/nationalcasedeath_eueea_daily_ei/csv"
urllib.request.urlretrieve(url_of_file, outfilename)


import csv
csv_data = []
with open(outfilename) as file_obj:
    reader = csv.reader(file_obj)
    for row in reader:
        csv_data.append(row)
        
import pandas as pd        
df = pd.DataFrame(csv_data)

labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
          "FR","HR","IT","CY","LV","LT","LU","HU","MT",
          "NL","AT","PL","PT","RO","SI","SK","FI","SE"]

for label in labels:

    this_country = df[df[7]==label][:]
    covid_cases = np.flip(this_country[:][4])
    filename = 'data/covid_'+label+'.txt'
    with open(filename, 'w') as f:
        for entry in covid_cases:
            f.write(entry + '\n')
