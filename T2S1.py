import numpy as np
import csv


def 


with open('Data/Titanic_dataset.csv') as csvfile:
    titanic_reader = csv.DictReader(csvfile)
    for row in titanic_reader:
        print(row['Name'])
