import csv

with open('/home/nimnim/PycharmProjects/Lumierre_Research/DataSet - DataSet(1).csv', 'r') as file:
    reader = csv.DictReader(file)
    count = 0
    for row in reader:
        if row['bias'] == 'Non-biased' or row['bias'] == 'Biased':
            count += 1

print(count)

