import pandas as pd
from Dbias.bias_classification import classifier, classify
df = pd.read_csv('/home/nimnim/PycharmProjects/Lumierre_Research/DataSet - DataSet(1).csv')
for index, row in df.iterrows():
    value = row['description']
    try:
        result = classifier(value)
        print(result[0]['label'])
        df.loc[index, 'bias'] = result[0]['label']
        df.to_csv("/home/nimnim/PycharmProjects/Lumierre_Research/DataSet - DataSet(1).csv", index=False)
    except:
        pass
