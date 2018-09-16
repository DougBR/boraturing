import pandas as pd
df = pd.read_csv('hackaturing.dsv', sep='|',engine='python')
validCid = df['base_hackaturing.cid']
validData = df.dropna()
df.to_csv('onlyvalidCID.csv', sep=',')
