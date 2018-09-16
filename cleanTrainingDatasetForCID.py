import pandas as pd
df = pd.read_csv('hackaturing.dsv', sep='|',engine='python')
validCid = df[~df['base_hackaturing.cid'].isna()]
validData.to_csv('onlyvalidCID.csv', sep=',')
