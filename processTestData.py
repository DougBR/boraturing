import pandas as pd
df = pd.read_csv('hackaturing.dsv', sep='|')
df = df[df['base_hackaturing.cid'].isna()]
df = df.drop('base_hackaturing.cid', axis = 1)
#df = df.drop(['base_hackaturing.valor_pago', 'base_hackaturing.valor_cobrado', 'base_hackaturing.prestador'], axis=1)
df.to_csv('testData80_allfeatures.csv', sep=',')