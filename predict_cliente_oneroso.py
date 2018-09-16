import pandas as pd

dataset = 'processTestData.csv'
fd = pd.read_csv(dataset)

features_onerosas = ['cid', 'valor_item']
dataset_oneroso = df[features_onerosas]

dataset_oneroso = dataset_oneroso[dataset_oneroso.valor_item > 5e3]
#dataset_oneroso.describe()

features_onerosas.append('id_beneficiario')
clientes_de_interesse = df[features_onerosas]
pd.merge(clientes_de_interesse, dataset_oneroso)
fdm = df[df.id_beneficiario.isin(clientes_de_interesse.id_beneficiario)]

