import pandas as pd
df = pd.read_csv('hackaturing.dsv', sep='|',engine='python')
df.describe()

