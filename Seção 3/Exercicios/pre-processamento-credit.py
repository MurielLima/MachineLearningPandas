import pandas as pd
base = pd.read_csv('.\credit-data.csv')
base.describe()
base.loc[base['age']<0]
# apagar a coluna
base.drop('age',1,inplace=True)
#apagar somente os registros com problema
base.drop(base[base.age<0].index,inplace=True)
#preencher os valores manualmente
#preencher os valores manualmente COM A MEDIA
base.mean()
base['age'].mean()
base['age'][base.age>0].mean()
base.loc[base.age<0,'age']=40.92


#-------------
#tratamento de valores faltantes
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

#dividindo colunas do dataframe
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=    imputer.fit(previsores[:,0:3])
previsores[:,0:3]   =   imputer.transform(previsores[:,0:3])
#-------------
#padronizando escala de dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores[:,0:3])

#dividindo em teste e treinamento
from sklearn.cross_validation import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento,classe_teste = train_test_split(previsores,classe,test_size=0.25,random_state=0)