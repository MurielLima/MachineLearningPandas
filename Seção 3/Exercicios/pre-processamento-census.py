# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:39:48 2018

@author: Muriel
"""
#dividir o data frame
import pandas as pd
base = pd.read_csv('./census.csv')
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

#transformar atributos nominais em descritivos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores=LabelEncoder()
#labels=labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

#transformando atributos de raça
onehotencoder=OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

#transformando o renda
labelencoder_classe= LabelEncoder()
classe=labelencoder_classe.fit_transform(classe)

#escalonando atributos numericos
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
previsores=scaler.fit_transform(previsores)
previsores_treinamento,previsores_teste,classe_treinamento,classe_teste = train_test_split(previsores,classe,test_size=0.25,random_state=0)