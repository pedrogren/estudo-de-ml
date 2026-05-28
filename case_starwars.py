#%%
import pandas as pd
from sklearn import tree
from matplotlib import pyplot

df = pd.read_parquet('dados_clones.parquet')
df = pd.read_parquet('dados_clones.parquet')
df.columns = df.columns.str.strip() 
features = ['p2o_master_id', 'Massa(em kilos)', 'General Jedi encarregado',
             'Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio',
               'Tamanho dos pés', 'Tempo de existência(em meses)']
x = df[features]
y = df['Status']
x = x.replace({'Yoda': 1, 'Shaak Ti': 2, 'Mace Windu': 3, 
               'Obi-Wan Kenobi': 4, 'Aayla Secura': 5, 'Tipo 1': 1, 
               'Tipo 2': 2, 'Tipo 3': 3, 'Tipo 4': 4, 'Tipo 5': 5,
               'Defeituoso': 0, 'Apto': 1})
modelo = tree.DecisionTreeClassifier(max_depth=5)
modelo.fit(x, y)
tree.plot_tree(modelo, feature_names=features, class_names=modelo.classes_, filled= True)
# %%
