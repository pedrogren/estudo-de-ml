#%%
import pandas as pd
from sklearn import tree
from matplotlib import pyplot

df = pd.read_excel("dados_cerveja.xlsx")
features = ['temperatura', 'copo', 'espuma', 'cor']
x = df[features]
y = df['classe']
modelo = tree.DecisionTreeClassifier()
x = x.replace({
    "mud": 1,
    "pint": 2,
    "sim": 1,
    "não": 0,
    'clara': 0, 'escura': 1
}) 
# %%
modelo.fit(x, y)
# %%
tree.plot_tree(modelo, feature_names=features, class_names=modelo.classes_,
                filled=True )
# %%
