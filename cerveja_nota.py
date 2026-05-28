#%%
import pandas as pd
from sklearn import linear_model
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel('dados_cerveja_nota.xlsx')
x = df[['cerveja']]
y = df['nota']
reg = linear_model.LinearRegression()
reg.fit(x, y)
predict_reg = reg.predict(x.drop_duplicates())
arvore = tree.DecisionTreeRegressor()
arvore.fit(x, y)
predict_arvore = arvore.predict(x.drop_duplicates())
arvored2 = tree.DecisionTreeRegressor(max_depth=2)
arvored2.fit(x, y)
predict_arvored2= arvored2.predict(x.drop_duplicates())

plt.plot(x['cerveja'], y, 'o')
plt.grid(True)
plt.title('CERVEJA x NOTA')
plt.xlabel('CERVEJA')
plt.ylabel('NOTA')
plt.plot(x.drop_duplicates(), predict_reg)
plt.plot(x.drop_duplicates()['cerveja'], predict_arvore)
plt.plot(x.drop_duplicates()['cerveja'], predict_arvored2)
# %%
tree.plot_tree(arvore, feature_names=['cerveja'], filled=True)
# %%
