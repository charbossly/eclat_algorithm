#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#data set
dataset = [['Pain','Beurre','Confiture'],
           ['Beurre','Coca'],
           ['Beurre','Lait'],
           ['Pain','Beurre','Coca'],
           ['Pain','Lait'],
           ['Beurre','Lait'],
           ['Pain','Lait'],
           ['Pain','Beurre','Lait','Confiture'],
           ['Pain','Beurre','Lait']
 ]


freq_itemsets = pd.DataFrame(dataset)
print(freq_itemsets)

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary.astype(int), columns=te.columns_)
print(df)


# Finding frequent itemsets with Eclat
min_support = 2/len(df)
frequent_itemsets = apriori(df.astype(bool), min_support=min_support, use_colnames=True)

# Displaying the results
print(frequent_itemsets)


# Filtering for frequent itemsets with three items
frequent_3_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 3)]

# Displaying the results
print(frequent_3_itemsets)




