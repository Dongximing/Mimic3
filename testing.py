import pandas as pd
import numpy as np

df = pd.DataFrame({"Name": ["Alice", "Bob", "Mallory", "Mallory", "Bob", "Mallory"],
                 "City":["Seattle", "Seattle", "Portland", "Seattle", "Seattle", "Portland"],
                 "Val":[4, 3, 3, np.nan, np.nan, 4]})
print(df)
