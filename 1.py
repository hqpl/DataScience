import numpy as np
import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

np.random.seed(25)

services = ['RMP', 'SFP', 'WD', 'WPC', 'SRBP']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

DF_obj = DataFrame(np.random.rand(len(services)*len(months)).reshape((len(services),len(months))),
                   index=services,
                   columns=months)

DF_obj['average'] = DF_obj.mean(axis=1)


print(DF_obj.fillna(0))

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
fig.canvas.manager.set_window_title('Services')
for i, row in enumerate(DF_obj.index):
    row_data = DF_obj.loc[row, :'Dec']
    avg_val = DF_obj.loc[row, 'average']
    colors = ['#FFD1A9' if val > avg_val else '#B8E6B8' for val in row_data]
    row_data.plot(kind='bar', ax=axes[i], title=row, color=colors)
    axes[i].axhline(y=avg_val, color='#FFB3BA', linestyle='--', label='Average')
plt.tight_layout()
plt.show()
