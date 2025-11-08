import numpy as np
import os
import shlex


cell_types = np.array(['CD16+ monocyte', 'CD4+ T cell', 'Natural killer cell', 'Dendritic cell',
                       'Cytotoxic T cell', 'Megakaryocyte', 'CD14+ monocyte', 'Plasmacytoid dendritic cell', 'B cell'])

for k in [1,2,3,4,5,10,20,30]:

    for i in cell_types:
        print(i)
        os.system("python stats_auroc_score.py --cell_type " + shlex.quote(i) + " --number " + str(k))
