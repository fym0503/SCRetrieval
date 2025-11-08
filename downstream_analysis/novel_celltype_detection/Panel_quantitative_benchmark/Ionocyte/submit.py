import numpy as np
import os
import shlex
# cell_types = np.array(['DC_c3-LAMP3', 'Epi-AT2', 'Epi-Ciliated', 'Epi-Secretory',
#        'Epi-Squamous', 'Macro_c3-EREG', 'Macro_c4-DNAJB1', 'Mast',
#        'Neu_c4-RSAD2', 'Neu_c6-FGF23'])

# cell_types = np.array(['DC_c3-LAMP3','Macro_c3-EREG', 'Macro_c4-DNAJB1',
#        'Neu_c4-RSAD2', 'Neu_c6-FGF23'])

# for i in cell_types:
#     os.system("python plot.py --cell_type " + i)

# cell_types = np.array(['acinar','activated_stellate','alpha','beta','delta','ductal','endothelial',
#                        'epsilon','gamma','macrophage','mast','quiescent_stellate', 'schwann'])

# for i in cell_types:
for k in [1,2,3,4,5,10,20,30]:
    # print(i)
    os.system("python stats_auroc_score.py  --number " + str(k))
