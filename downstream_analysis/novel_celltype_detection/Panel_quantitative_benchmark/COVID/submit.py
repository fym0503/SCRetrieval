import numpy as np
import os
import shlex



cell_types = np.array(['DC_c3-LAMP3','Macro_c3-EREG', 'Macro_c4-DNAJB1',
       'Neu_c4-RSAD2', 'Neu_c6-FGF23'])

for k in [1,2,3,4,5,10,20,30,50]:
       for i in cell_types:
              os.system("python stats_auroc_score.py --cell_type " + i + " --number " + str(k))
 
