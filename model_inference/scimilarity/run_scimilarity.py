# TESTING 
import time 
import scanpy as sc
import numpy as np
import argparse
import warnings
from scimilarity import CellAnnotation, align_dataset
from scimilarity.utils import lognorm_counts

# Ignore warnings
warnings.filterwarnings('ignore')

# Argument parser setup
parser = argparse.ArgumentParser(description='Process single-cell data.')
parser.add_argument('--input_adata', type=str, required=True, help='Path to input .h5ad file')
parser.add_argument('--output_adata', type=str, required=True, help='Path to output .npy file')
parser.add_argument('--model_path', type=str, required=True, help='Path to scimilarity model')
args = parser.parse_args()

# Load data
adams = sc.read(args.input_adata)
print(adams.var)
# Align and preprocess data
annotation_path = args.model_path + 'models/annotation_model_v1' 
ca = CellAnnotation(model_path=annotation_path)
# print(adams.obs.columns)
# print(adams.var.columns)
#print(ca)
# TODO make this less hardcoded
# print(adams.obs)
#adams.obs.columns = ['NAME', 'CellType', 'Experiment', 'Method']
#adams.obs.columns = ['soma_joinid', 'dataset_id', 'assay', 'assay_ontology_term_id',
 #      'celltype_raw', 'cell_type_ontology_term_id', 'development_stage',
  #     'development_stage_ontology_term_id', 'disease',
   #    'disease_ontology_term_id', 'donor_id', 'is_primary_data',
    #   'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id',
     #  'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue',
      # 'tissue_ontology_term_id', 'tissue_general',
       #'tissue_general_ontology_term_id', 'raw_sum', 'nnz', 'raw_mean_nnz',
       #'raw_variance_nnz', 'n_measured_vars']
# TODO feature_name or gene_name or feature_types or genome
tmp = list()
for i in range(len(adams.var.index)):
    tmp.append(adams.var.feature_name[adams.var.index[i]])
adams.var.index = tmp
# print(ca.gene_order)
adams.var_names_make_unique()
print(adams.var)
adams = align_dataset(adams, ca.gene_order,gene_overlap_threshold=1000)
print(adams)
adams.layers['counts'] = adams.X
adams = lognorm_counts(adams)

# print(adams.var)
# print(adams.obs)
# print(adams.obs.columns)
# Retreive the cell embeddings 
print('Getting embeddings')
start_time = time.time()
X_scimilarity = np.array(ca.get_embeddings(adams.X))
end_time = time.time()
execution_time = end_time - start_time

print(f"It took {execution_time} seconds to get embeddings to {adams.shape[0]} cells")

print(X_scimilarity)
print(X_scimilarity.shape)
# Save the modified AnnData object
print('Saving the embeddings in npy format')
np.save(args.output_adata, X_scimilarity)



