import numpy as np
from collections import Counter
from collections import defaultdict
import pandas as pd
def top_k_vote_accuracy(correct_labels, top_k_predictions, k=5):
    """
    Computes Top-K Voting Accuracy by determining the most frequent label
    among the top-K predictions and comparing it to the correct label.

    Args:
        correct_labels (list): List of true labels (size N).
        top_k_predictions (list of lists): List of top-K predictions for each query (size N*K).
        k (int): Number of predictions to consider for Top-K voting.

    Returns:
        tuple: (List of individual accuracies for each query, mean accuracy).
    """
    accuracies = []
    temp_store = {'cell':[],'retrieved':[]}
    for i, correct in enumerate(correct_labels):
        # Count occurrences of each label in the top-k predictions
        counter = Counter(top_k_predictions[i][:k])
        
        # Determine the most common label (ties resolved arbitrarily by Counter)
        most_common_label, _ = counter.most_common(1)[0]
        
        # Compare with correct label
        accuracies.append(1 if most_common_label == correct else 0)
    #     if model != None and most_common_label == correct:
        # temp_store['retrieved'].append(counter)
        # temp_store['cell'].append(correct)
    # if model != None:
    # temp_store = pd.DataFrame(temp_store)
    # temp_store.to_csv(f"scimilarity_vote_acc_correct.csv")


    mean_accuracy = np.mean(accuracies)
    # print(mean_accuracy)
    return accuracies, mean_accuracy


def top_k_average_accuracy(correct_labels, top_k_predictions, k=5):
    """
    Computes Top-K Average Accuracy.

    Args:
        correct_labels (list): List of true labels (size N).
        top_k_predictions (list of lists): List of top-K predictions for each query (size N*K).
        k (int): Number of predictions to consider for Top-K metrics.

    Returns:
        tuple: (List of individual accuracies for each query, mean accuracy).
    """
    accuracies = []
    for i, correct in enumerate(correct_labels):
        relevance = [1 if pred == correct else 0 for pred in top_k_predictions[i][:k]]
        accuracies.append(sum(relevance) / k)
    return accuracies, np.mean(accuracies)


def mean_average_precision(correct_labels, top_k_predictions, k=5):
    """
    Computes Mean Average Precision (MAP).

    Args:
        correct_labels (list): List of true labels (size N).
        top_k_predictions (list of lists): List of top-K predictions for each query (size N*K).
        k (int): Number of predictions to consider for MAP.

    Returns:
        tuple: (List of individual average precisions, mean average precision).
    """
    average_precisions = []
    for i, correct in enumerate(correct_labels):
        relevance = [1 if pred == correct else 0 for pred in top_k_predictions[i][:k]]
        precision_at_k = [
            sum(relevance[:j + 1]) / (j + 1) for j in range(len(relevance)) if relevance[j] == 1
        ]
        ap = sum(precision_at_k) / len(precision_at_k) if precision_at_k else 0
        average_precisions.append(ap)
    return average_precisions, np.mean(average_precisions)


def ndcg(correct_labels, top_k_predictions, k=5):
    """
    Computes Normalized Discounted Cumulative Gain (nDCG).

    Args:
        correct_labels (list): List of true labels (size N).
        top_k_predictions (list of lists): List of top-K predictions for each query (size N*K).
        k (int): Number of predictions to consider for nDCG.

    Returns:
        tuple: (List of individual nDCG scores, mean nDCG score).
    """
    ndcg_scores = []
    for i, correct in enumerate(correct_labels):
        relevance = [1 if pred == correct else 0 for pred in top_k_predictions[i][:k]]
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(relevance)])
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    return ndcg_scores, np.mean(ndcg_scores)


def mean_reciprocal_rank(correct_labels, top_k_predictions, k=5):
    """
    Computes Mean Reciprocal Rank (MRR).

    Args:
        correct_labels (list): List of true labels (size N).
        top_k_predictions (list of lists): List of top-K predictions for each query (size N*K).
        k (int): Number of predictions to consider for MRR.

    Returns:
        tuple: (List of individual reciprocal ranks, mean reciprocal rank).
    """
    reciprocal_ranks = []
    for i, correct in enumerate(correct_labels):
        found = False
        for rank, pred in enumerate(top_k_predictions[i][:k]):
            if pred == correct:
                reciprocal_ranks.append(1 / (rank + 1))
                found = True
                break
        if not found:
            reciprocal_ranks.append(0)
    return reciprocal_ranks, np.mean(reciprocal_ranks)

def compute_cell_type_matching_metrics(correct_labels, top_k_predictions ,k=5):
    metrics = {
        "Top-K Vote Accuracy": top_k_vote_accuracy(correct_labels, top_k_predictions, k)[1],
        "Top-K Average Accuracy": top_k_average_accuracy(correct_labels, top_k_predictions, k)[1],
        "Mean Average Precision (MAP)": mean_average_precision(correct_labels, top_k_predictions, k)[1],
        "Normalized Discounted Cumulative Gain (nDCG)": ndcg(correct_labels, top_k_predictions, k)[1],
        "Mean Reciprocal Rank (MRR)": mean_reciprocal_rank(correct_labels, top_k_predictions, k)[1]
    }
    return metrics 

def compute_entropy_batch(top_k_labels_batch):
    """
    Computes the average entropy of batch labels for a batch of queries.

    Args:
        top_k_labels_batch (list of lists): List where each element is a list of top-K batch labels for a query.

    Returns:
        float: Average entropy across all queries in the batch.
    """
    entropy_per_query = []
    total_entropy = 0
    for batch_labels in top_k_labels_batch:
        # Count the frequency of each batch label in the top-K results
        label_counts = Counter(batch_labels)
        total = sum(label_counts.values())

        # Compute probabilities for each batch label
        probabilities = [count / total for count in label_counts.values()]

        # Compute entropy for this query
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        total_entropy += entropy
        entropy_per_query.append(entropy)

    # Return the average entropy
    return entropy_per_query, total_entropy / len(top_k_labels_batch)

def compute_alpha_ndcg_batch(queries, top_k_labels_cell_type, top_k_labels_batch, alpha=0.5):
    """
    Computes the average α-NDCG for a batch of queries. Relevance is measured using cell type labels,
    and diversity is encouraged using batch labels.

    Args:
        queries (list): List of query labels (size N).
        top_k_labels_cell_type (list of lists): List where each element is a list of top-K predicted cell type labels.
        top_k_labels_batch (list of lists): List where each element is a list of top-K batch labels.
        alpha (float): Diversity parameter (default is 0.5).

    Returns:
        float: Average α-NDCG across all queries in the batch.
    """
    alpha_ndcg_per_query = [] 
    total_alpha_ndcg = 0
    for query_label, top_k_labels, batch_labels in zip(queries, top_k_labels_cell_type, top_k_labels_batch):
        # Initialize variables for gain and seen batch labels
        gain = 0
        seen_batch_labels = Counter()

        # Compute the gain (actual DCG)
        for i, (cell_type_label, batch_label) in enumerate(zip(top_k_labels, batch_labels)):
            # Relevance: Check if the predicted cell type label matches the query
            if cell_type_label == query_label:
                # Diversity: Encourage diverse batch labels
                gain += (1 - alpha) ** seen_batch_labels[batch_label] / np.log2(i + 2)
            # Track how many times each batch label has been seen (for diversity discounting)
            seen_batch_labels[batch_label] += 1

        # Compute the ideal DCG (IDCG) using both relevance and diversity
        idcg = 0
        ideal_seen_batch_labels = Counter()
        for i, batch_label in enumerate(batch_labels):
            # In the ideal ranking, we assume all cell type labels match the query
            idcg += (1 - alpha) ** ideal_seen_batch_labels[batch_label] / np.log2(i + 2)
            ideal_seen_batch_labels[batch_label] += 1

        # Normalize by IDCG
        alpha_ndcg = gain / idcg if idcg > 0 else 0
        alpha_ndcg_per_query.append(alpha_ndcg)
        total_alpha_ndcg += alpha_ndcg

    # Return the average α-NDCG across all queries
    return alpha_ndcg_per_query,total_alpha_ndcg / len(queries)



def compute_gini_index(scores):
    """
    Computes the Gini index for a list of query-level scores.

    Args:
        scores (list): A list of query-level scores (e.g., individual accuracies).

    Returns:
        float: Gini index (between 0 and 1), where 0 means perfect equality
               and 1 means maximum inequality.
    """
    # Convert to numpy array for easy manipulation
    scores = np.array(scores)

    if np.sum(scores) == 0:
        return 0

    sorted_scores = np.sort(scores)
    n = len(sorted_scores)

    # Compute cumulative sum with a leading zero
    cum_sum = np.zeros(n + 1, dtype=sorted_scores.dtype)
    cum_sum[1:] = np.cumsum(sorted_scores)

    # Calculate terms using vectorized operations
    i = np.arange(n)
    terms = i * sorted_scores - cum_sum[:-1]  # Exclude last element of cum_sum

    cumulative_differences = 2 * np.sum(terms)
    total_sum = cum_sum[-1]
    gini_index = cumulative_differences / (2 * n * total_sum)
    
    return gini_index
def compute_cell_type_gini_index(cell_types, scores):
    """
    Computes the Gini index at the cell type level by first averaging scores per cell type.

    Args:
        cell_types (list): A list of cell types corresponding to each query (length N).
        scores (list): A list of query-level scores (length N).

    Returns:
        float: Gini index for the averaged scores across cell types.
    """
    # Group scores by cell type
    cell_type_scores = defaultdict(list)
    for cell_type, score in zip(cell_types, scores):
        cell_type_scores[cell_type].append(score)
    
    # Compute average score for each cell type
    cell_type_averages = [np.mean(scores) for scores in cell_type_scores.values()]
    
    # Compute Gini Index based on cell type averages
    gini_index = compute_gini_index(cell_type_averages)
    
    return gini_index

def compute_balanced_metrics(correct_labels, top_k_predictions, k=3):
    """
    Computes multiple metrics and their Gini indices, including cell-type-based Gini indices.

    Args:
        correct_labels (list): A list of correct labels for each query.
        top_k_predictions (list): A list of lists, where each inner list contains the Top-K predictions for a query.
        queries (list): A list of query identifiers (length N, one for each query).
        top_k_label_cell_type (list): A list of cell types corresponding to each query (length N).
        k (int): The number of top predictions to consider.

    Returns:
        dict: A dictionary containing the computed metrics and their Gini indices.
    """
    # Step 1: Compute individual metrics
    metrics = {
        "Top-K Vote Accuracy": top_k_vote_accuracy(correct_labels, top_k_predictions, k)[0],
        "Top-K Average Accuracy": top_k_average_accuracy(correct_labels, top_k_predictions, k)[0],
        "Mean Average Precision (MAP)": mean_average_precision(correct_labels, top_k_predictions, k)[0],
        "Normalized Discounted Cumulative Gain (nDCG)": ndcg(correct_labels, top_k_predictions, k)[0],
        "Mean Reciprocal Rank (MRR)": mean_reciprocal_rank(correct_labels, top_k_predictions, k)[0]
    }

    # Step 2: Compute Gini indices for each metric
    return {
        "Top-K Vote Accuracy_gini": compute_gini_index(metrics["Top-K Vote Accuracy"]),
        "Top-K Vote Accuracy_celltype_gini": compute_cell_type_gini_index(correct_labels, metrics["Top-K Vote Accuracy"]),
        "Top-K Average Accuracy_gini": compute_gini_index(metrics["Top-K Average Accuracy"]),
        "Top-K Average Accuracy_celltype_gini": compute_cell_type_gini_index(correct_labels, metrics["Top-K Average Accuracy"]),
        "MAP_gini": compute_gini_index(metrics["Mean Average Precision (MAP)"]),
        "MAP_celltype_gini": compute_cell_type_gini_index(correct_labels, metrics["Mean Average Precision (MAP)"]),
        "nDCG_gini": compute_gini_index(metrics["Normalized Discounted Cumulative Gain (nDCG)"]),
        "nDCG_celltype_gini": compute_cell_type_gini_index(correct_labels, metrics["Normalized Discounted Cumulative Gain (nDCG)"]),
        "MRR_gini": compute_gini_index(metrics["Mean Reciprocal Rank (MRR)"]),
        "MRR_celltype_gini": compute_cell_type_gini_index(correct_labels, metrics["Mean Reciprocal Rank (MRR)"])
    }

def compute_batch_diversity_metrics(queries, top_k_labels_cell_type, top_k_labels_batch, alpha=0.5):
    """
    Computes Entropy and α-NDCG for a batch of queries.

    Args:
        queries (list): List of query labels (size N).
        top_k_labels_cell_type (list of lists): List of top-K predicted cell type labels for each query.
        top_k_labels_batch (list of lists): List of top-K batch labels for each query.
        alpha (float): Diversity parameter for α-NDCG (default is 0.5).

    Returns:
        dict: A dictionary containing the average Entropy and α-NDCG for the batch.
    """
    # Compute Entropy (measures diversity using batch labels)
    entropy, average_entropy = compute_entropy_batch(top_k_labels_batch)

    # Compute α-NDCG (measures relevance using cell type labels and diversity using batch labels)
    alpha_ndcg, average_alpha_ndcg = compute_alpha_ndcg_batch(queries, top_k_labels_cell_type, top_k_labels_batch, alpha)

    # Return results as a dictionary
    return {
        "Average Entropy": average_entropy,
        "Average α-NDCG": average_alpha_ndcg,
        "Entropy_gini":compute_gini_index(entropy),
        "α-NDCG_gini":compute_gini_index(alpha_ndcg),
        "Entropy_cell_type_gini":compute_cell_type_gini_index(queries,entropy),
        "α-NDCG_cell_type_gini":compute_cell_type_gini_index(queries,alpha_ndcg),

    }