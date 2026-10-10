import numpy as np
from collections import Counter
from collections import defaultdict
import pandas as pd



def _validate_k(k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")


def _reference(values, name):
    if values is None:
        raise ValueError(name + " is required: supply the complete eligible reference pool")
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or pd.isna(values).any():
        raise ValueError(name + " must be a nonempty 1D array without missing labels")
    return values


def _ranked(queries, predictions, k):
    _validate_k(k)
    queries = np.asarray(queries)
    predictions = np.asarray(predictions)
    if queries.ndim != 1 or predictions.ndim != 2 or len(queries) != len(predictions):
        raise ValueError("query labels and ranked predictions must have matching rows")
    if not len(queries) or not predictions.shape[1]:
        raise ValueError("ranked results must not be empty")
    if pd.isna(queries).any() or pd.isna(predictions).any():
        raise ValueError("query and retrieved labels must not be missing")
    return queries, predictions[:, :k]


def _finite_mean(scores):
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    return float(np.mean(finite)) if len(finite) else np.nan


def _finite_gini(scores, labels=None):
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    if not np.any(valid):
        return 0.0
    if labels is None:
        return compute_gini_index(scores[valid])
    return compute_cell_type_gini_index(np.asarray(labels)[valid], scores[valid])


def load_reference_metadata(path, label_column, batch_column):
    """Read one row per eligible reference cell (never deduplicate labels)."""
    reference = pd.read_csv(path)
    return (
        _reference(reference[label_column].values, "reference_labels"),
        _reference(reference[batch_column].values, "reference_batches"),
    )

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


def ndcg(correct_labels, top_k_predictions, k=5, *, reference_labels=None):
    """Binary NDCG@k; IDCG uses all eligible reference cells of the query type.

    Queries without a matching reference type return NaN and are excluded from
    the mean. The reference pool must match this particular split/direction.
    """
    queries, predictions = _ranked(correct_labels, top_k_predictions, k)
    reference = _reference(reference_labels, "reference_labels")
    counts = Counter(reference)
    discounts = 1.0 / np.log2(np.arange(1, k + 1, dtype=float) + 1.0)
    ideals = {label: np.sum(discounts[:min(k, counts[label])]) for label in queries}
    scores = []
    for label, row in zip(queries, predictions):
        if any(count > counts[value] for value, count in Counter(row).items()):
            raise ValueError("retrieved labels exceed the eligible reference counts")
        ideal = ideals[label]
        scores.append(float(np.sum((row == label) * discounts[:len(row)]) / ideal)
                      if ideal > 0 else np.nan)
    return scores, _finite_mean(scores)


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

def compute_cell_type_matching_metrics(correct_labels, top_k_predictions, k=5, *, reference_labels=None):
    metrics = {
        "Top-K Vote Accuracy": top_k_vote_accuracy(correct_labels, top_k_predictions, k)[1],
        "Top-K Average Accuracy": top_k_average_accuracy(correct_labels, top_k_predictions, k)[1],
        "Mean Average Precision (MAP)": mean_average_precision(correct_labels, top_k_predictions, k)[1],
        "Normalized Discounted Cumulative Gain (nDCG)": ndcg(correct_labels, top_k_predictions, k, reference_labels=reference_labels)[1],
        "Mean Reciprocal Rank (MRR)": mean_reciprocal_rank(correct_labels, top_k_predictions, k)[1]
    }
    return metrics 

def compute_entropy_batch(top_k_labels_batch, *, reference_batches=None, k=50):
    """Entropy@k / log2(B), where B counts all eligible reference batches."""
    reference = _reference(reference_batches, "reference_batches")
    _, batches = _ranked(np.zeros(len(top_k_labels_batch)), top_k_labels_batch, k)
    available = Counter(reference)
    denominator = np.log2(len(available)) if len(available) > 1 else None
    scores = []
    for row in batches:
        counts = Counter(row)
        if any(count > available[batch] for batch, count in counts.items()):
            raise ValueError("retrieved batches exceed the eligible reference counts")
        probabilities = np.asarray(list(counts.values()), dtype=float) / len(row)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        scores.append(float(entropy / denominator) if denominator else 0.0)
    return scores, _finite_mean(scores)


def _corrected_results(queries, labels, batches, reference_labels, reference_batches, k, alpha):
    queries, labels = _ranked(queries, labels, k)
    _, batches = _ranked(queries, batches, k)
    if labels.shape != batches.shape:
        raise ValueError("retrieved labels and batches must have equal shape")
    reference_labels = _reference(reference_labels, "reference_labels")
    reference_batches = _reference(reference_batches, "reference_batches")
    if len(reference_labels) != len(reference_batches):
        raise ValueError("reference labels and batches must have equal length")
    available = Counter(zip(reference_labels, reference_batches))
    for row_labels, row_batches in zip(labels, batches):
        counts = Counter(zip(row_labels, row_batches))
        if any(count > available[pair] for pair, count in counts.items()):
            raise ValueError("retrieved label/batch counts exceed the eligible reference pool; "
                             "check the original reference metadata and index mapping")
    # Avoid nanmean warnings for queries with no relevant reference cells.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return evaluate_ranked_results(
            queries, labels, batches, reference_labels, reference_batches, k=k, alpha=alpha)


def compute_alpha_ndcg_batch(queries, top_k_labels_cell_type, top_k_labels_batch,
                             alpha=0.5, *, reference_labels=None,
                             reference_batches=None, k=50):
    """Reference-normalized alpha-NDCG; only correct cells receive novelty gain."""
    result = _corrected_results(
        queries, top_k_labels_cell_type, top_k_labels_batch,
        reference_labels, reference_batches, k, alpha)
    return result["alpha_ndcg_scores"].tolist(), result["alpha_ndcg_mean"]


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

def compute_balanced_metrics(correct_labels, top_k_predictions, k=3, *, reference_labels=None):
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
        "Normalized Discounted Cumulative Gain (nDCG)": ndcg(correct_labels, top_k_predictions, k, reference_labels=reference_labels)[0],
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
        "nDCG_gini": _finite_gini(metrics["Normalized Discounted Cumulative Gain (nDCG)"]),
        "nDCG_celltype_gini": _finite_gini(metrics["Normalized Discounted Cumulative Gain (nDCG)"], correct_labels),
        "MRR_gini": compute_gini_index(metrics["Mean Reciprocal Rank (MRR)"]),
        "MRR_celltype_gini": compute_cell_type_gini_index(correct_labels, metrics["Mean Reciprocal Rank (MRR)"])
    }

def compute_batch_diversity_metrics(queries, top_k_labels_cell_type, top_k_labels_batch,
                                    alpha=0.5, *, reference_labels=None,
                                    reference_batches=None, k=50):
    """Corrected diversity metrics. 'Average Entropy' now means normalized entropy."""
    result = _corrected_results(
        queries, top_k_labels_cell_type, top_k_labels_batch,
        reference_labels, reference_batches, k, alpha)
    entropy = result["normalized_entropy_scores"]
    alpha_ndcg = result["alpha_ndcg_scores"]
    return {
        "Average Entropy": result["normalized_entropy_mean"],
        "Average α-NDCG": result["alpha_ndcg_mean"],
        "Entropy_gini": compute_gini_index(entropy),
        "α-NDCG_gini": _finite_gini(alpha_ndcg),
        "Entropy_cell_type_gini": compute_cell_type_gini_index(queries, entropy),
        "α-NDCG_cell_type_gini": _finite_gini(alpha_ndcg, queries),
        "nDCG_valid_fraction": result["ndcg_valid_fraction"],
        "α-NDCG_valid_fraction": result["alpha_ndcg_valid_fraction"],
    }
# ---------------------------------------------------------------------------
# Corrected reference-pool metrics
# ---------------------------------------------------------------------------
# These functions are kept in this module so the public package has one
# numerical implementation.  They are also exposed for scalar/unit testing.
def _validate_alpha(alpha):
    if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be finite and in [0, 1)")


def normalized_entropy_at_k(retrieved_batches, reference_batches, k=50):
    """Compute batch-label entropy@k normalized by log2(reference batches)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    retrieved_batches = np.asarray(retrieved_batches)[:k]
    reference_batches = np.asarray(reference_batches)
    if k == 0 or len(retrieved_batches) == 0:
        return 0.0

    batch_count = len(np.unique(reference_batches))
    if batch_count <= 1:
        return 0.0

    counts = Counter(retrieved_batches)
    total = float(len(retrieved_batches))
    entropy = -sum(
        (count / total) * np.log2(count / total)
        for count in counts.values()
        if count > 0
    )
    score = entropy / np.log2(batch_count)
    if score < -1e-10 or score > 1.0 + 1e-10:
        raise RuntimeError("normalized entropy outside [0, 1]: {}".format(score))
    return float(np.clip(score, 0.0, 1.0))


def ndcg_at_k(query_label, retrieved_labels, reference_labels, k=50):
    """Compute binary-relevance, reference-pool normalized NDCG@k."""
    if k < 0:
        raise ValueError("k must be non-negative")

    retrieved_labels = np.asarray(retrieved_labels)[:k]
    reference_labels = np.asarray(reference_labels)
    if k == 0:
        return 0.0

    relevance = (retrieved_labels == query_label).astype(np.float64)
    ranks = np.arange(1, len(relevance) + 1, dtype=np.float64)
    dcg = np.sum(relevance / np.log2(ranks + 1.0))

    relevant_count = int(np.sum(reference_labels == query_label))
    ideal_length = min(k, relevant_count)
    if ideal_length == 0:
        return np.nan

    ideal_ranks = np.arange(1, ideal_length + 1, dtype=np.float64)
    idcg = np.sum(1.0 / np.log2(ideal_ranks + 1.0))
    return float(dcg / idcg)


def _alpha_dcg(query_label, retrieved_labels, retrieved_batches, k, alpha):
    _validate_alpha(alpha)
    retrieved_labels = np.asarray(retrieved_labels)[:k]
    retrieved_batches = np.asarray(retrieved_batches)[:k]
    if len(retrieved_labels) != len(retrieved_batches):
        raise ValueError("retrieved labels and batches must have equal length")

    seen_relevant_batches = Counter()
    score = 0.0
    for rank, (label, batch) in enumerate(
        zip(retrieved_labels, retrieved_batches), start=1
    ):
        if label != query_label:
            continue
        novelty = (1.0 - alpha) ** seen_relevant_batches[batch]
        score += novelty / np.log2(rank + 1.0)
        # Incorrect cells never affect the redundancy count.
        seen_relevant_batches[batch] += 1
    return float(score)


def _ideal_alpha_dcg(query_label, reference_labels, reference_batches, k, alpha):
    """Greedy alpha-IDCG from the complete reference pool."""
    _validate_alpha(alpha)
    reference_labels = np.asarray(reference_labels)
    reference_batches = np.asarray(reference_batches)
    if len(reference_labels) != len(reference_batches):
        raise ValueError("reference labels and batches must have equal length")

    available = Counter(reference_batches[reference_labels == query_label])
    if not available:
        return np.nan

    selected = Counter()
    score = 0.0
    for rank in range(1, k + 1):
        candidates = [batch for batch, count in available.items() if count > 0]
        if not candidates:
            break
        # Maximizing (1-alpha)^selected is equivalent to selecting the least
        # represented available batch; sorting makes ties deterministic.
        batch = min(candidates, key=lambda value: (selected[value], str(value)))
        score += (1.0 - alpha) ** selected[batch] / np.log2(rank + 1.0)
        selected[batch] += 1
        available[batch] -= 1
    return float(score)


def _ideal_alpha_dcg_from_counts(available_counts, k, alpha):
    """Compute alpha-IDCG from precomputed relevant-cell batch counts."""
    _validate_alpha(alpha)
    available = Counter(available_counts)
    if not available:
        return np.nan
    selected = Counter()
    score = 0.0
    for rank in range(1, k + 1):
        candidates = [batch for batch, count in available.items() if count > 0]
        if not candidates:
            break
        batch = min(candidates, key=lambda value: (selected[value], str(value)))
        score += (1.0 - alpha) ** selected[batch] / np.log2(rank + 1.0)
        selected[batch] += 1
        available[batch] -= 1
    return float(score)


def alpha_ndcg_at_k(
    query_label,
    retrieved_labels,
    retrieved_batches,
    reference_labels,
    reference_batches,
    k=50,
    alpha=0.5,
):
    """Compute alpha-NDCG@k with batch novelty and reference-pool IDCG."""
    if k < 0:
        raise ValueError("k must be non-negative")
    dcg = _alpha_dcg(query_label, retrieved_labels, retrieved_batches, k, alpha)
    idcg = _ideal_alpha_dcg(
        query_label, reference_labels, reference_batches, k, alpha
    )
    if np.isnan(idcg) or idcg <= 0.0:
        return np.nan
    score = dcg / idcg
    if score < -1e-10 or score > 1.0 + 1e-10:
        raise RuntimeError("alpha-NDCG outside [0, 1]: {}".format(score))
    return float(np.clip(score, 0.0, 1.0))


def evaluate_ranked_results(
    query_labels,
    retrieved_labels,
    retrieved_batches,
    reference_labels,
    reference_batches,
    k=50,
    alpha=0.5,
):
    """Return per-query and mean corrected entropy/NDCG/alpha-NDCG."""
    query_labels = np.asarray(query_labels)
    retrieved_labels = np.asarray(retrieved_labels)
    retrieved_batches = np.asarray(retrieved_batches)
    if retrieved_labels.shape != retrieved_batches.shape:
        raise ValueError("retrieved labels and batches must have equal shape")
    if len(query_labels) != len(retrieved_labels):
        raise ValueError("query labels and ranked results must have equal length")

    _validate_alpha(alpha)
    reference_labels = np.asarray(reference_labels)
    reference_batches = np.asarray(reference_batches)
    if len(reference_labels) != len(reference_batches):
        raise ValueError("reference labels and batches must have equal length")

    reference_label_counts = Counter(reference_labels)
    reference_batch_counts = defaultdict(Counter)
    for label, batch in zip(reference_labels, reference_batches):
        reference_batch_counts[label][batch] += 1

    ndcg_idcg = {}
    alpha_idcg = {}
    for label in np.unique(query_labels):
        ideal_length = min(k, reference_label_counts[label])
        if ideal_length:
            ranks = np.arange(1, ideal_length + 1, dtype=np.float64)
            ndcg_idcg[label] = float(np.sum(1.0 / np.log2(ranks + 1.0)))
        else:
            ndcg_idcg[label] = np.nan
        alpha_idcg[label] = _ideal_alpha_dcg_from_counts(
            reference_batch_counts[label], k, alpha
        )

    discounts = 1.0 / np.log2(np.arange(1, k + 1, dtype=np.float64) + 1.0)
    batch_count = len(np.unique(reference_batches))
    entropy_denominator = np.log2(batch_count) if batch_count > 1 else None
    entropy_scores = []
    ndcg_scores = []
    alpha_scores = []
    for query_label, labels, batches in zip(
        query_labels, retrieved_labels, retrieved_batches
    ):
        labels = np.asarray(labels)[:k]
        batches = np.asarray(batches)[:k]
        relevant = labels == query_label

        if entropy_denominator is None or len(batches) == 0:
            entropy_scores.append(0.0)
        else:
            counts = Counter(batches)
            total = float(len(batches))
            entropy = -sum(
                (count / total) * np.log2(count / total)
                for count in counts.values()
                if count > 0
            )
            entropy_score = entropy / entropy_denominator
            if entropy_score < -1e-10 or entropy_score > 1.0 + 1e-10:
                raise RuntimeError(
                    "normalized entropy outside [0, 1]: {}".format(entropy_score)
                )
            entropy_scores.append(float(np.clip(entropy_score, 0.0, 1.0)))

        idcg = ndcg_idcg[query_label]
        if np.isnan(idcg):
            ndcg_scores.append(np.nan)
        else:
            ndcg_scores.append(float(np.sum(relevant * discounts[:len(labels)]) / idcg))

        ideal_alpha = alpha_idcg[query_label]
        if np.isnan(ideal_alpha) or ideal_alpha <= 0.0:
            alpha_scores.append(np.nan)
        else:
            alpha_dcg = _alpha_dcg(
                query_label, labels, batches, k, alpha
            )
            score = alpha_dcg / ideal_alpha
            if score < -1e-10 or score > 1.0 + 1e-10:
                raise RuntimeError("alpha-NDCG outside [0, 1]: {}".format(score))
            alpha_scores.append(float(np.clip(score, 0.0, 1.0)))

    entropy_scores = np.asarray(entropy_scores, dtype=float)
    ndcg_scores = np.asarray(ndcg_scores, dtype=float)
    alpha_scores = np.asarray(alpha_scores, dtype=float)
    return {
        "normalized_entropy_scores": entropy_scores,
        "ndcg_scores": ndcg_scores,
        "alpha_ndcg_scores": alpha_scores,
        "normalized_entropy_mean": float(np.mean(entropy_scores)),
        "ndcg_mean": float(np.nanmean(ndcg_scores)),
        "alpha_ndcg_mean": float(np.nanmean(alpha_scores)),
        "ndcg_valid_fraction": float(np.mean(~np.isnan(ndcg_scores))),
        "alpha_ndcg_valid_fraction": float(np.mean(~np.isnan(alpha_scores))),
    }
