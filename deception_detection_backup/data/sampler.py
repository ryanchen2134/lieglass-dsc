from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Args:
        labels: list of int (0 or 1) for each sample in dataset.
    Returns:
        WeightedRandomSampler that oversamples the minority class so each epoch
        draws approximately 50/50 class balance.
    """
    class_counts = [labels.count(0), labels.count(1)]
    sample_weights = [1.0 / class_counts[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
