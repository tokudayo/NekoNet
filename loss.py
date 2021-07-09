import torch
import torch.nn as nn

def _get_anchor_positive_triplet_mask(labels):
    device = labels.device
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal.byte() ^ 1

    return mask

def _get_triplet_mask(labels):
    # Check that i, j and k are distinct
    device = labels.device
    indices_equal = torch.eye(labels.size(0), device=device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j
    mask = (valid_labels & distinct_indices).byte()
    return mask

def _pairwise_distance(embeddings):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    device = embeddings.device

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def _semihard(labels, embeddings, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = _pairwise_distance(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss

def _hard(labels, embeddings, device, margin=1.0, hardest=False):
    """
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = _pairwise_distance(embeddings)

    if hardest:
        # Get the hardest positive pairs
        mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
        valid_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

        # Get the hardest negative pairs
        mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
        max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.nn.functional.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
        triplet_loss = torch.mean(triplet_loss)
    else:
        anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
        anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        loss = anc_pos_dist - anc_neg_dist + margin

        mask = _get_triplet_mask(labels).float()
        triplet_loss = loss * mask

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = torch.nn.functional.relu(triplet_loss)

        # Count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

    return triplet_loss
  
def _gor(labels, embeddings):
    dim = embeddings.shape[1]

    pairwise_product = embeddings @ embeddings.T
    anchor_negative_mask = _get_anchor_negative_triplet_mask(labels)

    ff = pairwise_product * anchor_negative_mask
    N = torch.sum(torch.gt(ff, 1e-16).float())
    
    M1 = 1.0/N*torch.sum(ff)
    M2 = 1.0/N*torch.sum(torch.square(ff))
    Lgor = torch.square(M1)
    if M2 - 1/dim > 0: Lgor += M2 - 1/dim
    return Lgor

class SemiHardTripletLossWithGOR(nn.Module):
    def __init__(self, device, margin=1.0, gor_sample_size=None, alpha_gor=1.0):
        super().__init__()
        self.device = device
        self.margin = margin
        self.gor_sample_size = gor_sample_size
        self.alpha_gor = alpha_gor

    def forward(self, input, target, **kwargs):
        return _semihard(target, input, self.device, self.margin) + self.alpha_gor*_gor(target, input)

class HardTripletLossWithGOR(nn.Module):
    def __init__(self, device, margin=1.0, gor_sample_size=None, alpha_gor=1.0, hardest=False):
        super().__init__()
        self.device = device
        self.margin = margin
        self.gor_sample_size = gor_sample_size
        self.alpha_gor = alpha_gor
        self.hardest = hardest

    def forward(self, input, target, **kwargs):
        return _hard(target, input, self.device, margin=self.margin, hardest=self.hardest) + self.alpha_gor*_gor(target, input)