import torch
from einops import rearrange
def make_mesh(H, W, device="cuda"):
    """
    Create a mesh grid of shape (H, W, 2) with coordinates in the range [0, H-1] and [0, W-1].
    
    Args:
        H (int): Height of the grid.
        W (int): Width of the grid.
        device (str): Device to place the tensor on (default: "cuda").
        
    Returns:
        torch.Tensor: A tensor of shape (H, W, 2) containing the coordinates.
    """
    y = torch.arange(H, device=device).view(-1, 1).expand(H, W)
    x = torch.arange(W, device=device).view(1, -1).expand(H, W)
    return torch.stack((y, x), dim=-1)  # Shape: (H, W, 2)

class TokenIndexManager:
    def __init__(self, B, H_feat, W_feat, expansion_factor, device="cuda"):
        self.B = B
        self.H_feat = H_feat
        self.W_feat = W_feat
        self.exp = expansion_factor
        self.device = device

        self.H_exp = H_feat * expansion_factor
        self.W_exp = W_feat * expansion_factor

        # self.token_id_counter = 0
        self.refined_from = {}  # new_id -> parent_id
        self.token_ids_per_granularity = {}
        self.token_pos_per_granularity = {}
        self.token_id_map = torch.full((B, self.H_exp, self.W_exp), -1, dtype=torch.long, device=device)
        self.token_id_map_per_granularity = {}  # granularity -> token_id_map
        self.token_assignment = []
        
        box_id_to_token_id = []
        for i in range(self.H_exp):
            for j in range(self.W_exp):
                # if 2 * H_feat > i >= H_feat and 2 * W_feat > j >= W_feat:
                #     continue
                box_id_to_token_id.append((i, j))
        self.box_id_to_token_id = torch.tensor(box_id_to_token_id, device=device)

    def allocate_coarse_token_ids(self, granularity=4):
        token_idx = []
        token_pos = []
        H, W = self.H_exp // granularity, self.W_exp // granularity
        for b in range(self.B):
            token_idx_per_batch = []
            token_pos_per_batch = []
            token_id_counter = 0
            for i in range(H):
                for j in range(W):
                    if 2 * self.H_feat > i * granularity >= self.H_feat and 2* self.W_feat > j * granularity >= self.W_feat:
                        self.token_id_map[b, i * granularity:(i + 1) * granularity, j * granularity:(j + 1) * granularity] = -1
                        continue
                    token_id = token_id_counter
                    token_idx_per_batch.append(token_id)
                    token_pos_per_batch.append((i * granularity, j * granularity))
                    assign_map = torch.arange(granularity*granularity).view(granularity, granularity)
                    self.token_id_map[b, i * granularity:(i + 1) * granularity, j * granularity:(j + 1) * granularity] = token_id
                    token_id_counter += 1
            token_idx.append(torch.tensor(token_idx_per_batch, device=self.device, dtype=torch.long))
            token_pos.append(torch.tensor(token_pos_per_batch, device=self.device, dtype=torch.long))

        token_idx = torch.stack(token_idx)
        token_pos = torch.stack(token_pos)
        self.token_ids_per_granularity[granularity] = token_idx
        self.token_pos_per_granularity[granularity] = token_pos
        self.token_id_map_per_granularity[granularity] = self.token_id_map.clone()  # Store the token id map for this granularity




    def refine_token_indices(self, idx_batch_tensor, idx_batch_tensor_kept, from_granularity, to_granularity):
        """
        Refine token indices from a coarser granularity to a finer granularity.
        
        Args:
            idx_batch_tensor (torch.Tensor): Tensor of shape (B, N) containing indices of tokens at the coarser granularity.
            idx_batch_tensor_kept (torch.Tensor): Tensor of shape (B, N_kept) containing indices of tokens to be kept at the coarser granularity.
            from_granularity (int): The granularity level from which to refine.
            to_granularity (int): The granularity level to which to refine.
        
        """
        B, N = idx_batch_tensor.shape
        B2, N_kept = idx_batch_tensor_kept.shape
        B = B if B == B2 else max(B, B2)
        # print(f'B, N: {B}, {N}')
        
        # assert from_granularity == 2 * to_granularity, \
        #     f"from_granularity ({from_granularity}) must be twice to_granularity ({to_granularity})"

        multiplier = from_granularity // to_granularity
        assert multiplier in [1, 2, 4], \
            f"Multiplier must be 1, 2, or 4, but got {multiplier} for from_granularity {from_granularity} and to_granularity {to_granularity}"
            
        assert from_granularity == multiplier * to_granularity, \
            f"from_granularity ({from_granularity}) must be a multiple of to_granularity ({to_granularity})"
            
        if from_granularity == to_granularity:
            self.token_assignment = self.token_pos_per_granularity[to_granularity]
            return 

        token_idx = []
        token_pos = []

        idx_batch_tensor, _ = torch.sort(idx_batch_tensor, dim=-1, descending=False)
        idx_batch_tensor_kept, _ = torch.sort(idx_batch_tensor_kept, dim=-1, descending=False)

        token_pos_from = self.token_pos_per_granularity[from_granularity]
        token_idx_from = self.token_ids_per_granularity[from_granularity]
        token_assignment = []
    
        # Need to consider all previously allocated token ids
        _token_id_counter = sum([len(self.token_ids_per_granularity[g][0]) for g in self.token_ids_per_granularity.keys()])
        # _token_id_counter_from = sum([len(self.token_ids_per_granularity[g][0]) for g in self.token_ids_per_granularity.keys() if g > from_granularity])
        for b in range(B):
            token_idx_per_batch = []
            token_pos_per_batch = []
            token_assignment_per_batch = []
            
            token_id_counter = _token_id_counter
            if len(idx_batch_tensor) == B:
                for idx in idx_batch_tensor[b]:
                    i = token_pos_from[b][idx][0] # H
                    j = token_pos_from[b][idx][1] # W
                    if multiplier == 2:
                        sub_coords = [
                            (i, j), (i, j + to_granularity),
                            (i + to_granularity, j), (i + to_granularity, j + to_granularity)
                        ]
                    else: # multiplier == 4
                        # 4x4 sub-grids
                        sub_coords = [
                            (i, j), (i, j + to_granularity), (i, j + 2 * to_granularity), (i, j + 3 * to_granularity),
                            (i + to_granularity, j), (i + to_granularity, j + to_granularity), (i + to_granularity, j + 2 * to_granularity), (i + to_granularity, j + 3 * to_granularity),
                            (i + 2 * to_granularity, j), (i + 2 * to_granularity, j + to_granularity), (i + 2 * to_granularity, j + 2 * to_granularity), (i + 2 * to_granularity, j + 3 * to_granularity),
                            (i + 3 * to_granularity, j), (i + 3 * to_granularity, j + to_granularity), (i + 3 * to_granularity, j + 2 * to_granularity), (i + 3 * to_granularity, j + 3 * to_granularity)
                        ]
                    for ii, jj in sub_coords:
                        token_idx_per_batch.append(token_id_counter)
                        token_pos_per_batch.append((ii, jj))
                        self.token_id_map[b, ii:ii + to_granularity, jj:jj + to_granularity] = token_id_counter
                        token_id_counter += 1
                    
            # token_id_counter_from = _token_id_counter_from
            if len(idx_batch_tensor_kept) == B:
                for idx in idx_batch_tensor_kept[b]:
                    i = token_pos_from[b][idx][0] # H
                    j = token_pos_from[b][idx][1] # W
                    sub_assignment = make_mesh(from_granularity, from_granularity, device=self.device) + torch.tensor([i, j], device=self.device)
                    sub_assignment = rearrange(sub_assignment, 'H W C -> (H W) C')
                    token_assignment_per_batch.append(sub_assignment)
                
            token_idx.append(torch.tensor(token_idx_per_batch, device=self.device, dtype=torch.long))
            token_pos.append(torch.tensor(token_pos_per_batch, device=self.device, dtype=torch.long))
            token_assignment.append(torch.cat(token_assignment_per_batch, dim=0))  # B, N, 2
            
        token_idx = torch.stack(token_idx)
        token_pos = torch.stack(token_pos)
        token_assignment = torch.stack(token_assignment)  # B, N, 2
        
        self.token_ids_per_granularity[to_granularity] = token_idx
        self.token_pos_per_granularity[to_granularity] = token_pos
        self.token_id_map_per_granularity[to_granularity] = self.token_id_map.clone()  # Store the token id map for this granularity
        
        
        if len(self.token_assignment) == 0:
            self.token_assignment = token_assignment
        else:
            # print(f'self.token_assignment.shape: {self.token_assignment.shape}, token_assignment.shape: {token_assignment.shape}')
            self.token_assignment = torch.cat((self.token_assignment, token_assignment), dim=1)
        
    def _init_pos_to_idx_table_per_gradularity(self, gradularities):
        """
        Get the index in the position encoding for a given granularity and position.
        
        Args:
            gradularities (list): List of granularity levels to initialize.
        """
        
        pos_to_idx_per_granularity = {}
        
        for granularity in gradularities:
            H, W = self.H_exp // granularity, self.W_exp // granularity
            pos_idx_map = torch.full((H, W), -1, dtype=torch.long, device=self.device)
            pos_idx = 0
            for i in range(H):
                for j in range(W):
                    if 2 * self.H_feat > i * granularity >= self.H_feat and 2 * self.W_feat > j * granularity >= self.W_feat:
                        continue
                    # Assign a unique index for each position in the granularity grid
                    pos_idx_map[i, j] = pos_idx
                    pos_idx += 1
            pos_to_idx_per_granularity[granularity] = pos_idx_map
            
        self.pos_to_idx_per_granularity = pos_to_idx_per_granularity
        # return pos_to_idx_per_granularity
    
    def get_pos_idx(self, granularity, pos):
        x = pos[..., 0] // granularity
        y = pos[..., 1] // granularity
        pos_idx_map = self.pos_to_idx_per_granularity[granularity]
        pos_idx = pos_idx_map[x, y]
        return pos_idx
    
    def get_value_with_2d_index(self, value, index_2d, granularity, dim_first=True):
        """
        Get values from a 2D tensor using 2D indices.

        Args:
            value (torch.Tensor): The input tensor of shape (B, C, H, W), (B, H, W, C), (H, W, C), or (C, H, W).
            index_2d (torch.Tensor): The 2D indices of shape (B, N, 2).
            granularity (int): The granularity level.
            dim_first (bool): If True, the input tensor is assumed to have dimensions (B, C, H, W) or (C, H, W). If False, it is assumed to have dimensions (B, H, W, C) or (H, W, C).

        Returns:
            torch.Tensor: The values at the specified indices. (B, N, C)
        """

        assert value.dim() >= 3, "Value tensor must be of shape (B, C, H, W), (B, H, W, C), (H, W, C), or (C, H, W)"
        assert index_2d.dim() == 3 and index_2d.shape[-1] == 2, "Index tensor must be of shape (B, N, 2)"
        
        B = value.shape[0]
        
        if not dim_first:
            if value.dim() == 3:
                # Rearrange from (H, W, C) to (C, H, W)
                value = rearrange(value, 'H W C -> C H W')
            elif value.dim() == 4:
                # Rearrange from (B, H, W, C) to (B, C, H, W)
                value = rearrange(value, 'B H W C -> B C H W')
        
        if value.dim() == 3:
            return rearrange(value[:, index_2d[..., 0]//granularity, index_2d[..., 1]//granularity], 'C B N -> B N C') # [B, N, C]
        elif value.dim() == 4:
            N = index_2d.shape[1]
            # Build batch index: [B, N]
            batch_idx = torch.arange(B, device=value.device).unsqueeze(1).expand(B, N)
            return value[batch_idx, :, index_2d[..., 0]//granularity, index_2d[..., 1]//granularity] # [B, N, C]

    def get_token_id_map(self):
        return self.token_id_map