import torch
import bev_pool_ext  # Import your module

# Generate some dummy data for testing
batch_size = 2
num_points = 100
num_channels = 3
num_intervals = 5

depth = torch.rand((batch_size, num_points, 1, 1, 1))
feat = torch.rand((batch_size, num_points, 1, 1, num_channels))
ranks_depth = torch.randint(0, num_points, (num_points,)).int()
ranks_feat = torch.randint(0, num_points, (num_points,)).int()
ranks_bev = torch.randint(0, num_intervals, (num_points,)).int()
interval_lengths = torch.randint(1, 5, (num_intervals,)).int()
interval_starts = torch.cumsum(interval_lengths, dim=0) - interval_lengths

# Create an output tensor to store the result
out = torch.zeros((batch_size, num_channels, num_intervals, 1, 1), dtype=torch.float32)

# Call the forward pass function
bev_pool_ext.bev_pool_v2_forward(
    depth,
    feat,
    out,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_lengths,
    interval_starts,
)

# Print the result
print("Forward pass output:")
print(out)
