import torch


def generate_grid(n_vox, interval):
    """Generate flattened grid"""
    with torch.no_grad():
        # create voxel grid
        grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
        # given xyz, to create the volume grid
        grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
        grid = grid.unsqueeze(0).cuda().float()  # 1 3 dx dy dz
        # flattened grid (3, dx*dy*dz)
        grid = grid.view(1, 3, -1)
    return grid
