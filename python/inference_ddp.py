import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from unet import UNet

def read_pgm(filename):
    """Reads a binary PGM (P5) image file into a numpy array."""
    with open(filename, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header != 'P5':
            raise ValueError("Only PGM binary format (P5) is supported")
        
        # Skip comments
        line = f.readline().decode('utf-8').strip()
        while line.startswith('#'):
            line = f.readline().decode('utf-8').strip()
            
        width, height = map(int, line.split())
        max_val = int(f.readline().decode('utf-8').strip())
        
        data = np.fromfile(f, dtype=np.uint8)
        # Handle cases where trailing bytes exist
        data = data[:width * height]
        return data.reshape((height, width))

def write_pgm(filename, array):
    """Writes a numpy array as a binary PGM (P5) image file."""
    height, width = array.shape
    with open(filename, 'wb') as f:
        f.write(b"P5\n")
        f.write(f"{width} {height}\n".encode('utf-8'))
        f.write(b"255\n")
        f.write(array.astype(np.uint8).tobytes())

class PatchDataset(Dataset):
    """Slices a massive image into patches for GPU ingestion without OOMs."""
    def __init__(self, image, patch_size=512, stride=400):
        self.image = image
        self.patch_size = patch_size
        self.height, self.width = image.shape
        self.patches = []

        # Create coordinate list of patch top-left points
        for y in range(0, self.height - patch_size + 1, stride):
            for x in range(0, self.width - patch_size + 1, stride):
                self.patches.append((y, x))
                
        # Capture bottom and right borders
        if (self.height - patch_size) % stride != 0:
            for x in range(0, self.width - patch_size + 1, stride):
                self.patches.append((self.height - patch_size, x))
        if (self.width - patch_size) % stride != 0:
            for y in range(0, self.height - patch_size + 1, stride):
                self.patches.append((y, self.width - patch_size))
                
    def __len__(self):
        return len(self.patches)
        
    def __getitem__(self, idx):
        y, x = self.patches[idx]
        patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
        # Map to 0-1 range float tensor
        tensor = torch.from_numpy(patch).float() / 255.0
        tensor = tensor.unsqueeze(0) # (1, H, W)
        return tensor, y, x

def main():
    # Initialize process group for DDP
    # Uses NCCL backend for multi-GPU communication (Gloo for fallback/CPU development)
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {local_rank}/{world_size} using GPU {torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
        print(f"Rank {local_rank}/{world_size} using CPU")

    image_path = "data/dfsar_focused_lunar_surface.pgm"
    if not os.path.exists(image_path):
        if local_rank == 0:
            print(f"Error: DFSAR image not found at {image_path}. Please run C++ pipeline first.")
        dist.destroy_process_group()
        sys.exit(1)

    # All ranks load the image (fast read on local storage)
    img = read_pgm(image_path)
    height, width = img.shape
    patch_size = 512

    if local_rank == 0:
        print(f"Loaded image size: {width}x{height}")
        print("Slicing patches and preparing dataloaders...")

    dataset = PatchDataset(img, patch_size=patch_size, stride=400)
    
    # Use DistributedSampler to partition patches across ranks
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2 if use_cuda else 0)

    # Initialize U-Net model
    model = UNet(n_channels=1, n_classes=1).to(device)
    
    # Load dummy/pretrained model weights (simulating a trained model for inference)
    # In practice: model.load_state_dict(torch.load("crater_model.pth", map_location=device))
    if local_rank == 0:
        print("Model initialized. Running distributed inference...")

    if use_cuda:
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    model.eval()

    # Local accumulators for stitching
    local_pred = np.zeros((height, width), dtype=np.float32)
    local_count = np.zeros((height, width), dtype=np.float32)

    with torch.no_grad():
        for batch_tensors, ys, xs in dataloader:
            batch_tensors = batch_tensors.to(device)
            # Forward pass through model
            logits = model(batch_tensors)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(ys)):
                y, x = ys[i].item(), xs[i].item()
                local_pred[y:y+patch_size, x:x+patch_size] += probs[i, 0]
                local_count[y:y+patch_size, x:x+patch_size] += 1.0

    # Convert arrays to tensors for DDP reduction
    local_pred_tensor = torch.from_numpy(local_pred).to(device)
    local_count_tensor = torch.from_numpy(local_count).to(device)

    if local_rank == 0:
        print("Inference complete. Reducing predictions from all GPUs...")

    # Sum predictions and counts from all ranks on Rank 0
    dist.reduce(local_pred_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(local_count_tensor, dst=0, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        # Convert back to numpy
        global_pred = local_pred_tensor.cpu().numpy()
        global_count = local_count_tensor.cpu().numpy()

        # Normalize overlapping regions
        global_count[global_count == 0] = 1.0
        final_probs = global_pred / global_count

        # Threshold to create binary mask: 1 for crater, 0 for background
        binary_map = (final_probs > 0.5).astype(np.uint8) * 255

        # Save result image
        out_path = "data/crater_segmented_map.pgm"
        write_pgm(out_path, binary_map)
        print(f"Stitched crater segmentation map saved to {out_path}!")
        print("DDP Inference completed successfully.")

    # Clean up distributed group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
