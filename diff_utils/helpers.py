import math
import torch
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import random 
from inspect import isfunction
import os
import json 
import glob
import shutil
#import open3d as o3d


def log_msg(msg, rank0_only=False):
    """
    AI Hub / server 环境优先用 _log。
    本地或 _log 不可用时 fallback 到 print。
    """
    rank = os.environ.get("RANK", None)
    is_rank0 = rank is None or int(rank) == 0

    if rank0_only and not is_rank0:
        return

    msg = str(msg)

    if hub_log is not None:
        try:
            hub_log(msg)
            return
        except TypeError:
            # 如果你们项目里的 _log 不是 _log(msg)，
            # 而是 _log(exp_dir, msg) 这种格式，尝试兼容一下。
            try:
                exp_dir = globals().get("args", None).exp_dir
                hub_log(exp_dir, msg)
                return
            except Exception:
                pass
        except Exception:
            pass

    print(msg, flush=True)


def get_split_filenames(data_source, split_file, f_name="sdf_data.csv"):
    split = json.load(open(split_file))
    csvfiles = []
    for dataset in split: # e.g. "acronym" "shapenet"
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(data_source, dataset, class_name, instance_name, f_name)
                if not os.path.isfile(instance_filename):
                    print("Requested non-existent file '{}'".format(instance_filename))
                    continue
                csvfiles.append(instance_filename)
    return csvfiles

def sample_pc(f, samp=1024, add_flip_augment=False): 
    '''
    f: path to csv file
    '''
    data = torch.from_numpy(pd.read_csv(f, sep=',',header=None).values).float()
    pc = data[data[:,-1]==0][:,:3]
    pc_idx = torch.randperm(pc.shape[0])[:samp] 
    pc = pc[pc_idx]

    if add_flip_augment:
        pcs = []
        flip_axes = torch.tensor([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], device=pc.device)
        for idx, axis in enumerate(flip_axes):
            pcs.append(pc * axis)
        return pcs


    return pc

def perturb_point_cloud(pc, perturb, pc_size=None, crop_percent=0.25):
    '''
    if pc_size is None, return entire pc; else return with shape of pc_size
    '''
    assert perturb in [None, "partial", "noisy"]
    if perturb is None:
        pc_idx = torch.randperm(pc.shape[1])[:pc_size] 
        pc = pc[:,pc_idx]   
        #print("pc shape: ", pc.shape)
        return pc
    elif perturb == "partial":
        return crop_pc(pc, crop_percent, pc_size)
    elif perturb == "noisy":
        return jitter_pc(pc, pc_size)

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def crop_pc(xyz, crop, pc_size=None, fixed_points = None, padding_zeros = False):
    '''
     crop the point cloud given a randomly selected view
     input point cloud: xyz, with shape (B, N, 3)
     crop: float, percentage of points to crop out (e.g. 0.25 means keep 75% of points)
     pc_size: integer value, how many points to return; None if return all (all meaning xyz size * crop)
    '''
    

    if pc_size is not None:
        xyz = xyz[:, torch.randperm(xyz.shape[1])[:pc_size] ]
    
    _,n,c = xyz.shape
    device = xyz.device
        
    crop = int(xyz.shape[1]*crop)
    #print("pc shape: ", xyz.shape, crop)

    
    assert c == 3
    if crop == n:
        return xyz # , None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3, device=device),p=2,dim=-1)
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).to(device)

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous() #, crop_data.contiguous()



def visualize_pc(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.reshape(-1,3))
    o3d.io.write_point_cloud("./pc.ply", pcd)
    #o3d.visualization.draw_geometries([pcd])

def jitter_pc(pc, pc_size=None, sigma=0.05, clip=0.1):
    device = pc.device
    pc += torch.clamp(sigma*torch.randn(*pc.shape, device=device), -1*clip, clip)
    if pc_size is not None:
        if len(pc.shape) == 3: # B, N, 3
            pc = pc[:, torch.randperm(pc.shape[1])[:pc_size] ]
        else: # N, 3
            pc = pc[torch.randperm(pc.shape[0])[:pc_size] ]

    return pc


def normalize_pc(pc):
    pc -= torch.mean(pc, axis=0)
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
    #bbox_length = torch.sqrt( torch.sum((torch.max(pc, axis=0)[0] - torch.min(pc, axis=0)[0])**2) )
    pc /= m
    return pc

def save_model(iters, model, optimizer, loss, path):
    torch.save({'iters': iters,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
                path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None
    
    model.load_state_dict(checkpoint['model_state_dict'])
    loss = checkpoint['loss']
    iters = checkpoint['iters']
    print("loading from iter {}...".format(iters))
    return iters, model, optimizer, loss


def _atomic_copy_file(src, dst):
    """Copy one file without exposing a partially written NAS artifact."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = f"{dst}.tmp.{os.getpid()}"
    try:
        shutil.copy2(src, tmp_dst)
        os.replace(tmp_dst, dst)
    finally:
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)


def _atomic_dump_json(value, dst):
    """Write JSON atomically; keep UTF-8 comments/descriptions readable."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = f"{dst}.tmp.{os.getpid()}"
    try:
        with open(tmp_dst, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_dst, dst)
    finally:
        if os.path.exists(tmp_dst):
            os.remove(tmp_dst)


def save_run_configs(conf_dir, config_path=None, resolved_specs=None,
                     runtime_args=None):
    """Save the exact input config plus the effective runtime configuration."""
    os.makedirs(conf_dir, exist_ok=True)
    if config_path is not None:
        config_path = os.path.abspath(os.path.expanduser(config_path))
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"training config does not exist: {config_path}")
        _atomic_copy_file(
            config_path, os.path.join(conf_dir, "config_used.json"))
    if resolved_specs is not None:
        _atomic_dump_json(
            resolved_specs, os.path.join(conf_dir, "config_resolved.json"))
    if runtime_args is not None:
        _atomic_dump_json(
            runtime_args, os.path.join(conf_dir, "train_args.json"))


def save_code_to_conf(conf_dir, config_path=None, resolved_specs=None,
                      runtime_args=None):
    """Snapshot training code and configs under ``<exp_dir>/code``.

    Resolve sources from the repository containing this module instead of the
    process working directory.  shutil raises on copy failures, unlike the old
    unchecked ``os.system('cp ...')`` implementation.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_dir = os.path.join(conf_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    source_folders = [
        "utils", "models", "diff_utils", "dataloader", "metrics",
        "config", "scripts", "process_data", "processed_data",
    ]
    ignore = shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "*.pyo", ".DS_Store")
    copied_folders = []
    for folder in source_folders:
        src = os.path.join(repo_root, folder)
        if not os.path.isdir(src):
            continue
        shutil.copytree(
            src,
            os.path.join(code_dir, folder),
            dirs_exist_ok=True,
            symlinks=True,
            ignore=ignore,
        )
        copied_folders.append(folder)

    for src in glob.glob(os.path.join(repo_root, "*.py")):
        _atomic_copy_file(src, os.path.join(code_dir, os.path.basename(src)))

    save_run_configs(
        conf_dir,
        config_path=config_path,
        resolved_specs=resolved_specs,
        runtime_args=runtime_args,
    )
    print(
        f"[code backup] saved folders={copied_folders}, "
        f"config={os.path.join(conf_dir, 'config_used.json')}",
        flush=True,
    )

class ScheduledOpt:
    '''
    optimizer = ScheduledOpt(4000, torch.optim.Adam(model.parameters(), lr=0))
    '''
    "Optim wrapper that implements rate."
    def __init__(self, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self._rate = 0
    
    # def state_dict(self):
    #     """Returns the state of the warmup scheduler as a :class:`dict`.
    #     It contains an entry for every variable in self.__dict__ which
    #     is not the optimizer.
    #     """
    #     return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    # def load_state_dict(self, state_dict):
    #     """Loads the warmup scheduler's state.
    #     Arguments:
    #         state_dict (dict): warmup scheduler state. Should be an object returned
    #             from a call to :meth:`state_dict`.
    #     """
    #     self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        #print("rate: ",rate)

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step

        warm_schedule = torch.linspace(0, 3e-4, self.warmup, dtype = torch.float64)
        if step < self.warmup:
            return warm_schedule[step]
        else:
            return 3e-4 / (math.sqrt(step-self.warmup+1))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

#from 0,1 to -1,1
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# from -1,1 to 0,1
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# from any batch to [0,1]
# f should have shape (batch, -1)
def normalize_to_zero_to_one(f):
    f -= f.min(1, keepdim=True)[0]
    f /= f.max(1, keepdim=True)[0]
    return f


# extract the appropriate t index for a batch of indices
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    #print("using LINEAR schedule")
    scale = 1000 / timesteps
    beta_start = scale * 0.0001 
    beta_end = scale * 0.02 
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    #print("using COSINE schedule")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    cos_in = ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
    np_in = cos_in.numpy()
    alphas_cumprod = np.cos(np_in)  ** 2
    alphas_cumprod = torch.from_numpy(alphas_cumprod)
    #alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)


def print_trainable_parameters(model, title="Trainable Parameters Summary"):
    """Print trainable module / parameter statistics before training."""
    from collections import defaultdict

    all_params = list(model.named_parameters())
    trainable = [
        (name, tuple(param.shape), param.numel(), param.dtype)
        for name, param in all_params
        if param.requires_grad
    ]
    frozen = [
        (name, tuple(param.shape), param.numel())
        for name, param in all_params
        if not param.requires_grad
    ]

    total_all = sum(p.numel() for _, p in all_params)
    total_trainable = sum(n for _, _, n, _ in trainable)
    total_frozen = total_all - total_trainable

    sep = "=" * 100
    print(sep)
    print(title)
    print(f"Model class: {model.__class__.__name__}")
    print(
        f"Total: {total_all:,} | trainable: {total_trainable:,} | "
        f"frozen: {total_frozen:,} | "
        f"trainable ratio: {100.0 * total_trainable / max(total_all, 1):.4f}%"
    )
    print(sep)

    # Top-level child modules (e.g. gs_model, diffusion_model)
    print("\n[Top-level modules]")
    for child_name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        child_trainable = sum(
            p.numel() for p in child.parameters() if p.requires_grad
        )
        print(
            f"  {child_name}: trainable {child_trainable:,} / {child_total:,} "
            f"({100.0 * child_trainable / max(child_total, 1):.2f}% trainable)"
        )

    # Aggregate by parent module path (all components except param name)
    by_module = defaultdict(lambda: {"numel": 0, "tensors": 0, "params": []})
    for name, shape, numel, dtype in trainable:
        parts = name.split(".")
        mod = ".".join(parts[:-1]) if len(parts) > 1 else "(root)"
        by_module[mod]["numel"] += numel
        by_module[mod]["tensors"] += 1
        by_module[mod]["params"].append((parts[-1], shape, numel, dtype))

    print("\n[Trainable submodules (sorted by param count)]")
    for mod, info in sorted(by_module.items(), key=lambda x: -x[1]["numel"]):
        print(
            f"  {mod}: {info['numel']:,} params, {info['tensors']} tensor(s)"
        )

    print("\n[Trainable parameter tensors (full list)]")
    for name, shape, numel, dtype in sorted(trainable, key=lambda x: x[0]):
        print(f"  {name}")
        print(f"    shape={shape}, numel={numel:,}, dtype={dtype}")

    if frozen:
        print(f"\n[Frozen parameters] ({len(frozen)} tensor(s), {total_frozen:,} params)")
        for name, shape, numel in sorted(frozen, key=lambda x: x[0]):
            print(f"  {name}: shape={shape}, numel={numel:,}")

    print(sep)
