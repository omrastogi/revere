"""STEP 0: frozen V-JEPA 2 segment embeddings for every UCF-Crime video, cached once.

python extract_emb.py run   <master_h5> <emb_dir> <n_shards> <workers>   # claims free shards via lock files
python extract_emb.py merge <master_h5> <emb_dir> <n_shards>             # -> emb_dir/all.npz

Per video: 32 equal segments, 16 consecutive frames each (7.5 fps, ~2 s) -> X (4 views, 32, 1024) float16.
Views: 0 center clip, 1 hflip of 0, 2 random offset inside the segment, 3 brightness/contrast +-15% and
Gaussian blur (p=0.3) of 0. Randomness is seeded per video id, so views are reproducible.
Only missing (video, view) pairs are computed; existing ones are never recomputed.
"""
import csv, os, subprocess, sys, time, zlib

import h5py
import numpy as np
import torch
import torch.nn.functional as F

SEG, T, V, D, S = 32, 16, 4, 1024, 224
MODEL = "facebook/vjepa2-vitl-fpc64-256"
CHUNK = 16  # clips per forward pass


def load_index(master):
    with open(os.path.join(os.path.dirname(master), "index.csv")) as f:
        return sorted(csv.DictReader(f), key=lambda r: r["vid"])


def shards_of(rows, n):
    # cost per video is fixed (32 x 16 frames x 4 views) whatever its length, so balance by count
    return [rows[i::n] for i in range(n)]


def load_cache(path):
    if not os.path.exists(path):
        return {}
    z = np.load(path)
    return {v: (x, d) for v, x, d in zip(z["vid"], z["X"], z["done"])}


def save_cache(path, cache):
    vids = sorted(cache)
    with open(path + ".tmp", "wb") as f:
        np.savez(f, vid=np.array(vids), X=np.stack([cache[v][0] for v in vids]),
                 done=np.stack([cache[v][1] for v in vids]))
    os.replace(path + ".tmp", path)


def clip_starts(n, rng):
    b = np.linspace(0, n, SEG + 1)
    last = max(n - T, 0)
    center = np.clip(np.round((b[:-1] + b[1:]) / 2 - T / 2).astype(int), 0, last)
    lo = np.floor(b[:-1]).astype(int)
    hi = np.maximum(np.ceil(b[1:]).astype(int) - T, lo)
    jitter = np.clip(rng.integers(lo, hi + 1), 0, last)
    return center, jitter


class Videos(torch.utils.data.Dataset):
    def __init__(self, master, todo):
        self.master, self.todo, self.h = master, todo, None

    def __len__(self):
        return len(self.todo)

    def read(self, ds, starts):
        out = np.empty((SEG, T, S, S, 3), np.uint8)
        for j, s in enumerate(starts):
            x = ds[s:s + T]
            if len(x) < T:  # short video: pad by repeating the last frame
                x = np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])
            out[j] = x
        return torch.from_numpy(out)

    def __getitem__(self, i):
        if self.h is None:  # one handle per DataLoader worker
            self.h = h5py.File(self.master, "r")
        vid, need = self.todo[i]
        ds = self.h[vid]["frames"]
        rng = np.random.default_rng(zlib.crc32(vid.encode()))  # draws always in this order
        center, jitter = clip_starts(ds.shape[0], rng)
        photo = torch.from_numpy(rng.uniform(0.85, 1.15, (SEG, 2)).astype(np.float32))
        blur = torch.from_numpy(rng.random(SEG) < 0.3)
        sigma = torch.from_numpy(rng.uniform(0.5, 1.0, SEG).astype(np.float32))
        c = self.read(ds, center) if need[[0, 1, 3]].any() else None
        j = self.read(ds, jitter) if need[2] else None
        return i, c, j, photo, blur, sigma


def gaussian_blur(x, sigma):  # x (T, 3, H, W)
    t = torch.arange(5, device=x.device, dtype=x.dtype) - 2
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, 5).expand(3, 1, 1, 5)
    x = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="reflect"), k, groups=3)
    return F.conv2d(F.pad(x, (0, 0, 2, 2), mode="reflect"), k.transpose(2, 3), groups=3)


class Embedder:
    def __init__(self, dev):
        from transformers import AutoModel, AutoVideoProcessor
        vp = AutoVideoProcessor.from_pretrained(MODEL)
        self.mean = torch.tensor(vp.image_mean, device=dev).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(vp.image_std, device=dev).view(1, 1, 3, 1, 1)
        self.model = AutoModel.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, x):  # x (SEG, T, 3, S, S) float in [0, 1] -> (SEG, D) float16
        x = ((x - self.mean) / self.std).half()  # model normalization, no resize (frames are already 224)
        out = [self.model(pixel_values_videos=x[i:i + CHUNK], skip_predictor=True).last_hidden_state.float().mean(1)
               for i in range(0, SEG, CHUNK)]
        return torch.cat(out).half().cpu().numpy()

    def views(self, c, j, photo, blur, sigma, need):
        out = {}
        to = lambda u: u.to(self.dev, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255
        if c is not None:
            c = to(c)
            if need[0]:
                out[0] = self(c)
            if need[1]:
                out[1] = self(c.flip(-1))
            if need[3]:
                p = photo.to(self.dev)
                m = c.mean(dim=(1, 2, 3, 4), keepdim=True)
                x = ((c - m) * p[:, 1].view(-1, 1, 1, 1, 1) + m) * p[:, 0].view(-1, 1, 1, 1, 1)
                x = torch.stack([gaussian_blur(x[i], float(sigma[i])) if blur[i] else x[i] for i in range(SEG)])
                out[3] = self(x.clamp(0, 1))
        if j is not None:
            out[2] = self(to(j))
        return out


def run(master, emb, n_shards, workers):
    os.makedirs(emb, exist_ok=True)
    rows = load_index(master)
    full = load_cache(os.path.join(emb, "all.npz"))
    if all(r["vid"] in full and full[r["vid"]][1].all() for r in rows):
        print("CACHE_COMPLETE: nothing to extract")
        return
    if os.environ.get("SLURM_ARRAY_TASK_ID") == "0":  # GPU utilization log, to spot loader bottlenecks
        os.makedirs("logs", exist_ok=True)
        subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used",
                          "--format=csv", "-l", "5"], stdout=open(f"logs/gpu_util_{os.environ['SLURM_JOB_ID']}.csv", "w"))
    dev = "cuda"
    emb_model = None
    for sid, shard in enumerate(shards_of(rows, int(n_shards))):
        path = os.path.join(emb, f"shard_{sid:02d}.npz")
        try:  # claim; whichever task gets here first takes it
            os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:  # ponytail: a crashed task leaves a stale .lock; delete it to redo the shard
            continue
        cache = load_cache(path)
        for r in shard:
            if r["vid"] in full and r["vid"] not in cache:
                cache[r["vid"]] = full[r["vid"]]
        todo = []
        for r in shard:
            x, done = cache.get(r["vid"], (np.zeros((V, SEG, D), np.float16), np.zeros(V, bool)))
            cache[r["vid"]] = (x, done)
            if not done.all():
                todo.append((r["vid"], ~done))
        t = time.time()
        if todo:
            emb_model = emb_model or Embedder(dev)
            loader = torch.utils.data.DataLoader(Videos(master, todo), batch_size=None, num_workers=int(workers),
                                                 prefetch_factor=2, pin_memory=True)
            for k, (i, c, j, photo, blur, sigma) in enumerate(loader):
                vid, need = todo[i]
                x, done = cache[vid]
                for v, feat in emb_model.views(c, j, photo, blur, sigma, need).items():
                    x[v], done[v] = feat, True
                if k % 20 == 19:
                    save_cache(path, cache)
        save_cache(path, cache)
        os.remove(path + ".lock")  # done shards can be re-claimed later; they just find nothing to do
        print(f"shard {sid:02d}: {len(todo)} videos in {(time.time() - t) / 60:.1f} min", flush=True)
    print("RUN_OK")


def merge(master, emb, n_shards):
    rows = load_index(master)
    cache = load_cache(os.path.join(emb, "all.npz"))
    for sid in range(int(n_shards)):
        cache.update(load_cache(os.path.join(emb, f"shard_{sid:02d}.npz")))
    zero = (np.zeros((V, SEG, D), np.float16), np.zeros(V, bool))
    got = [cache.get(r["vid"], zero) for r in rows]
    col = lambda k, f=str: np.array([f(r[k]) for r in rows])
    with open(os.path.join(emb, "all.npz.tmp"), "wb") as f:
        np.savez(f, vid=col("vid"), cls=col("cls"), split=col("split"), n_frames=col("n_frames", int),
                 n_src_frames=col("n_src_frames", int), X=np.stack([g[0] for g in got]),
                 done=np.stack([g[1] for g in got]))
    os.replace(os.path.join(emb, "all.npz.tmp"), os.path.join(emb, "all.npz"))
    missing = sum(not g[1].all() for g in got)
    print(f"MERGE_OK videos={len(rows)} incomplete={missing}")


if __name__ == "__main__":
    {"run": run, "merge": merge}[sys.argv[1]](*sys.argv[2:])
