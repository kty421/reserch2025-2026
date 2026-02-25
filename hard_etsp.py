import argparse, math, os, json, random, time, csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 乱数固定
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pairwise_euclid(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))

def hash_coords(xy: np.ndarray) -> str:
    return f"{xy.shape[0]}:" + np.round(xy.flatten(), 6).tobytes().hex()

def ensure_dir(d: str):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Gurobiソルバ
def solve_etsp(
    coords: np.ndarray,
    timelimit: float = 15.0,
    threads: int = 1,
    mipfocus: int = 1,
    seed: int = 0,
    heur: float = 0.01,
    cuts: int = 2,
    presolve: int = 2
) -> Dict[str, float]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        return dict(obj=float("inf"), runtime=timelimit, nodecount=0.0, lazy=0.0, mipgap=float("inf"), solved=0.0)

    n = coords.shape[0]
    D = pairwise_euclid(coords)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    try:
        m = gp.Model("etsp")
        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", timelimit)
        m.setParam("Threads", threads)
        m.setParam("MIPFocus", mipfocus)
        m.setParam("Seed", seed)
        m.setParam("Heuristics", heur)
        m.setParam("Cuts", cuts)
        m.setParam("Presolve", presolve)

        x = m.addVars(edges, vtype=GRB.BINARY, name="x")
        m.setObjective(gp.quicksum(D[i, j] * x[i, j] for (i, j) in edges), GRB.MINIMIZE)

        for i in range(n):
            m.addConstr(gp.quicksum(x[i, j] if i < j else x[j, i] for j in range(n) if j != i) == 2)

        lazy_count = 0

        # サブツアー検出
        def find_cycles(selected_edges):
            adj = [[] for _ in range(n)]
            for (i, j) in selected_edges:
                adj[i].append(j); adj[j].append(i)
            visited = [False]*n; cycles = []
            for s in range(n):
                if visited[s]: continue
                cur = s; prev = -1; cyc = []
                while not visited[cur]:
                    visited[cur] = True; cyc.append(cur)
                    nbr = adj[cur]
                    if not nbr: break
                    nxt = nbr[0] if nbr[0] != prev else (nbr[1] if len(nbr)>1 else nbr[0])
                    prev, cur = cur, nxt
                cycles.append(cyc)
            return cycles

        # Lazy Constraint追加
        def cb(model, where):
            nonlocal lazy_count
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(x)
                chosen = [(i,j) for (i,j) in edges if vals[i,j] > 0.5]
                cycles = find_cycles(chosen)
                for cyc in cycles:
                    if len(cyc) < n:
                        lazy_count += 1
                        cut = []
                        for i in cyc:
                            for j in cyc:
                                if i < j and (i, j) in x: cut.append((i, j))
                        model.cbLazy(gp.quicksum(x[i, j] for (i, j) in cut) <= len(cyc) - 1)

        m.Params.LazyConstraints = 1
        m.optimize(cb)

        status = m.Status
        solved_opt = 1.0 if status == GRB.OPTIMAL else 0.0
        obj = float(m.ObjVal) if m.SolCount > 0 else float("inf")
        mipgap = float(m.MIPGap) if m.SolCount > 0 and math.isfinite(m.MIPGap) else float("inf")
        return dict(
            obj=obj,
            runtime=float(m.Runtime),
            nodecount=float(m.NodeCount),
            lazy=float(lazy_count),
            mipgap=mipgap,
            solved=solved_opt
        )
    except Exception:
        return dict(obj=float("inf"), runtime=timelimit, nodecount=0.0, lazy=0.0, mipgap=float("inf"), solved=0.0)

# 方策ネットワーク
class GMMPolicy(nn.Module):
    def __init__(self, K: int = 8, min_scale: float = 5e-4,
                 scale_min: float = 0.0003, scale_max: float = 0.2):
        super().__init__()
        self.K = K
        self.centers_raw = nn.Parameter(torch.randn(K, 2) * 0.1)
        self.log_scales  = nn.Parameter(torch.full((K, 2), -2.0))
        self.theta_raw   = nn.Parameter(torch.zeros(K))
        self.logits      = nn.Parameter(torch.zeros(K))
        self.min_scale = min_scale
        self.scale_min = scale_min
        self.scale_max = scale_max

    def params_constrained(self):
        centers = torch.sigmoid(self.centers_raw)
        scales  = torch.nn.functional.softplus(self.log_scales) + self.min_scale
        theta   = math.pi * torch.tanh(self.theta_raw)
        weights = torch.softmax(self.logits, dim=0)
        return centers, scales, theta, weights

    @torch.no_grad()
    def sample_instances(self, n_points: int, batch_size: int,
                         antithetic: bool = True, device: str = "cpu"):
        centers, scales, theta, weights = self.params_constrained()
        centers, scales, theta, weights = centers.to(device), scales.to(device), theta.to(device), weights.to(device)
        cat = torch.distributions.Categorical(probs=weights)

        def rot(t):
            c = torch.cos(t); s = torch.sin(t)
            return torch.stack([torch.stack([c, -s], -1),
                                torch.stack([s,  c], -1)], -2)
        R = rot(theta)

        coords_list, idx_list, noise_list = [], [], []
        half = batch_size // 2 if antithetic else batch_size

        for _ in range(half):
            z = cat.sample((n_points,))
            eps = torch.randn(n_points, 2, device=device)
            s  = scales[z]
            mu = centers[z]
            Rz = R[z]
            x = mu + torch.bmm(Rz, (s * eps).unsqueeze(-1)).squeeze(-1)
            x = torch.clamp(x, 0.0, 1.0)
            coords_list.append(x.cpu()); idx_list.append(z.cpu()); noise_list.append(eps.cpu())

            if antithetic:
                eps2 = -eps
                x2 = mu + torch.bmm(Rz, (s * eps2).unsqueeze(-1)).squeeze(-1)
                x2 = torch.clamp(x2, 0.0, 1.0)
                coords_list.append(x2.cpu()); idx_list.append(z.cpu()); noise_list.append(eps2.cpu())

        while len(coords_list) < batch_size:
            coords_list.append(coords_list[-1].clone())
            idx_list.append(idx_list[-1].clone())
            noise_list.append(noise_list[-1].clone())

        info = {
            "centers": centers.cpu().numpy().tolist(),
            "scales":  scales.cpu().numpy().tolist(),
            "theta":   theta.cpu().numpy().tolist(),
            "weights": weights.cpu().numpy().tolist(),
        }
        return coords_list[:batch_size], idx_list[:batch_size], noise_list[:batch_size], info

    def log_prob_batch(self, idx_list: List[torch.Tensor], noise_list: List[torch.Tensor]) -> torch.Tensor:
        _, scales, _, weights = self.params_constrained()
        device = self.centers_raw.device
        logw = torch.log(weights + 1e-9)
        logps = []
        for z, eps in zip(idx_list, noise_list):
            z = z.to(device); eps = eps.to(device)
            s = scales[z]
            logdet = torch.log(s[:, 0] * s[:, 1] + 1e-12)
            logp_x_given_z = (- math.log(2 * math.pi) - logdet - 0.5 * (eps ** 2).sum(-1)).sum()
            logp_z = logw[z].sum()
            logps.append(logp_x_given_z + logp_z)
        return torch.stack(logps)

# 報酬計算
@dataclass
class RewardWeights:
    w_rt: float = 1.0
    w_nd: float = 0.6
    w_lz: float = 0.6
    w_gap: float = 0.0
    unsolved_bonus: float = 0.3

def make_reward(metric: Dict[str, float], w: RewardWeights, timeout_boost: float = 1.0) -> float:
    r = 0.0
    rt = max(0.0, metric.get("runtime", 0.0))
    nd = max(0.0, metric.get("nodecount", 0.0))
    lz = max(0.0, metric.get("lazy", 0.0))
    gap = metric.get("mipgap", float("inf"))
    solved = metric.get("solved", 0.0)

    w_nd_use = w.w_nd * (timeout_boost if solved < 0.5 else 1.0)
    w_lz_use = w.w_lz * (timeout_boost if solved < 0.5 else 1.0)

    r += w.w_rt * math.log1p(rt)
    r += w_nd_use * math.log1p(nd)
    r += w_lz_use * math.log1p(lz)

    if w.w_gap != 0.0 and math.isfinite(gap) and gap > 0:
        r += w.w_gap * math.log1p(gap * 100.0)
    if solved < 0.5:
        r += w.unsolved_bonus
    return r

def make_reward_nodes(metric: Dict[str, float]) -> float:
    nd = max(0.0, metric.get("nodecount", 0.0))
    return math.log1p(nd)

def make_reward_lazy(metric: Dict[str, float]) -> float:
    lz = max(0.0, metric.get("lazy", 0.0))
    return math.log1p(lz)

def centered_ranks(x: torch.Tensor) -> torch.Tensor:
    y = torch.argsort(torch.argsort(x))
    r = y.float() / (x.numel() - 1 + 1e-9)
    return r - 0.5

def mst_bottleneck_length(coords: np.ndarray) -> float:
    n = coords.shape[0]
    D = pairwise_euclid(coords)
    in_mst = np.zeros(n, dtype=bool); dist = np.full(n, np.inf); parent = np.full(n, -1)
    dist[0] = 0.0; max_edge = 0.0
    for _ in range(n):
        u = np.argmin(dist + np.where(in_mst, np.inf, 0.0))
        in_mst[u] = True
        if parent[u] != -1: max_edge = max(max_edge, D[u, parent[u]])
        for v in range(n):
            if not in_mst[v] and D[u, v] < dist[v]:
                dist[v] = D[u, v]; parent[v] = u
    return float(max_edge)

def nn_distance_variance(coords: np.ndarray) -> float:
    D = pairwise_euclid(coords); np.fill_diagonal(D, np.inf)
    return float(np.var(D.min(axis=1)))

# 画像保存用
def save_points_grid(samples: List[np.ndarray], path: str, ncols: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(samples)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = 3.2*ncols; fig_h = 3.2*nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, xy in enumerate(samples):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.scatter(xy[:,0], xy[:,1], s=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"#{i}", fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_curves(steps, series_dict, xlabel, ylabel, title, path):
    if len(steps) == 0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, values in series_dict.items():
        if len(values) != len(steps):
            continue
        plt.plot(steps, values, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if len(series_dict) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

# メインループ
def train(args):
    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"
    if device == "cuda" and args.allow_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    autocast_ctx = torch.amp.autocast(
        device_type="cuda", dtype=torch.float16, enabled=(device == "cuda" and args.amp)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and args.amp))

    set_seed(args.seed)

    policy = GMMPolicy(K=args.k, min_scale=5e-4, scale_min=args.scale_min, scale_max=args.scale_max).to(device)
    opt = optim.Adam(policy.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * args.lr_decay) \
        if args.lr_decay < 1.0 else None

    start_step = 1
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        if "state_dict" in ckpt:
            policy.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            try: opt.load_state_dict(ckpt["optimizer"])
            except Exception: pass
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"[resume] loaded: {args.resume} (step={start_step-1})")

    rw = RewardWeights(
        w_rt=args.w_rt, w_nd=args.w_nd, w_lz=args.w_lz,
        w_gap=args.w_gap, unsolved_bonus=args.unsolved_bonus
    )

    entropy_coeff = args.entropy
    repulsion_w = args.repulsion_w

    metric_cache: "OrderedDict[str, Dict[str,float]]" = OrderedDict()
    cache_size = max(0, args.cache_size)

    ema_r: Optional[float] = None
    best_avg = -1e9
    best_tail_score = -1e9
    ensure_dir(args.outdir)

    csv_path = os.path.join(args.outdir, f"metrics_{args.reward_mode}.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step","mode","R_mean","R_std",
                "avg_rt","avg_nodes","avg_lazy","avg_gap","avg_solved",
                "tail_rt","tail_nodes","tail_lazy","tail_solved",
                "loss","Hmix","rep","scale_reg",
                "entropy_coeff","repulsion_w",
                "scale_mean","scale_min","scale_max","center_pairdist_mean"
            ])

    with torch.no_grad():
        before_batch, _, _, _ = policy.sample_instances(
            n_points=args.n, batch_size=min(args.viz_samples, args.batch*2),
            antithetic=not args.no_antithetic, device=device
        )
    before_np = [t.cpu().numpy() for t in before_batch]
    save_points_grid(before_np, os.path.join(args.outdir, f"before_after_points_step000001.png"))

    hist_step = []
    hist_R_mean = []
    hist_avg_rt = []
    hist_avg_nodes = []
    hist_avg_lazy = []
    hist_Hmix = []
    hist_rep = []
    hist_scale_mean = []
    hist_scale_min = []
    hist_scale_max = []
    hist_center_pairdist = []
    hist_entropy_coeff = []
    hist_repulsion_w = []

    topq = args.topq_start

    for step in range(start_step, args.steps + 1):
        adv_temp = args.adv_temp * (args.adv_temp_decay ** (step-1))

        coords_batch, idx_batch, noise_batch, info = policy.sample_instances(
            n_points=args.n, batch_size=args.batch, antithetic=not args.no_antithetic, device=device
        )

        rewards = []
        metrics_list = []
        for xy_t in coords_batch:
            xy = xy_t.detach().cpu().numpy()
            key = hash_coords(xy)

            if cache_size > 0 and key in metric_cache:
                m = metric_cache.pop(key)
                metric_cache[key] = m
            else:
                m = solve_etsp(
                    xy,
                    timelimit=args.timelimit, threads=args.threads,
                    mipfocus=args.mipfocus, seed=args.seed, heur=args.heuristics,
                    cuts=args.cuts, presolve=args.presolve
                )
                if cache_size > 0:
                    metric_cache[key] = m
                    if len(metric_cache) > cache_size:
                        metric_cache.popitem(last=False)

            if args.reward_mode == "nodes":
                base = make_reward_nodes(m)
            elif args.reward_mode == "lazy":
                base = make_reward_lazy(m)
            else:
                base = make_reward(m, rw, timeout_boost=args.timeout_boost)

            if args.reward_mode == "all":
                if args.alpha_mst > 0:
                    base += args.alpha_mst * math.log1p(mst_bottleneck_length(xy))
                if args.alpha_nnvar > 0:
                    base += args.alpha_nnvar * math.log1p(nn_distance_variance(xy))

            rewards.append(base); metrics_list.append(m)

        r_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        batch_mean = float(r_t.mean()); batch_std = float(r_t.std())

        if ema_r is None: ema_r = batch_mean
        else: ema_r = args.ema_alpha * batch_mean + (1.0 - args.ema_alpha) * ema_r

        topq = max(args.topq_end, topq * args.topq_decay)
        q = min(max(topq, 1/len(r_t)), 0.99)
        thr = torch.quantile(r_t, 1.0 - q)
        tail_mask = (r_t >= thr).float()

        ranks = centered_ranks(r_t)
        adv = ((r_t - ema_r) / (r_t.std() + 1e-6)) * adv_temp

        weights_pg = (args.rank_mix * ranks + (1.0 - args.rank_mix) * adv) * tail_mask
        weights_pg = (weights_pg - weights_pg.mean()) / (weights_pg.std() + 1e-6)

        logps = policy.log_prob_batch(idx_batch, noise_batch).to(device)
        loss_pg = -(weights_pg.detach() * logps).mean()

        centers, scales, theta, mix_w = policy.params_constrained()
        ent_mix = - (mix_w * torch.log(mix_w + 1e-9)).sum()

        if args.repulsion_tau > 0:
            C = centers
            d2 = ((C.unsqueeze(1) - C.unsqueeze(0))**2).sum(-1)
            rep_kernel = torch.exp(-d2 / args.repulsion_tau)
            repulsion = (rep_kernel.sum() - torch.diag(rep_kernel).sum()) / (C.shape[0]*(C.shape[0]-1)+1e-9)
        else:
            repulsion = torch.tensor(0.0, device=device)

        s = scales
        scale_reg = ((torch.relu(args.scale_min - s))**2 + (torch.relu(s - args.scale_max))**2).mean()

        loss = loss_pg \
               - max(entropy_coeff, args.entropy_min) * ent_mix \
               + repulsion_w * repulsion \
               + args.scale_reg_w * scale_reg

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            with autocast_ctx:
                loss_mixed = loss
            scaler.scale(loss_mixed).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            opt.step()

        entropy_coeff *= args.entropy_decay
        repulsion_w *= args.repulsion_decay
        if scheduler is not None:
            scheduler.step()

        tail_idx = (r_t >= thr).detach().cpu().numpy().astype(bool)
        if any(tail_idx):
            t_nodes = float(np.mean([m["nodecount"] for i,m in enumerate(metrics_list) if tail_idx[i]]))
            t_runtime = float(np.mean([m["runtime"]  for i,m in enumerate(metrics_list) if tail_idx[i]]))
            t_lazy = float(np.mean([m["lazy"]     for i,m in enumerate(metrics_list) if tail_idx[i]]))
            t_solved = float(np.mean([m["solved"]  for i,m in enumerate(metrics_list) if tail_idx[i]]))
        else:
            t_nodes = t_runtime = t_lazy = t_solved = 0.0

        mean_metric = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
        avg_rt, avg_nodes = mean_metric["runtime"], mean_metric["nodecount"]
        avg_lazy, avg_gap, avg_solved = mean_metric["lazy"], mean_metric["mipgap"], mean_metric["solved"]

        with torch.no_grad():
            scale_mean_val = float(scales.mean().detach().cpu())
            scale_min_val = float(scales.min().detach().cpu())
            scale_max_val = float(scales.max().detach().cpu())
            C_np = centers.detach().cpu().numpy()
            if C_np.shape[0] > 1:
                D_c = pairwise_euclid(C_np)
                center_pairdist_mean = float(D_c[np.triu_indices(C_np.shape[0], k=1)].mean())
            else:
                center_pairdist_mean = 0.0

        hist_step.append(step)
        hist_R_mean.append(batch_mean)
        hist_avg_rt.append(avg_rt)
        hist_avg_nodes.append(avg_nodes)
        hist_avg_lazy.append(avg_lazy)
        hist_Hmix.append(float(ent_mix.detach().cpu()))
        hist_rep.append(float(repulsion.detach().cpu()))
        hist_scale_mean.append(scale_mean_val)
        hist_scale_min.append(scale_min_val)
        hist_scale_max.append(scale_max_val)
        hist_center_pairdist.append(center_pairdist_mean)
        hist_entropy_coeff.append(float(max(entropy_coeff, args.entropy_min)))
        hist_repulsion_w.append(float(repulsion_w))

        if (step % args.logint == 0) or (step == 1):
            print(
                f"[{step}/{args.steps}] mode={args.reward_mode} "
                f"R(mean)={batch_mean:.3f} R(std)={batch_std:.3f} "
                f"| avg: rt={avg_rt:.3f}s nodes={avg_nodes:.1f} "
                f"lazy={avg_lazy:.1f} gap={avg_gap:.4f} solved={avg_solved:.2f} "
                f"| TAIL(q={q:.2f}): rt={t_runtime:.3f}s nodes={t_nodes:.1f} lazy={t_lazy:.1f} solved={t_solved:.2f} "
                f"| loss={float(loss):.3f} Hmix={float(ent_mix):.3f} rep={float(repulsion):.3f} sreg={float(scale_reg):.3f} "
                f"| entC={max(entropy_coeff, args.entropy_min):.4f} lr={opt.param_groups[0]['lr']:.2e}"
            )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, args.reward_mode, batch_mean, batch_std,
                avg_rt, avg_nodes, avg_lazy, avg_gap, avg_solved,
                t_runtime, t_nodes, t_lazy, t_solved,
                float(loss), float(ent_mix), float(repulsion), float(scale_reg),
                float(max(entropy_coeff, args.entropy_min)), float(repulsion_w),
                scale_mean_val, scale_min_val, scale_max_val, center_pairdist_mean
            ])

        if args.snapint > 0 and (step % args.snapint == 0):
            np.save(os.path.join(args.outdir, f"rewards_step{step:06d}.npy"),
                    r_t.detach().cpu().numpy())
            samp_np = [c.detach().cpu().numpy() for c in coords_batch]
            save_points_grid(samp_np, os.path.join(args.outdir, f"points_step{step:06d}.png"))
            with open(os.path.join(args.outdir, f"metrics_step{step:06d}.jsonl"), "w", encoding="utf-8") as jf:
                for met in metrics_list: jf.write(json.dumps(met) + "\n")

        avg_r = float(r_t.mean())
        if avg_r > best_avg:
            best_avg = avg_r
            ckpt = {"step": step, "state_dict": policy.state_dict(), "info": info,
                    "best_avg": best_avg, "args": vars(args), "optimizer": opt.state_dict()}
            torch.save(ckpt, os.path.join(args.outdir, f"best_{args.reward_mode}.pt"))

        score_tail = 0.5 * (t_nodes / max(1.0, args.tail_nodes_norm)) \
                   + 0.5 * (t_runtime / max(1e-9, args.timelimit))
        if score_tail > best_tail_score:
            best_tail_score = score_tail
            ckpt = {"step": step, "state_dict": policy.state_dict(), "info": info,
                    "best_tail": best_tail_score, "args": vars(args), "optimizer": opt.state_dict()}
            torch.save(ckpt, os.path.join(args.outdir, f"best_tail_{args.reward_mode}.pt"))

        if args.saveint > 0 and (step % args.saveint == 0):
            torch.save({"step": step, "state_dict": policy.state_dict(), "args": vars(args),
                        "optimizer": opt.state_dict()},
                       os.path.join(args.outdir, f"step_{args.reward_mode}_{step:06d}.pt"))

    with torch.no_grad():
        after_batch, _, _, _ = policy.sample_instances(
            n_points=args.n, batch_size=min(args.viz_samples, args.batch*2),
            antithetic=not args.no_antithetic, device=device
        )
    after_np = [t.cpu().numpy() for t in after_batch]
    save_points_grid(after_np, os.path.join(args.outdir, f"before_after_points_last_{args.reward_mode}.png"))

    torch.save({"step": args.steps, "state_dict": policy.state_dict(), "args": vars(args),
                "optimizer": opt.state_dict()},
               os.path.join(args.outdir, f"last_{args.reward_mode}.pt"))

    plot_curves(
        hist_step,
        {
            "Hmix": hist_Hmix,
            "scale_mean": hist_scale_mean,
            "center_pairdist": hist_center_pairdist,
            "entropy_coeff": hist_entropy_coeff,
            "repulsion_w": hist_repulsion_w,
        },
        xlabel="step",
        ylabel="value",
        title=f"Distribution Hyperparams ({args.reward_mode})",
        path=os.path.join(args.outdir, f"curve_hparams_{args.reward_mode}.png"),
    )

    plot_curves(
        hist_step,
        {
            "R_mean": hist_R_mean,
            "runtime": hist_avg_rt,
            "nodes": hist_avg_nodes,
            "lazy": hist_avg_lazy,
        },
        xlabel="step",
        ylabel="value",
        title=f"Reward / Metrics ({args.reward_mode})",
        path=os.path.join(args.outdir, f"curve_reward_{args.reward_mode}.png"),
    )

# 評価用
def load_policy(path: str, k: int):
    ckpt = torch.load(path, map_location="cpu")
    p = GMMPolicy(K=k)
    p.load_state_dict(ckpt["state_dict"])
    return p, ckpt

def evaluate(args):
    p, _ = load_policy(args.ckpt, args.k)
    coords_batch, _, _, _ = p.sample_instances(n_points=args.n, batch_size=args.n_samples, device="cpu")
    ensure_dir(args.outdir)
    all_metrics = []
    samp_np = []
    for i, xy in enumerate(coords_batch):
        xy = xy.numpy()
        met = solve_etsp(
            xy,
            timelimit=args.timelimit, threads=args.threads,
            mipfocus=args.mipfocus, seed=args.seed, heur=args.heuristics,
            cuts=args.cuts, presolve=args.presolve
        )
        all_metrics.append(met); samp_np.append(xy)
        np.savetxt(os.path.join(args.outdir, f"coords_{i:03d}.txt"), xy, fmt="%.6f")

    with open(os.path.join(args.outdir, f"eval_metrics_{args.reward_mode}.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    save_points_grid(samp_np, os.path.join(args.outdir, f"eval_points_{args.reward_mode}.png"))

    mean_metric = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}
    print("[EVAL] ",
          f"rt={mean_metric['runtime']:.3f}s nodes={mean_metric['nodecount']:.1f} "
          f"lazy={mean_metric['lazy']:.1f} gap={mean_metric['mipgap']:.4f} "
          f"solved={mean_metric['solved']:.2f}")

# パラメータ設定
def build_argparser():
    p = argparse.ArgumentParser(description="TSP Hard Instance Generator")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--k", type=int, default=8)

    p.add_argument("--reward_mode", type=str, default="all", choices=["all","nodes","lazy"])
    p.add_argument("--exp_tag", type=str, default="")

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lr_decay", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--entropy_decay", type=float, default=0.995)
    p.add_argument("--entropy_min", type=float, default=0.005)

    p.add_argument("--repulsion_w", type=float, default=0.01)
    p.add_argument("--repulsion_tau", type=float, default=0.08)
    p.add_argument("--repulsion_decay", type=float, default=0.998)
    p.add_argument("--scale_reg_w", type=float, default=0.01)
    p.add_argument("--scale_min", type=float, default=0.0003)
    p.add_argument("--scale_max", type=float, default=0.2)
    p.add_argument("--no_antithetic", action="store_true")

    p.add_argument("--adv_temp", type=float, default=2.0)
    p.add_argument("--adv_temp_decay", type=float, default=0.999)

    p.add_argument("--rank_mix", type=float, default=0.5)

    p.add_argument("--topq_start", type=float, default=0.25)
    p.add_argument("--topq_end", type=float, default=0.05)
    p.add_argument("--topq_decay", type=float, default=0.995)

    p.add_argument("--ema_alpha", type=float, default=0.10)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--logint", type=int, default=10)
    p.add_argument("--saveint", type=int, default=0)
    p.add_argument("--cache_size", type=int, default=5000)
    p.add_argument("--snapint", type=int, default=50)

    g = p.add_argument_group("reward weights")
    g.add_argument("--w_rt", type=float, default=1.0)
    g.add_argument("--w_nd", type=float, default=0.6)
    g.add_argument("--w_lz", type=float, default=0.6)
    g.add_argument("--w_gap", type=float, default=0.0)
    g.add_argument("--unsolved_bonus", type=float, default=0.3)
    g.add_argument("--timeout_boost", type=float, default=1.5)

    p.add_argument("--alpha_mst", type=float, default=0.05)
    p.add_argument("--alpha_nnvar", type=float, default=0.05)

    p.add_argument("--timelimit", type=float, default=60.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--mipfocus", type=int, default=1)
    p.add_argument("--heuristics", type=float, default=0.01)
    p.add_argument("--cuts", type=int, default=2)
    p.add_argument("--presolve", type=int, default=2)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--allow_cudnn_benchmark", action="store_true")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="./hardgen_out")
    p.add_argument("--resume", type=str, default="")

    p.add_argument("--viz_samples", type=int, default=12)

    p.add_argument("--eval", action="store_true")
    p.add_argument("--ckpt", type=str, default="./hardgen_out/best_tail_all.pt")
    p.add_argument("--n_samples", type=int, default=12)

    p.add_argument("--tail_nodes_norm", type=float, default=1000.0)

    return p

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
