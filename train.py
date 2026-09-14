import argparse
import os
import sys
import time
from typing import Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from maba_sparse.baselines.dense_transformer import DenseTransformerForCausalLM
from maba_sparse.config import MabaSparseConfig
from maba_sparse.model import MabaSparseForCausalLM, get_101m_config


class SyntheticLanguageDataset(Dataset):
    def __init__(self, vocab_size: int = 32768, seq_len: int = 256, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.rng = torch.Generator().manual_seed(42)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        tok = torch.randint(1, self.vocab_size, (self.seq_len,), generator=self.rng)
        tgt = torch.roll(tok, -1)
        tgt[-1] = 0
        return {"input_ids": tok, "targets": tgt}


def setup_distributed() -> tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        r = int(os.environ["RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        lr = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(lr)
        dist.init_process_group(backend=backend, rank=r, world_size=ws)
        return r, ws, lr, True
    else:
        return 0, 1, 0, False


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def build_model(
    model_type: str,
    device: torch.device,
    dim: int = 640,
    n_layers: int = 20,
    vocab_size: int = 32768,
    d_emb: int = 128,
    intermediate_size: int = 1248,
) -> nn.Module:
    if model_type.lower() in ("maba", "maba_sparse", "maba-sparse"):
        cfg = MabaSparseConfig(
            dim=dim,
            n_heads=10,
            d_head=64,
            n_layers=n_layers,
            vocab_size=vocab_size,
            d_emb=d_emb,
            intermediate_size=intermediate_size,
            residual_gate_bias=2.0,
        )
        m = MabaSparseForCausalLM(cfg)
    elif model_type.lower() in ("dense", "dense_transformer"):
        m = DenseTransformerForCausalLM(
            vocab_size=vocab_size,
            d_emb=d_emb,
            dim=dim,
            n_layers=n_layers,
            n_heads=10,
            d_head=64,
            intermediate_size=1728,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return m.to(device)


def train(args: argparse.Namespace) -> None:
    r, ws, lr, dist_flag = setup_distributed()
    main = r == 0
    dev = torch.device(f"cuda:{lr}") if torch.cuda.is_available() else torch.device("cpu")

    if main:
        print(f"=== Starting Training ===")
        print(f"Model: {args.model}")
        print(f"Device: {dev} | Distributed: {dist_flag} (World Size: {ws})")
        print(f"Steps: {args.steps} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}")

    m = build_model(
        model_type=args.model,
        device=dev,
        dim=args.dim,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        d_emb=args.d_emb,
        intermediate_size=args.intermediate_size,
    )

    pc = sum(p.numel() for p in set(m.parameters()))
    if main:
        print(f"Model Parameters: {pc:,} ({pc/1e6:.2f}M)")

    if dist_flag:
        m = DDP(m, device_ids=[lr] if torch.cuda.is_available() else None, find_unused_parameters=True)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())
    ds = SyntheticLanguageDataset(
        vocab_size=args.vocab_size, seq_len=args.seq_len, num_samples=args.steps * args.batch_size * 2
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    it = iter(loader)

    m.train()
    t0 = time.time()
    toks = 0

    for step in range(1, args.steps + 1):
        ts = time.time()
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)

        ids = b["input_ids"].to(dev)
        tgt = b["targets"].to(dev)

        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
            out = m(ids, targets=tgt)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        st = time.time() - ts
        ntoks = ids.numel() * ws
        toks += ntoks
        tp = ntoks / max(st, 1e-6)

        if main and (step % args.log_interval == 0 or step == args.steps):
            vram = 0.0
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
            print(
                f"Step {step:4d}/{args.steps} | Loss: {loss.item():.4f} | "
                f"Throughput: {tp:8.1f} tok/s | Step Time: {st*1000:6.1f}ms | "
                f"VRAM: {vram:6.1f}MB"
            )

    el = time.time() - t0
    atp = toks / max(el, 1e-6)

    if main:
        print("=== Training Complete ===")
        print(f"Total Steps: {args.steps} | Total Time: {el:.2f}s")
        print(f"Average Throughput: {atp:.1f} tokens/sec")
        print(f"Final Step Loss: {loss.item():.4f}")

    cleanup_distributed(dist_flag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maba-Sparse Multi-GPU / Local Training")
    parser.add_argument("--model", type=str, default="maba_sparse", choices=["maba_sparse", "dense"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=20)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--d_emb", type=int, default=128)
    parser.add_argument("--intermediate_size", type=int, default=1248)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
