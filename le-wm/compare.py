"""
Evaluation script for Transformer vs RWKV-6 Predictor

用法:
    python eval_predictors.py --transformer_ckpt <path> --rwkv_ckpt <path> --config config/train/lewm.yaml
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import OmegaConf

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from rwkv_module import RWKV_ARPredictor
from utils import get_column_normalizer, get_img_preprocessor


class PredictorEvaluator:
    """评估 Predictor 的性能"""

    def __init__(self, device="cuda"):
        self.device = device

    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def benchmark_inference(self, model: nn.Module, batch_size: int, seq_len: int,
                           hidden_dim: int, num_runs: int = 100) -> Dict[str, float]:
        """基准测试推理速度和显存"""
        model.eval()

        x = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
        c = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, c)

        torch.cuda.synchronize()

        # 测量延迟
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x, c)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        latency_ms = np.mean(times) * 1000
        latency_std = np.std(times) * 1000
        throughput = batch_size / np.mean(times)

        # 显存
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x, c)
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

        return {
            "latency_ms": latency_ms,
            "latency_std_ms": latency_std,
            "throughput_samples_per_sec": throughput,
            "gpu_memory_peak_mb": gpu_memory_mb,
        }

    def evaluate_prediction_accuracy(self, model: nn.Module, val_loader: DataLoader,
                                     num_batches: int = 50) -> Dict[str, float]:
        """评估预测精度"""
        model.eval()

        losses_1step = []
        losses_multistep = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                emb = batch['emb'].to(self.device)  # (B, T, D)
                act_emb = batch['act_emb'].to(self.device)

                if emb.size(1) < 2:
                    continue

                # 单步预测: 用前 T-1 时间步预测第 T 时间步
                ctx_emb = emb[:, :-1]
                ctx_act = act_emb[:, :-1]
                tgt_emb = emb[:, 1:]

                pred_emb = model(ctx_emb, ctx_act)

                # 单步误差（只看最后一步）
                loss_1step = F.mse_loss(pred_emb[:, -1:], tgt_emb[:, -1:])
                losses_1step.append(loss_1step.item())

                # 多步误差（所有步）
                loss_multistep = F.mse_loss(pred_emb, tgt_emb)
                losses_multistep.append(loss_multistep.item())

        return {
            "mse_1step": np.mean(losses_1step) if losses_1step else 0.0,
            "mse_multistep": np.mean(losses_multistep) if losses_multistep else 0.0,
            "rmse_1step": np.sqrt(np.mean(losses_1step)) if losses_1step else 0.0,
            "rmse_multistep": np.sqrt(np.mean(losses_multistep)) if losses_multistep else 0.0,
        }

    def analyze_gradient_flow(self, model: nn.Module, batch: Dict,
                             num_batches: int = 5) -> Dict[str, float]:
        """分析梯度流"""
        model.train()
        grad_norms = []

        for _ in range(num_batches):
            model.zero_grad()

            emb = batch['emb'].to(self.device)
            act_emb = batch['act_emb'].to(self.device)

            ctx_emb = emb[:, :-1]
            ctx_act = act_emb[:, :-1]
            tgt_emb = emb[:, 1:]

            pred_emb = model(ctx_emb, ctx_act)
            loss = F.mse_loss(pred_emb, tgt_emb)

            loss.backward()

            # 计算总梯度范数
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = np.sqrt(total_norm)
            grad_norms.append(total_norm)

        return {
            "gradient_norm_mean": np.mean(grad_norms),
            "gradient_norm_max": np.max(grad_norms),
            "gradient_norm_min": np.min(grad_norms),
        }


def load_data(config):
    """加载数据"""
    dataset = swm.data.HDF5Dataset(**config.data.dataset, transform=None)

    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=config.img_size)]
    for col in config.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        normalizer = get_column_normalizer(dataset, col, col)
        transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(config.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[config.train_split, 1 - config.train_split], generator=rnd_gen
    )

    val_loader = DataLoader(
        val_set, batch_size=config.loader.batch_size, shuffle=False, num_workers=0
    )

    return val_loader


def build_predictor(predictor_type: str, config):
    """构建 Predictor"""
    predictor_kwargs = {
        "num_frames": config.wm.history_size,
        "input_dim": config.wm.embed_dim,
        "hidden_dim": 768,
        "output_dim": 768,
        "depth": config.predictor.depth,
        "heads": config.predictor.heads,
        "mlp_dim": config.predictor.mlp_dim,
        "dim_head": config.predictor.dim_head,
        "dropout": config.predictor.dropout,
        "emb_dropout": config.predictor.emb_dropout,
    }

    if predictor_type.lower() == "rwkv":
        return RWKV_ARPredictor(**predictor_kwargs)
    else:
        return ARPredictor(**predictor_kwargs)


def evaluate_predictor(predictor_type: str, config, device="cuda"):
    """评估单个 Predictor"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {predictor_type.upper()}")
    print(f"{'='*70}\n")

    # 构建模型
    model = build_predictor(predictor_type, config).to(device)
    evaluator = PredictorEvaluator(device=device)

    # 加载数据
    val_loader = load_data(config)

    # 评估项
    print("[1/4] 计算模型参数...")
    num_params = evaluator.count_parameters(model)
    print(f"  Parameters: {num_params:,}")

    print("[2/4] 基准测试推理性能...")
    inference_metrics = evaluator.benchmark_inference(
        model,
        batch_size=config.loader.batch_size,
        seq_len=config.wm.history_size,
        hidden_dim=config.wm.embed_dim,
        num_runs=100
    )
    for key, val in inference_metrics.items():
        print(f"  {key}: {val:.4f}")

    print("[3/4] 评估预测精度...")
    accuracy_metrics = evaluator.evaluate_prediction_accuracy(model, val_loader, num_batches=50)
    for key, val in accuracy_metrics.items():
        print(f"  {key}: {val:.6f}")

    print("[4/4] 分析梯度流...")
    # 获取一个批次用于梯度分析
    batch = next(iter(val_loader))
    gradient_metrics = evaluator.analyze_gradient_flow(model, batch, num_batches=5)
    for key, val in gradient_metrics.items():
        print(f"  {key}: {val:.6f}")

    # 汇总结果
    results = {
        "model_type": predictor_type,
        "num_parameters": int(num_params),
        **inference_metrics,
        **accuracy_metrics,
        **gradient_metrics,
    }

    return results


def format_comparison_table(results_dict: Dict) -> str:
    """生成对比表格"""
    transformer = results_dict.get("transformer", {})
    rwkv = results_dict.get("rwkv", {})

    if not transformer or not rwkv:
        return ""

    lines = [
        "",
        "=" * 90,
        "COMPREHENSIVE BENCHMARK COMPARISON: TRANSFORMER vs RWKV-6",
        "=" * 90,
        "",
        "| Metric | Transformer | RWKV-6 | Difference | Winner |",
        "|--------|-------------|--------|-----------|--------|",
    ]

    # 模型参数
    trans_params = transformer.get("num_parameters", 0)
    rwkv_params = rwkv.get("num_parameters", 0)
    lines.append(f"| Parameters | {trans_params:,} | {rwkv_params:,} | {trans_params-rwkv_params:+,} | {'Tie' if trans_params == rwkv_params else ('RWKV' if rwkv_params < trans_params else 'Transformer')} |")

    # 推理延迟
    trans_lat = transformer.get("latency_ms", 0)
    rwkv_lat = rwkv.get("latency_ms", 0)
    speedup = trans_lat / rwkv_lat if rwkv_lat > 0 else 1.0
    lines.append(f"| Latency (ms) | {trans_lat:.3f} | {rwkv_lat:.3f} | {trans_lat-rwkv_lat:+.3f} ({speedup:.2f}x) | {'RWKV' if rwkv_lat < trans_lat else 'Transformer'} |")

    # 吞吐量
    trans_thr = transformer.get("throughput_samples_per_sec", 0)
    rwkv_thr = rwkv.get("throughput_samples_per_sec", 0)
    lines.append(f"| Throughput (samples/s) | {trans_thr:.1f} | {rwkv_thr:.1f} | {rwkv_thr-trans_thr:+.1f} | {'RWKV' if rwkv_thr > trans_thr else 'Transformer'} |")

    # 显存
    trans_mem = transformer.get("gpu_memory_peak_mb", 0)
    rwkv_mem = rwkv.get("gpu_memory_peak_mb", 0)
    lines.append(f"| GPU Memory (MB) | {trans_mem:.1f} | {rwkv_mem:.1f} | {trans_mem-rwkv_mem:+.1f} | {'RWKV' if rwkv_mem < trans_mem else 'Transformer'} |")

    # 单步MSE
    trans_mse = transformer.get("mse_1step", 0)
    rwkv_mse = rwkv.get("mse_1step", 0)
    lines.append(f"| MSE (1-step) | {trans_mse:.6f} | {rwkv_mse:.6f} | {trans_mse-rwkv_mse:+.6f} | {'RWKV' if rwkv_mse < trans_mse else 'Transformer'} |")

    # 多步MSE
    trans_multi = transformer.get("mse_multistep", 0)
    rwkv_multi = rwkv.get("mse_multistep", 0)
    lines.append(f"| MSE (multi-step) | {trans_multi:.6f} | {rwkv_multi:.6f} | {trans_multi-rwkv_multi:+.6f} | {'RWKV' if rwkv_multi < trans_multi else 'Transformer'} |")

    # 梯度流
    trans_grad = transformer.get("gradient_norm_mean", 0)
    rwkv_grad = rwkv.get("gradient_norm_mean", 0)
    lines.append(f"| Avg Gradient Norm | {trans_grad:.6f} | {rwkv_grad:.6f} | {trans_grad-rwkv_grad:+.6f} | {'RWKV' if rwkv_grad > 0 and (trans_grad == 0 or rwkv_grad < trans_grad) else 'Transformer'} |")

    lines.extend([
        "",
        "=" * 90,
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Transformer vs RWKV-6 Predictor")
    parser.add_argument("--config", type=str, default="config/train/lewm.yaml",
                       help="Config file path")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    args = parser.parse_args()

    # 加载配置
    cfg = OmegaConf.load(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # 评估两个模型
    results = {}

    try:
        results["transformer"] = evaluate_predictor("transformer", cfg, device=device)
    except Exception as e:
        print(f"Error evaluating Transformer: {e}")

    try:
        results["rwkv"] = evaluate_predictor("rwkv", cfg, device=device)
    except Exception as e:
        print(f"Error evaluating RWKV: {e}")

    # 输出对比表格
    comparison = format_comparison_table(results)
    print(comparison)

    # 保存结果
    output_file = Path("evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
