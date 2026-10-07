import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from evaluation import evaluate
from model import DigitTransformer
from reinforce import Reinforce
from rewards import make_reward
from sft import sft_step
from tasks import make_task, split_dataset


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    training = cfg.training
    if (training.batch_size < 1 or training.weight_decay < 0
            or training.max_grad_norm <= 0):
        raise ValueError("Need positive batch size, max_grad_norm, and nonnegative weight decay")
    phases = cfg.method.phases
    if not phases:
        raise ValueError("At least one training phase is required")
    for phase_name, phase in phases.items():
        if phase_name not in {"sft", "reinforce"}:
            raise ValueError(f"Unknown phase: {phase_name}")
        if not isinstance(phase.epochs, int) or phase.epochs < 1 or phase.lr <= 0:
            raise ValueError("Each phase needs positive integer epochs and positive lr")
    if "reinforce" in phases:
        if not 0 <= cfg.method.baseline_decay < 1:
            raise ValueError("baseline_decay must be in [0, 1)")
        if cfg.method.num_rollouts != 1:
            raise ValueError("Only num_rollouts=1 is supported for now")
        for coefficient in (cfg.method.entropy_coef, cfg.method.kl_coef):
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError("Entropy and KL coefficients must be finite and nonnegative")
        if cfg.method.kl_coef > 0 and list(phases).index("reinforce") == 0:
            raise ValueError("KL regularization requires an SFT phase before REINFORCE")

    torch.manual_seed(cfg.seed)
    device_name = cfg.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    task = make_task(cfg.task)
    reward_fn = make_reward(cfg.method.reward, task) if "reinforce" in phases else None

    train, test = split_dataset(
        task.dataset(), cfg.data.train_fraction, cfg.data.split_seed, cfg.data.test_fraction,
    )
    train_loader = DataLoader(
        train, batch_size=training.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    test_loader = DataLoader(test, batch_size=training.batch_size)

    model_config = dict(
        max_seq_len=task.input_length + task.output_length - 1,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
    )
    model = DigitTransformer(**model_config).to(device)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["output_dir"] = str(output_dir)
    config["resolved_device"] = str(device)
    config["dataset_sizes"] = {"train": len(train), "test": len(test)}
    with (output_dir / "config.json").open("x") as file:
        json.dump(config, file, indent=2)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(train):,} test={len(test):,} output={output_dir}", flush=True)

    optimizer_steps = 0
    examples_seen = 0
    epoch = 0
    training_seconds = 0.0
    with (output_dir / "metrics.jsonl").open("w") as log:
        for phase_index, (phase_name, phase) in enumerate(phases.items(), start=1):
            phase_reward_fn = reward_fn if phase_name == "reinforce" else None
            if phase_name == "sft":
                training_step = sft_step
            else:
                # freeze current model - assumption is that no one will use kl_coef > 0 if only doing REINFORCE
                reference_model = deepcopy(model) if cfg.method.kl_coef > 0 else None
                training_step = Reinforce(
                    reward_fn, cfg.method.baseline_decay, cfg.method.num_rollouts,
                    entropy_coef=cfg.method.entropy_coef,
                    kl_coef=cfg.method.kl_coef,
                    reference_model=reference_model,
                )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=phase.lr, weight_decay=training.weight_decay,
            )
            for phase_epoch in range(1, phase.epochs + 1):
                epoch += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = perf_counter()
                model.train()
                loss_sum = 0.0
                n_examples = 0
                statistic_sums = {}
                for prompts, answers in train_loader:
                    prompts = prompts.to(device)  # [B, P]
                    answers = answers.to(device)  # [B, A]

                    optimizer.zero_grad()
                    loss, batch_statistics = training_step(model, prompts, answers)  # scalar + metrics
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
                    optimizer.step()

                    batch_size = prompts.size(0)
                    loss_sum += loss.item() * batch_size
                    n_examples += batch_size
                    optimizer_steps += 1
                    examples_seen += batch_size
                    for name, value in batch_statistics.items():
                        statistic_sums[name] = statistic_sums.get(name, 0.0) + value * batch_size

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                training_seconds += perf_counter() - started
                test_metrics = evaluate(model, test_loader, device, phase_reward_fn)
                metrics = {
                    "epoch": epoch,
                    "method": phase_name,
                    "phase_index": phase_index,
                    "phase_epoch": phase_epoch,
                    "lr": phase.lr,
                    "training_seconds": training_seconds,
                    "optimizer_steps": optimizer_steps,
                    "examples_seen": examples_seen,
                    "train_loss": loss_sum / n_examples,
                    **{name: value / n_examples for name, value in statistic_sums.items()},
                    **test_metrics,
                }
                line = json.dumps(metrics)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

            checkpoint = {
                "task_name": cfg.task.name,
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "task_config": asdict(task),
                "run_config": config,
                "metrics": metrics,
            }
            torch.save(checkpoint, output_dir / f"phase_{phase_index}_{phase_name}.pt")

    torch.save(checkpoint, output_dir / "model.pt")


if __name__ == "__main__":
    main()
