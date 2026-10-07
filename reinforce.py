import math
import torch

from sequence import answer_logits


# def reinforce_loss(model, prompts, samples, advantages):
#     logits = answer_logits(model, prompts, samples)       # [B, A, 10]
#     log_probs = logits.log_softmax(dim=-1)                # [B, A, 10]
#     return policy_gradient_loss(log_probs, samples, advantages)


def policy_gradient_loss(log_probs, samples, advantages):
    # log probes: [B, A, 10], we gather only the sampled index
    chosen_log_probs = log_probs.gather(
        dim=-1, index=samples.unsqueeze(-1),             # samples: [B, A] -> [B, A, 1]
    ).squeeze(-1)                                        # [B, A]
    sequence_log_probs = chosen_log_probs.sum(dim=1)      # [B]
    return -(advantages.detach() * sequence_log_probs).mean()  # Scalar


class Reinforce:
    def __init__(self, reward_fn, baseline_decay=0.9, num_rollouts=1, *,
                 entropy_coef=0.0, kl_coef=0.0, reference_model=None):
        if not 0 <= baseline_decay < 1:
            raise ValueError("baseline_decay must be in [0, 1)")
        if num_rollouts != 1:
            raise ValueError("Only num_rollouts=1 is supported for now")
        for coefficient in (entropy_coef, kl_coef):
            if not math.isfinite(coefficient) or coefficient < 0:
                raise ValueError("Entropy and KL coefficients must be finite and nonnegative")
        if kl_coef > 0 and reference_model is None:
            raise ValueError("KL regularization requires a frozen SFT reference model")
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.reference_model = reference_model
        if reference_model is not None:
            reference_model.requires_grad_(False)
            reference_model.eval()
        self.reward_fn = reward_fn
        self.baseline_decay = baseline_decay
        self.baseline = 0.0

    def __call__(self, model, prompts, answers):
        samples = model.generate(prompts, answers.size(1), do_sample=True)  # [B, A]
        with torch.no_grad():
            rewards = self.reward_fn(samples, answers)  # [B]
            baseline_used = self.baseline              # scalar, previous batches only
            advantages = rewards - baseline_used      # [B]
        logits = answer_logits(model, prompts, samples)  # [B, A, 10]
        log_probs = logits.log_softmax(dim=-1)           # [B, A, 10]
        probs = log_probs.exp()                          # [B, A, 10]
        policy_loss = policy_gradient_loss(log_probs, samples, advantages)
        entropy = -(probs * log_probs).sum(dim=-1)       # [B, A]
        kl = torch.zeros_like(entropy)                  # [B, A]
        if self.kl_coef > 0:
            with torch.no_grad():
                reference_logits = answer_logits(self.reference_model, prompts, samples)
                reference_log_probs = reference_logits.log_softmax(dim=-1)  # [B, A, 10]
            kl = (probs * (log_probs - reference_log_probs)).sum(dim=-1)     # [B, A]
        # sum answer positions, then average sequence, i za entropy i za kl
        loss = (policy_loss - self.entropy_coef * entropy.sum(dim=1).mean()
                + self.kl_coef * kl.sum(dim=1).mean())

        mean_reward = rewards.mean().item()
        # baseline update
        self.baseline = (
            self.baseline_decay * baseline_used
            + (1 - self.baseline_decay) * mean_reward
        )
        return loss, {
            "train_policy_loss": policy_loss.item(),
            "train_entropy": entropy.mean().item(),
            **({"train_kl": kl.mean().item()} if self.kl_coef > 0 else {}),
            "train_reward": mean_reward,
            "train_baseline": baseline_used,
            "train_advantage": advantages.mean().item(),
            "train_sample_exact_match": (samples == answers).all(dim=1).float().mean().item(),
        }
