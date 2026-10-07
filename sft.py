from torch.nn import functional as F

from sequence import answer_logits


def sft_loss(model, prompts, answers):
    """Mean negative log probability of gold answer digits, excluding prompts."""
    logits = answer_logits(model, prompts, answers)  # [B, A, 10]

    flat_logits = logits.reshape(-1, logits.size(-1))  # [B * A, 10]
    flat_targets = answers.reshape(-1)                # [B * A]

    return F.cross_entropy(flat_logits, flat_targets)  # Scalar mean over BxA tokens


def sft_step(model, prompts, answers):
    return sft_loss(model, prompts, answers), {}
