import torch

from sft import sft_loss


@torch.no_grad()
def evaluate(model, loader, device, reward_fn=None):
    model.eval()
    loss_sum = 0.0
    correct_digits = 0
    exact_matches = 0
    n_digits = 0
    n_examples = 0
    reward_sum = 0.0
    for prompts, answers in loader:
        prompts = prompts.to(device)  # [B, P]
        answers = answers.to(device)  # [B, A]

        loss = sft_loss(model, prompts, answers)  # Scalar
        loss_sum += loss.item() * answers.numel()

        predictions = model.generate(prompts, answers.size(1), do_sample=False)  # [B, A]
        if reward_fn is not None:
            reward_sum += reward_fn(predictions, answers).sum().item()
        correct = predictions == answers             # [B, A], boolean
        correct_digits += correct.sum().item()
        exact_matches += correct.all(dim=1).sum().item()  # [B, A] -> [B] -> scalar
        n_digits += answers.numel()
        n_examples += answers.size(0)
    metrics = {
        "test_loss": loss_sum / n_digits,
        "test_digit_accuracy": correct_digits / n_digits,
        "test_exact_match": exact_matches / n_examples,
    }
    if reward_fn is not None:
        metrics["test_reward"] = reward_sum / n_examples
    return metrics
