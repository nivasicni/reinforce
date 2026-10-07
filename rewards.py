import torch

from tasks import ModularAddition

"""Per-sequence rewards from generated digits and dataset answer digits."""

def exact_match(predictions, targets):
    return (predictions == targets).all(dim=1).float()  # [B, A] -> [B]


def correct_digits(predictions, targets):
    return (predictions == targets).float().mean(dim=1)  # [B, A] -> [B]


def circular_distance(predictions, targets, modulus):
    if modulus < 2:
        raise ValueError("modulus must be >= 2")
    powers = 10 ** torch.arange(
        predictions.size(1) - 1, -1, -1, device=predictions.device,
    )  # [A]
    predicted_numbers = (predictions * powers).sum(dim=1)  # [B]
    target_numbers = (targets * powers).sum(dim=1)         # [B]
    valid = predicted_numbers < modulus                  # [B]
    difference = (predicted_numbers - target_numbers).abs()
    distance = torch.minimum(difference, modulus - difference)
    reward = 1.0 - distance.float() / (modulus // 2)
    return torch.where(valid, reward, 0.0)                # [B]


def make_reward(name, task):
    if name == "exact_match":
        return exact_match
    if name == "correct_digits":
        return correct_digits
    if name == "circular_distance":
        if not isinstance(task, ModularAddition):
            raise ValueError("circular_distance is only defined for modular_addition")
        return lambda predictions, targets: circular_distance(
            predictions, targets, task.modulus,
        )
    raise ValueError(f"Unknown reward: {name}")
