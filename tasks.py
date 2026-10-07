from dataclasses import dataclass
import random

import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class ModularAddition:
    modulus: int = 113
    digits: int = 3
    num_examples: int | None = None
    dataset_seed: int = 0

    def __post_init__(self):
        if self.digits < 1 or not 2 <= self.modulus <= 10 ** self.digits:
            raise ValueError("Need digits >= 1 and 2 <= modulus <= 10**digits")
        if self.num_examples is not None and not 2 <= self.num_examples <= self.modulus ** 2:
            raise ValueError("num_examples must be between 2 and modulus**2")

    @property
    def input_length(self) -> int:
        return 2 * self.digits

    @property
    def output_length(self) -> int:
        return self.digits

    def dataset(self) -> TensorDataset:
        if self.num_examples is None:
            pair_ids = range(self.modulus ** 2)
        else:
            pair_ids = random.Random(self.dataset_seed).sample(
                range(self.modulus ** 2), self.num_examples,
            )

        def encode(number):
            # For digits=3: 69 -> "069" -> [0, 6, 9].
            padded = str(number).zfill(self.digits)
            return [int(digit) for digit in padded]

        prompts = []
        answers = []

        for pair_id in pair_ids:
            a, b = divmod(pair_id, self.modulus)
            result = (a + b) % self.modulus
            prompts.append(encode(a) + encode(b))
            answers.append(encode(result))

        return TensorDataset(
            torch.tensor(prompts, dtype=torch.long),
            torch.tensor(answers, dtype=torch.long),
        )


def digit_strings(length, num_examples, dataset_seed):
    """unique, uniformly sampled inputs without building all 10**length strings."""
    if not 1 <= length <= 18:
        raise ValueError("length must be between 1 and 18")
    if not 2 <= num_examples <= 10 ** length:
        raise ValueError("num_examples must be between 2 and 10**length")
    numbers = random.Random(dataset_seed).sample(range(10 ** length), num_examples)
    return torch.tensor([
        [int(digit) for digit in str(number).zfill(length)] for number in numbers
    ], dtype=torch.long)  # [N, P]


@dataclass(frozen=True)
class Reversal:
    length: int = 6
    num_examples: int = 10_000
    dataset_seed: int = 0

    @property
    def input_length(self):
        return self.length

    @property
    def output_length(self):
        return self.length

    def dataset(self):
        prompts = digit_strings(self.length, self.num_examples, self.dataset_seed)
        return TensorDataset(prompts, prompts.flip(dims=[1]))


@dataclass(frozen=True)
class Repetition:
    length: int = 6
    repeats: int = 2
    num_examples: int = 10_000
    dataset_seed: int = 0

    def __post_init__(self):
        if self.repeats < 1:
            raise ValueError("repeats must be positive")

    @property
    def input_length(self):
        return self.length

    @property
    def output_length(self):
        return self.length * self.repeats

    def dataset(self):
        prompts = digit_strings(self.length, self.num_examples, self.dataset_seed)
        answers = prompts.repeat_interleave(self.repeats, dim=1)  # [N, P * repeats]
        return TensorDataset(prompts, answers)


def make_task(config):
    settings = dict(config)
    name = settings.pop("name")
    task_types = {
        "modular_addition": ModularAddition,
        "reversal": Reversal,
        "repetition": Repetition,
    }
    if name not in task_types:
        raise ValueError(f"Unknown task: {name}")
    return task_types[name](**settings)


def split_dataset(dataset: TensorDataset, train_fraction: float, seed: int, test_fraction: float = 0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if not 0 < train_fraction <= 1 - test_fraction:
        raise ValueError("train_fraction must be positive and <= 1 - test_fraction")
    n = len(dataset)
    n_train = int(n * train_fraction)
    n_pool = int(n * (1 - test_fraction))
    if n_train < 1 or n_pool == n:
        raise ValueError("Dataset/split must contain training and test examples")
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    def subset(indices):
        return TensorDataset(*(tensor[indices] for tensor in dataset.tensors))

    return subset(order[:n_train]), subset(order[n_pool:])
