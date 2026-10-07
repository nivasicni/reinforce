import torch


def answer_logits(model, prompts, answers):

    sequence = torch.cat((prompts, answers), dim=1)  # [B, P + A]

    input_tokens = sequence[:, :-1]               # [B, P + A - 1]
    all_logits = model(input_tokens)              # [B, P + A - 1, 10]

    # the last prompt position predicts the first answer digit.
    first_answer_position = prompts.size(1) - 1
    return all_logits[:, first_answer_position:]  # [B, A, 10]


