import torch
import torch.nn as nn


class ChronosWrapper(nn.Module):
    """Thin nn.Module wrapping Chronos T5 for gradient-based (PGD) attacks.

    Bypasses non-differentiable tokenization by projecting raw univariate
    input directly into the T5 encoder's embedding space via a learned linear
    layer, then running the encoder forward pass.
    """

    def __init__(self, pipeline):
        super().__init__()
        self.t5_model = pipeline.model.model  # T5ForConditionalGeneration
        self.embed_dim = self.t5_model.config.d_model
        self.projection = nn.Linear(1, self.embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, num_features) — use only column 0 (Power)
        univariate = x[:, :, 0:1]  # (batch, seq_len, 1)
        embeds = self.projection(univariate)  # (batch, seq_len, d_model)
        encoder_out = self.t5_model.encoder(inputs_embeds=embeds)
        return encoder_out.last_hidden_state
