"""
SMILES Variational Autoencoder (VAE)

LSTM-based VAE for molecular generation from SMILES strings.

Architecture:
    Encoder:  Embedding → BiLSTM → (mu, log_var)
    Decoder:  [Embedding(t) ⊕ z] → LSTM → Dropout → character probabilities
              z is concatenated at EVERY timestep (not just h0)
    Loss:     Reconstruction (CrossEntropy) + KL Divergence

Fixes applied (posterior collapse prevention):
    1. z concatenated at every decoder timestep (not just h0)
    2. Word dropout: randomly mask decoder input → forces z usage
    3. Proper autoregressive shift: predict next token, not current

Enhancements from literature review:
    4. Temperature-controlled sampling (All SMILES VAE, ICLR 2020)
    5. Decoder output dropout 0.2 (MOSES benchmark, Polykovskiy 2020)
    6. SMILES randomization in dataset (data augmentation)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.utils.constants import DEFAULT_LATENT_DIM, DEFAULT_EMBEDDING_DIM, DEFAULT_HIDDEN_DIM


class SmilesEncoder(nn.Module):
    """
    LSTM encoder: SMILES token sequence → latent space (mu, log_var).
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,  # PAD token index
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Bidirectional → hidden_dim * 2
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, input_tokens: torch.Tensor):
        """
        Args:
            input_tokens: (batch_size, sequence_length) integer token indices.

        Returns:
            mu:      (batch_size, latent_dim)
            log_var: (batch_size, latent_dim)
        """
        embedded = self.embedding(input_tokens)       # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)           # hidden: (num_layers*2, batch, hidden_dim)

        # Concatenate forward and backward final hidden states
        hidden_forward = hidden[-2]   # (batch, hidden_dim)
        hidden_backward = hidden[-1]  # (batch, hidden_dim)
        hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, hidden_dim*2)

        mu = self.fc_mu(hidden_combined)
        log_var = self.fc_log_var(hidden_combined)

        return mu, log_var


class SmilesDecoder(nn.Module):
    """
    LSTM decoder: latent vector z → reconstructed SMILES token sequence.

    Key design choices (posterior collapse prevention):
        - z is concatenated with token embeddings at EVERY timestep
          (not just used to initialize h0). This forces the decoder
          to condition on z throughout generation.
        - Word dropout: randomly replace input tokens with PAD (0)
          during training, forcing the decoder to rely on z for
          reconstruction instead of just copying the previous token.
        - Proper autoregressive shift: decoder input is tokens[:-1],
          predicting tokens[1:] (next-token prediction).
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        num_layers: int = 2,
        max_length: int = 122,  # MAX_MOLECULE_LENGTH + 2 (START + END)
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Project latent z to initial decoder hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.latent_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

        # LSTM input = token embedding + latent z concatenated at every step
        # This is the KEY fix: z conditions the decoder at every timestep
        self.lstm = nn.LSTM(
            input_size=embedding_dim + latent_dim,  # embedding ⊕ z
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output dropout (0.1): prevents decoder from memorizing sequences
        # Reduced from 0.2 → 0.1: MOSES uses 0.2 on a 3-layer decoder;
        # our 2-layer decoder needs less regularization to maintain validity.
        self.output_dropout = nn.Dropout(p=0.1)
        self.output_projection = nn.Linear(hidden_dim, vocabulary_size)

    def _init_hidden(self, latent_z: torch.Tensor):
        """Initialize LSTM hidden state from latent z."""
        batch_size = latent_z.size(0)

        hidden = self.latent_to_hidden(latent_z)  # (batch, hidden_dim * num_layers)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_dim)

        cell = self.latent_to_cell(latent_z)
        cell = cell.view(batch_size, self.num_layers, self.hidden_dim)
        cell = cell.permute(1, 0, 2).contiguous()

        return hidden, cell

    def forward(
        self,
        latent_z: torch.Tensor,
        target_tokens: torch.Tensor,
        word_dropout_rate: float = 0.0,
    ):
        """
        Teacher-forcing forward pass with autoregressive shift.

        The decoder predicts the NEXT token at each position:
            Input:  [START, tok1, tok2, ..., tokN]   (target[:, :-1])
            Output: [tok1,  tok2, tok3, ..., tokN+1] (predicts target[:, 1:])

        Args:
            latent_z:          (batch_size, latent_dim)
            target_tokens:     (batch_size, sequence_length) full token sequence
            word_dropout_rate: Fraction of input tokens to replace with PAD (0)
                               during training. Forces decoder to rely on z.

        Returns:
            logits: (batch_size, sequence_length - 1, vocabulary_size)
                    One fewer timestep than input (autoregressive shift)
        """
        # Autoregressive shift: input is all-but-last, predict all-but-first
        decoder_input = target_tokens[:, :-1]  # (batch, L-1)

        # Word dropout: randomly replace input tokens with PAD (0)
        # This forces the decoder to use z instead of just copying tokens
        if self.training and word_dropout_rate > 0:
            dropout_mask = torch.bernoulli(
                torch.full_like(decoder_input, 1.0 - word_dropout_rate, dtype=torch.float)
            ).long()
            # Keep START token (position 0) always visible
            dropout_mask[:, 0] = 1
            decoder_input = decoder_input * dropout_mask

        # Initialize hidden state from z
        hidden, cell = self._init_hidden(latent_z)

        # Embed tokens
        embedded = self.embedding(decoder_input)  # (batch, L-1, embed_dim)

        # Concatenate z at EVERY timestep — this is the key fix
        # Without this, the LSTM forgets z after a few steps
        seq_len = embedded.size(1)
        z_expanded = latent_z.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, L-1, latent_dim)
        lstm_input = torch.cat([embedded, z_expanded], dim=-1)  # (batch, L-1, embed_dim + latent_dim)

        # Decode
        lstm_output, _ = self.lstm(lstm_input, (hidden, cell))  # (batch, L-1, hidden_dim)
        lstm_output = self.output_dropout(lstm_output)  # Regularization
        logits = self.output_projection(lstm_output)  # (batch, L-1, vocab_size)

        return logits

    def generate(
        self,
        latent_z: torch.Tensor,
        start_token_index: int,
        max_length: Optional[int] = None,
        temperature: float = 0.8,
    ):
        """
        Autoregressive generation (no teacher forcing).

        Temperature controls the diversity-validity tradeoff:
            temperature < 1.0: sharper predictions → more valid, less diverse
            temperature = 1.0: standard sampling
            temperature > 1.0: softer predictions → more diverse, less valid
        (All SMILES VAE, ICLR 2020)

        Args:
            latent_z:          (batch_size, latent_dim)
            start_token_index: Index of the <START> token.
            max_length:        Maximum sequence length to generate.
            temperature:       Softmax temperature (default: 1.0).

        Returns:
            generated_tokens: (batch_size, max_length) integer indices.
        """
        if max_length is None:
            max_length = self.max_length

        batch_size = latent_z.size(0)
        device = latent_z.device

        # Initialize hidden state from z
        hidden, cell = self._init_hidden(latent_z)

        # Start with <START> token
        current_token = torch.full(
            (batch_size, 1), start_token_index, dtype=torch.long, device=device
        )

        generated_tokens = []

        for _ in range(max_length):
            embedded = self.embedding(current_token)  # (batch, 1, embed_dim)

            # Concatenate z at this timestep too — same as training
            z_step = latent_z.unsqueeze(1)  # (batch, 1, latent_dim)
            lstm_input = torch.cat([embedded, z_step], dim=-1)  # (batch, 1, embed_dim + latent_dim)

            lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            logits = self.output_projection(lstm_output.squeeze(1))  # (batch, vocab_size)

            # Temperature-scaled sampling
            # Lower temp = sharper (more valid), higher = softer (more diverse)
            scaled_logits = logits / max(temperature, 1e-8)
            probabilities = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # (batch, 1)

            generated_tokens.append(next_token)
            current_token = next_token

        return torch.cat(generated_tokens, dim=1)  # (batch, max_length)


class SmilesVAE(nn.Module):
    """
    Complete SMILES Variational Autoencoder.

    Combines encoder and decoder with reparameterization trick.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        num_layers: int = 2,
        max_length: int = 122,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = SmilesEncoder(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = SmilesDecoder(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            max_length=max_length,
            dropout=dropout,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon.

        Allows backpropagation through the sampling step.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            return mu + std * epsilon
        else:
            return mu  # deterministic at eval time

    def forward(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        word_dropout_rate: float = 0.0,
    ):
        """
        Full VAE forward pass.

        Args:
            input_tokens:      (batch_size, seq_len) — encoder input.
            target_tokens:     (batch_size, seq_len) — decoder target (teacher forcing).
            word_dropout_rate: Fraction of decoder input tokens to mask (forces z usage).

        Returns:
            reconstructed_logits: (batch_size, seq_len - 1, vocab_size)
            mu:                   (batch_size, latent_dim)
            log_var:              (batch_size, latent_dim)
        """
        mu, log_var = self.encoder(input_tokens)
        latent_z = self.reparameterize(mu, log_var)
        reconstructed_logits = self.decoder(
            latent_z, target_tokens, word_dropout_rate=word_dropout_rate
        )

        return reconstructed_logits, mu, log_var

    def generate_molecules(
        self,
        num_molecules: int,
        start_token_index: int,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate novel molecules by sampling from the latent space.

        Args:
            num_molecules:     Number of molecules to generate.
            start_token_index: Index of the <START> token.
            device:            Target device (cuda/cpu).
            temperature:       Softmax temperature for diversity control.

        Returns:
            generated_tokens: (num_molecules, max_length) integer indices.
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from standard normal prior
        latent_z = torch.randn(num_molecules, self.latent_dim, device=device)

        with torch.no_grad():
            generated_tokens = self.decoder.generate(
                latent_z, start_token_index, temperature=temperature
            )

        return generated_tokens


def compute_vae_loss(
    reconstructed_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 0.1,
    pad_index: int = 0,
) -> dict:
    """
    Compute VAE loss = Reconstruction Loss + KL Divergence.

    Handles autoregressive shift: if logits has fewer timesteps than
    target_tokens, the target is automatically shifted (target[:, 1:]).

    Args:
        reconstructed_logits: (batch, seq_len-1, vocab_size)
        target_tokens:        (batch, seq_len) — full sequence, auto-shifted
        mu:                   (batch, latent_dim)
        log_var:              (batch, latent_dim)
        kl_weight:            Weight for KL divergence term (beta-VAE).
        pad_index:            Index of PAD token (ignored in loss).

    Returns:
        Dictionary with total_loss, reconstruction_loss, and kl_loss.
    """
    # Auto-shift target to match decoder output length
    # Decoder outputs L-1 steps (predicting next token), target has L steps
    if reconstructed_logits.size(1) < target_tokens.size(1):
        target_tokens = target_tokens[:, 1:].contiguous()

    # Reconstruction loss (cross-entropy, ignoring PAD tokens)
    batch_size, seq_len, vocab_size = reconstructed_logits.shape
    reconstruction_loss = F.cross_entropy(
        reconstructed_logits.reshape(-1, vocab_size),
        target_tokens.reshape(-1),
        ignore_index=pad_index,
        reduction="mean",
    )

    # KL divergence: D_KL(q(z|x) || p(z))
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = reconstruction_loss + kl_weight * kl_loss

    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }
