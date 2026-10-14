import torch
import torch.nn as nn
from tabular_encoder import TabularEncoder
from string_encoder import MeanCharEncoder


class StringFieldDecoder(nn.Module):
    """
    Dekodiert aus dem gemeinsamen Latentraum für ein Stringfeld
    Zeichen-Logits der Form [B, L, V].

    Einfache Architektur:
    latent -> hidden -> max_seq_len * vocab_size
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        max_seq_len: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_seq_len * vocab_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input:
        - z: [B, latent_dim]

        Output:
        - logits: [B, L, V]
        """
        out = self.net(z)
        out = out.view(z.shape[0], self.max_seq_len, self.vocab_size)
        return out


class MultiModalAutoEncoder(nn.Module):
    """
    Multimodaler Autoencoder mit:
    - tabellarischem Encoder
    - String-Encodern pro Feld
    - gemeinsamer Fusionsschicht
    - tabellarischem Decoder
    - String-Decodern

    Output ist jetzt ein Dictionary statt nur eines Tupels.
    Dadurch können in main_v2.py mehrere SSL-Losses verwendet werden.
    """
    def __init__(
        self,
        tab_input_dim: int,
        vocab_size: int,
        char_emb_dim: int,
        string_emb_dim: int,
        latent_dim: int,
        tab_hidden_dims: tuple[int, ...],
        fusion_hidden_dims: tuple[int, ...],
        dropout: float,
        num_string_fields: int,
        max_seq_len: int,
        tab_encoder_out_dim: int = 64,
        string_decoder_hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_string_fields = num_string_fields
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.tab_encoder_out_dim = tab_encoder_out_dim
        self.string_emb_dim = string_emb_dim
        self.latent_dim = latent_dim

        # Tabellarischer Encoder
        self.tab_encoder = TabularEncoder(
            input_dim=tab_input_dim,
            hidden_dims=tab_hidden_dims,
            out_dim=tab_encoder_out_dim,
            dropout=dropout,
        )

        # Gemeinsamer String-Encoder für alle Stringfelder
        self.string_encoder = MeanCharEncoder(
            vocab_size=vocab_size,
            emb_dim=char_emb_dim,
            out_dim=string_emb_dim,
        )

        # Fusion aller Modalitäten
        fusion_in = tab_encoder_out_dim + num_string_fields * string_emb_dim
        fusion_layers = []
        prev = fusion_in
        for h in fusion_hidden_dims:
            fusion_layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        fusion_layers += [nn.Linear(prev, latent_dim)]
        self.fusion = nn.Sequential(*fusion_layers)

        # Tabellarischer Decoder
        self.tab_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, tab_input_dim),
        )

        # Ein Decoder pro Stringfeld
        self.string_decoders = nn.ModuleList([
            StringFieldDecoder(
                latent_dim=latent_dim,
                hidden_dim=string_decoder_hidden_dim,
                max_seq_len=max_seq_len,
                vocab_size=vocab_size,
                dropout=dropout,
            )
            for _ in range(num_string_fields)
        ])

    def encode_strings(self, x_strings: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Kodiert alle Stringfelder einzeln.

        Input:
        - Liste aus Tensoren [B, L]

        Output:
        - Liste aus Tensoren [B, string_emb_dim]
        """
        if self.num_string_fields == 0:
            return []

        z_strs = [self.string_encoder(xs) for xs in x_strings]
        return z_strs

    def forward(self, x_tab: torch.Tensor, x_strings: list[torch.Tensor]):
        """
        Forward-Pass.

        Rückgabe:
        {
            "x_tab_hat": ...,
            "x_str_logits": [...],
            "z_shared": ...,
            "z_tab": ...,
            "z_strs": [...],
        }
        """
        # Tabellarische Repräsentation
        z_tab = self.tab_encoder(x_tab)  # [B, tab_encoder_out_dim]

        # String-Repräsentationen
        z_strs = self.encode_strings(x_strings)  # Liste von [B, string_emb_dim]

        # Fusion
        if len(z_strs) > 0:
            z_cat = torch.cat([z_tab] + z_strs, dim=1)
        else:
            z_cat = z_tab

        z_shared = self.fusion(z_cat)  # [B, latent_dim]

        # Rekonstruktion tabellarisch
        x_tab_hat = self.tab_decoder(z_shared)

        # Rekonstruktion Stringfelder
        x_str_logits = []
        for decoder in self.string_decoders:
            logits = decoder(z_shared)  # [B, L, V]
            x_str_logits.append(logits)

        return {
            "x_tab_hat": x_tab_hat,
            "x_str_logits": x_str_logits,
            "z_shared": z_shared,
            "z_tab": z_tab,
            "z_strs": z_strs,
        }