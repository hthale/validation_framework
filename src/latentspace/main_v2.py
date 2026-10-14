from pathlib import Path
import json
import random
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os
import webbrowser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from delivery_memory_tabular import run_delivery_memory
from record_identity import IDENTITY_COLUMNS, build_record_key, with_record_key

import umap
import hdbscan

from latentspace.config_latent import LatentV2Config
from latentspace.schema import normalize_known_types
from latentspace.feature_prep import prepare_tabular_matrix
from latentspace.string_encoder import CharTokenizer
from latentspace.multimodal_autoencoder import MultiModalAutoEncoder
from latentspace.dataset import MultiModalDataset

from latentspace.anomaly_scores import build_outlier_scores, compute_lof_raw, lof_metadata, rank_by_lof
from latentspace.result_exports import build_neighbor_preview, save_umap_png, save_umap_html


CFG = LatentV2Config()


def set_seed(seed: int) -> None:
    """
    Setzt Seeds für Reproduzierbarkeit.

    Parameter:
    ----------
    seed : int
        Startwert für alle Zufallszahlengeneratoren.
        Wird für Python, NumPy und PyTorch gesetzt.

    Rückgabe:
    ---------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_string_columns(df: pd.DataFrame, cfg, tokenizer: CharTokenizer):
    """
    Kodiert die Stringspalten in Zeichen-IDs.

    Parameter:
    ----------
    df : pd.DataFrame
        Eingabedaten.
    cfg : object
        Konfiguration. Erwartet mindestens:
        - cfg.string_columns
        - cfg.max_seq_len
    tokenizer : CharTokenizer
        Tokenizer für zeichenbasiertes Encoding.

    Rückgabe:
    ---------
    tuple[list[np.ndarray], list[str]]
        encoded :
            Liste von Arrays je Stringspalte, jeweils Form [N, L].
        used_cols :
            Tatsächlich verwendete Stringspalten, die im DataFrame vorhanden sind.
    """
    encoded = []
    used_cols = []
    for col in cfg.string_columns:
        if col in df.columns:
            arr = np.array(
                [tokenizer.encode(v, cfg.max_seq_len) for v in df[col].astype("string").fillna("").tolist()],
                dtype=np.int64,
            )
            encoded.append(arr)
            used_cols.append(col)
    return encoded, used_cols


def get_loss_weights(cfg) -> dict:
    """
    Liest optionale Loss-Gewichte aus der Config.
    Falls dort noch nichts definiert ist, werden Defaults verwendet.

    Parameter:
    ----------
    cfg : object
        Konfigurationsobjekt.

    Rückgabe:
    ---------
    dict
        Dictionary mit Gewichtungen für:
        - tab_recon
        - str_recon
        - contrastive
        - alignment
    """
    return {
        "tab_recon": getattr(cfg, "weight_tab_recon", 1.0),
        "str_recon": getattr(cfg, "weight_str_recon", 0.5),
        "contrastive": getattr(cfg, "weight_contrastive", 0.1),
        "alignment": getattr(cfg, "weight_alignment", 0.1),
    }


def get_ssl_hyperparams(cfg) -> dict:
    """
    Liest optionale SSL-Hyperparameter aus der Config.

    Parameter:
    ----------
    cfg : object
        Konfigurationsobjekt.

    Rückgabe:
    ---------
    dict
        Dictionary mit:
        - tab_mask_prob : Maskierungsrate für tabellarische Features
        - tab_jitter_std : Stärke des Jitters auf Tabellendaten
        - str_mask_prob : Maskierungsrate für Zeichen/Tokens
        - contrastive_temperature : Temperatur für NT-Xent-Loss
    """
    return {
        "tab_mask_prob": getattr(cfg, "tab_mask_prob", 0.15),
        "tab_jitter_std": getattr(cfg, "tab_jitter_std", 0.02),
        "str_mask_prob": getattr(cfg, "str_mask_prob", 0.10),
        "contrastive_temperature": getattr(cfg, "contrastive_temperature", 0.2),
    }


def augment_tabular_view(x_tab: torch.Tensor, mask_prob: float, jitter_std: float) -> torch.Tensor:
    """
    Erzeugt eine augmentierte View des tabellarischen Inputs.

    Idee:
    - zufälliges Maskieren von Features
    - leichtes Jittern numerischer Werte

    Parameter:
    ----------
    x_tab : torch.Tensor
        Tabellarischer Input der Form [B, D].
        B = Batchgröße, D = Anzahl Features.
    mask_prob : float
        Wahrscheinlichkeit, dass einzelne Werte maskiert werden.
    jitter_std : float
        Standardabweichung des addierten Gaußrauschens.

    Rückgabe:
    ---------
    torch.Tensor
        Augmentierter tabellarischer Input.
    """
    x = x_tab.clone()

    # Zufälliges Maskieren einzelner Tabelleneinträge
    mask = torch.rand_like(x) < mask_prob
    x[mask] = 0.0

    # Leichtes Rauschen zur Erzeugung einer zweiten View
    if jitter_std > 0:
        noise = torch.randn_like(x) * jitter_std
        x = x + noise

    return x


def augment_string_views(
    x_strings: list[torch.Tensor],
    mask_prob: float,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> list[torch.Tensor]:
    """
    Erzeugt augmentierte Views der Stringeingaben.

    Idee:
    - zufälliges Maskieren einzelner Zeichen auf <UNK>
    - Padding bleibt unangetastet

    Parameter:
    ----------
    x_strings : list[torch.Tensor]
        Liste tokenisierter Stringfelder.
        Jedes Tensor-Element hat typischerweise Form [B, L].
    mask_prob : float
        Wahrscheinlichkeit, ein gültiges Zeichen/Token zu maskieren.
    pad_idx : int, default=0
        Index des Padding-Tokens.
    unk_idx : int, default=1
        Index des Unknown-/Maskierungs-Tokens.

    Rückgabe:
    ---------
    list[torch.Tensor]
        Liste augmentierter String-Tensoren.
    """
    aug_list = []

    for xs in x_strings:
        x = xs.clone()
        valid_mask = x != pad_idx
        random_mask = (torch.rand_like(x.float()) < mask_prob) & valid_mask
        x[random_mask] = unk_idx
        aug_list.append(x)

    return aug_list


def parse_model_output(model_output) -> dict:
    """
    Vereinheitlicht die Ausgabe des Modells.

    Unterstützt:
    - altes Format: (x_tab_hat, z_shared)
    - erweitertes Dict-Format mit zusätzlichen Feldern
    - erweitertes Tuple-Format

    Parameter:
    ----------
    model_output : dict | tuple
        Direkter Rückgabewert des Modells.

    Rückgabe:
    ---------
    dict
        Standardisiertes Dictionary mit:
        - x_tab_hat
        - x_str_logits
        - z_shared
        - z_tab
        - z_strs

    Fehler:
    -------
    ValueError
        Falls das Ausgabeformat unbekannt ist.
    """
    if isinstance(model_output, dict):
        return {
            "x_tab_hat": model_output.get("x_tab_hat"),
            "x_str_logits": model_output.get("x_str_logits"),
            "z_shared": model_output.get("z_shared"),
            "z_tab": model_output.get("z_tab"),
            "z_strs": model_output.get("z_strs"),
        }

    if isinstance(model_output, tuple):
        if len(model_output) == 2:
            x_tab_hat, z_shared = model_output
            return {
                "x_tab_hat": x_tab_hat,
                "x_str_logits": None,
                "z_shared": z_shared,
                "z_tab": None,
                "z_strs": None,
            }
        if len(model_output) == 5:
            x_tab_hat, x_str_logits, z_shared, z_tab, z_strs = model_output
            return {
                "x_tab_hat": x_tab_hat,
                "x_str_logits": x_str_logits,
                "z_shared": z_shared,
                "z_tab": z_tab,
                "z_strs": z_strs,
            }

    raise ValueError("Unbekanntes Modell-Output-Format.")


def compute_string_reconstruction_loss(
    x_str_logits: list[torch.Tensor] | None,
    x_strings: list[torch.Tensor],
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    String-Rekonstruktionsloss über CrossEntropy auf Zeichenebene.

    Erwartet pro Stringfeld Logits der Form [B, L, V].

    Parameter:
    ----------
    x_str_logits : list[torch.Tensor] | None
        Modell-Logits für jedes Stringfeld.
        Jedes Tensor-Element hat Form [B, L, V].
    x_strings : list[torch.Tensor]
        Zielsequenzen der Form [B, L].
    pad_idx : int, default=0
        Padding-Index, der beim Loss ignoriert wird.

    Rückgabe:
    ---------
    torch.Tensor
        Gemittelter Rekonstruktionsloss über alle Stringfelder.
        Falls keine String-Logits vorhanden sind, wird 0 zurückgegeben.
    """
    if x_str_logits is None or len(x_str_logits) == 0:
        return torch.tensor(0.0, device=x_strings[0].device if x_strings else "cpu")

    losses = []
    for logits, target in zip(x_str_logits, x_strings):
        # Ein vollständig fehlendes Textfeld im Batch liefert keine Zielzeichen.
        # CrossEntropy mit ausschließlich ignorierten Tokens wäre NaN.
        if not torch.any(target != pad_idx):
            continue
        # logits: [B, L, V] -> [B*L, V]
        # target: [B, L]    -> [B*L]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=pad_idx,
        )
        losses.append(loss)

    if not losses:
        return x_str_logits[0].sum() * 0.0
    return torch.stack(losses).mean()


def compute_per_row_string_reconstruction_error(
    x_str_logits: list[torch.Tensor] | None,
    x_strings: list[torch.Tensor],
    pad_idx: int = 0,
) -> np.ndarray:
    """
    Berechnet einen String-Rekonstruktionsfehler pro Zeile.

    Ergebnis:
    - shape [N]

    Parameter:
    ----------
    x_str_logits : list[torch.Tensor] | None
        Vorhersage-Logits je Stringfeld, Form [B, L, V].
    x_strings : list[torch.Tensor]
        Zielsequenzen je Stringfeld, Form [B, L].
    pad_idx : int, default=0
        Padding-Index, der ignoriert wird.

    Rückgabe:
    ---------
    np.ndarray
        Mittlerer String-Rekonstruktionsfehler pro Datenzeile.
    """
    if x_str_logits is None or len(x_str_logits) == 0:
        return np.zeros(len(x_strings[0])) if x_strings else np.array([])

    per_field = []

    for logits, target in zip(x_str_logits, x_strings):
        # Log-Softmax über das Vokabular
        log_probs = F.log_softmax(logits, dim=-1)

        # Wahrscheinlichkeiten der Zielzeichen auswählen
        gathered = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        # Nur echte Tokens werten, Padding ignorieren
        valid_mask = (target != pad_idx).float()
        neg_log_likelihood = -gathered * valid_mask

        # Zeilenweise mitteln
        token_count = valid_mask.sum(dim=1).clamp(min=1.0)
        row_loss = neg_log_likelihood.sum(dim=1) / token_count
        per_field.append(row_loss.detach().cpu().numpy())

    return np.mean(np.vstack(per_field), axis=0)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Einfacher SimCLR-/NT-Xent-Loss auf zwei augmentierten Views.

    Parameter:
    ----------
    z1 : torch.Tensor
        Latente Repräsentationen der ersten View, Form [B, D].
    z2 : torch.Tensor
        Latente Repräsentationen der zweiten View, Form [B, D].
    temperature : float, default=0.2
        Temperaturparameter für die Similarity-Skalierung.

    Rückgabe:
    ---------
    torch.Tensor
        Contrastive Loss.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)

    # Ähnlichkeitsmatrix aller Embeddings
    sim = torch.matmul(z, z.T) / temperature

    # Selbstvergleiche ausmaskieren
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positives Pair: i in View1 gehört zu i in View2
    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    loss = F.cross_entropy(sim, positives)
    return loss


def compute_cross_modal_alignment_loss(
    z_tab: torch.Tensor | None,
    z_strs: list[torch.Tensor] | None,
) -> torch.Tensor:
    """
    Einfacher Alignment-Loss zwischen tabellarischer Repräsentation
    und aggregierter String-Repräsentation.

    Der Loss ist hier bewusst einfach:
    MSE zwischen z_tab und Mittelwert der String-Embeddings.

    Parameter:
    ----------
    z_tab : torch.Tensor | None
        Latente Repräsentation der tabellarischen Modalität, Form [B, D].
    z_strs : list[torch.Tensor] | None
        Liste latenter Repräsentationen der Stringmodalitäten.

    Rückgabe:
    ---------
    torch.Tensor
        Alignment-Loss.
        Falls keine Modalitäten verfügbar sind, wird 0 zurückgegeben.
    """
    if z_tab is None or z_strs is None or len(z_strs) == 0:
        device = z_tab.device if z_tab is not None else "cpu"
        return torch.tensor(0.0, device=device)

    z_str_mean = torch.stack(z_strs, dim=0).mean(dim=0)

    # Falls Dimensionen unterschiedlich sind, nur gemeinsame Dimension vergleichen
    dim = min(z_tab.shape[1], z_str_mean.shape[1])
    return F.mse_loss(z_tab[:, :dim], z_str_mean[:, :dim])


def compute_modality_attribution(
    tab_error: np.ndarray,
    string_error: np.ndarray,
) -> pd.DataFrame:
    """
    Einfache, erklärbare Modalitätsattribution auf Basis der Fehleranteile.

    Parameter:
    ----------
    tab_error : np.ndarray
        Fehler pro Zeile aus der tabellarischen Rekonstruktion.
    string_error : np.ndarray
        Fehler pro Zeile aus der String-Rekonstruktion.

    Rückgabe:
    ---------
    pd.DataFrame
        Enthält:
        - tabular_modality_share
        - string_modality_share
        - dominant_modality
    """
    tab_error = np.asarray(tab_error, dtype=float)
    string_error = np.asarray(string_error, dtype=float)

    tab_norm = tab_error.copy()
    str_norm = string_error.copy()

    # Normierung auf [0,1] durch Division durch den Maximalwert
    if np.max(tab_norm) > 0:
        tab_norm = tab_norm / np.max(tab_norm)
    if np.max(str_norm) > 0:
        str_norm = str_norm / np.max(str_norm)

    total = tab_norm + str_norm
    total[total == 0] = 1.0

    tab_share = tab_norm / total
    str_share = str_norm / total

    dominant = np.where(tab_share >= str_share, "tabular", "string")

    return pd.DataFrame({
        "tabular_modality_share": tab_share,
        "string_modality_share": str_share,
        "dominant_modality": dominant,
    })


def train_model(df: pd.DataFrame, cfg, *, prepared_tabular=None):
    """
    Trainiert das multimodale Modell mit:
    - tabular reconstruction loss
    - string reconstruction loss
    - contrastive loss auf zwei augmentierten Views
    - einfachem cross-modal alignment loss

    Parameter:
    ----------
    df : pd.DataFrame
        Rohdaten.
    cfg : object
        Konfigurationsobjekt mit Modell-, Trainings- und Datenparametern.

    Rückgabe:
    ---------
    dict
        Ergebnisobjekt mit:
        - df
        - X_tab
        - X_tab_hat
        - latent
        - reconstruction_error
        - string_reconstruction_error
        - used_string_cols
        - numeric_cols
        - categorical_cols
        - modality_attribution
        - training_history
        - model
        - preprocessor
        - tokenizer
    """
    set_seed(cfg.random_state)

    # Datentypen vor der Feature-Erzeugung vereinheitlichen
    # Identitäten dienen nur der Zuordnung, nicht dem Modelltraining.
    df = df.drop(columns=list(IDENTITY_COLUMNS), errors="ignore")
    df = normalize_known_types(df, cfg)

    # Tabellarische Matrix + Preprocessor erzeugen
    # Optionaler, bereits dokumentierter generischer Merkmalsvertrag.
    # Der bestehende DOP-Einstieg behält seine bisherige Aufbereitung.
    prep = prepare_tabular_matrix(df) if prepared_tabular is None else prepared_tabular
    matrix = np.asarray(prep["X_tab"])
    if matrix.ndim != 2 or matrix.shape[0] != len(df) or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Vorbereitete Merkmalsmatrix passt nicht zur Eingabe oder enthält ungültige Werte.")

    # Übergang von Vorbereitung auf Modelltraining
    """ X_tab
    → fertige numerische Matrix für das Modell
    preprocessor
    → Objekt, das genau weiß, wie die Transformation durchgeführt wurde
    num_cols
    → welche numerischen Spalten erkannt wurden
    cat_cols
    → welche kategorialen Spalten erkannt wurden """
    X_tab = prep["X_tab"]
    preprocessor = prep["preprocessor"]
    num_cols = prep["numeric_cols"]
    cat_cols = prep["categorical_cols"]

    # Zeichenbasierte Kodierung der Textspalten, weil X_tab nur tabellarische Informationen enthält, werden hier die echten Zeichenketten separat kodiert. 
    # Das ermöglicht es dem Modell, die textuelle Modalität zu lernen, ohne dass diese Informationen in der tabellarischen Matrix verloren gehen.
    # CharTokenizer zerlegt Strings zeichenweise und bildet sie auf Integer-IDs ab.
    tokenizer = CharTokenizer()
    X_str_list, used_string_cols = encode_string_columns(df, cfg, tokenizer)

    # Gemeinsames Dataset aus Tabellen- und Stringdaten, aber nicht alle Daten werden gleichzeitig in den Speicher geladen, sondern nur die aktuellen Batches. 
    # Das ermöglicht das Training auch bei größeren Datensätzen.
    dataset = MultiModalDataset(X_tab, X_str_list)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Modellaufbau
    # Tabellarischer Encoder mit X_tba und String_Encoder mit X_str_list
    model = MultiModalAutoEncoder(
        tab_input_dim=X_tab.shape[1],
        vocab_size=tokenizer.vocab_size,
        char_emb_dim=cfg.char_emb_dim,
        string_emb_dim=cfg.string_emb_dim,
        latent_dim=cfg.latent_dim,
        tab_hidden_dims=cfg.tab_hidden_dims,
        fusion_hidden_dims=cfg.fusion_hidden_dims,
        dropout=cfg.dropout,
        num_string_fields=len(used_string_cols),
        max_seq_len=cfg.max_seq_len,
    )

    # Meine Parameter aus config_latent
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    tab_loss_fn = nn.MSELoss()
    weights = get_loss_weights(cfg)
    ssl_params = get_ssl_hyperparams(cfg)

    model.train()
    history = []

    #Input
    #↓
    #Encoder
    #↓
    #z_tab
    #z_strs
    #z_shared
    #↓
    #Decoder
    #↓
    #x_tab_hat
    #x_str_logits
    #↓
    #Loss-Berechnung
    for epoch in range(cfg.epochs):
        running_total = 0.0
        running_tab = 0.0
        running_str = 0.0
        running_ctr = 0.0
        running_align = 0.0

        # x_tab, x_strings → Encoder → latente Repräsentationen
        # Das Ergebnis davon sind: z_tab, z_strs, z_shared → latente Vektoren
        # x_tab_hat, x_str_logits → Decoder → ursprüngliche Tabellenwerte und String-Logits
        for x_tab, x_strings in loader:
            optimizer.zero_grad()

            # Hauptdurchlauf auf Originaldaten für Rekonstruktion

            out_main = parse_model_output(model(x_tab, x_strings))
            x_tab_hat = out_main["x_tab_hat"]
            x_str_logits = out_main["x_str_logits"]
            z_main = out_main["z_shared"]
            z_tab = out_main["z_tab"]
            z_strs = out_main["z_strs"]
            
            # !!!Rekonstruktionsfehler der Tabellendaten
            # Wenn das schlecht klappt: möglicher Ausreißer in den Daten, der die Rekonstruktion erschwert.
            loss_tab = tab_loss_fn(x_tab_hat, x_tab)

            # !!!Rekonstruktionsfehler der Strings, Gleiche wie Tabellendaten
            loss_str = compute_string_reconstruction_loss(
                x_str_logits=x_str_logits,
                x_strings=x_strings,
                pad_idx=0,
            )

            # !!! Angleichen der tabellarischen und textuellen Repräsentationen. 
            # Der Loss zwingt die latenten Repräsentationen der beiden Modalitäten dazu, sich anzugleichen, was die gemeinsame Struktur im Latent Space fördert.
            # Wenn ein Objekt tabellarisch und textuell zusammengehört, sollen beide im gleichen Bereich des Latent Spaces landen.
            loss_align = compute_cross_modal_alignment_loss(
                z_tab=z_tab,
                z_strs=z_strs,
            )

            # !!! SSL Teil:Contrastive Learning: Erzeugung von zwei ähnliche Datensätze.
            # kleines Rauschen und/oder Maskierung, dann lernt das Modell: gleicher Datensatz → ähnliche Latent-Repräsentation
            x_tab_v1 = augment_tabular_view(
                x_tab,
                mask_prob=ssl_params["tab_mask_prob"],
                jitter_std=ssl_params["tab_jitter_std"],
            )
            x_tab_v2 = augment_tabular_view(
                x_tab,
                mask_prob=ssl_params["tab_mask_prob"],
                jitter_std=ssl_params["tab_jitter_std"],
            )

            x_str_v1 = augment_string_views(
                x_strings,
                mask_prob=ssl_params["str_mask_prob"],
            )
            x_str_v2 = augment_string_views(
                x_strings,
                mask_prob=ssl_params["str_mask_prob"],
            )

            # Separate Forward-Pässe für die beiden SSL-Views
            out_v1 = parse_model_output(model(x_tab_v1, x_str_v1))
            out_v2 = parse_model_output(model(x_tab_v2, x_str_v2))

            # Contrastive Loss auf dem Shared Latent Space
            loss_contrast = nt_xent_loss(
                out_v1["z_shared"],
                out_v2["z_shared"],
                temperature=ssl_params["contrastive_temperature"],
            )

            # Gesamtloss als gewichtete Summe
            loss = (
                weights["tab_recon"] * loss_tab
                + weights["str_recon"] * loss_str
                + weights["contrastive"] * loss_contrast
                + weights["alignment"] * loss_align
            )
            # wie die Gewichte verändert werden müssen, damit die einzelnen Loss-Komponenten in etwa den gleichen Wertebereich haben.
            # Also welche Modellparameter haben den Fehler verursacht?
            loss.backward()
            # Gewichtung aktualisieren, damit der Fehler kleiner wird.
            optimizer.step()

            # running_total ist die Summe aller Verluste über die Epoche, gewichtet mit der Batchgröße, damit am Ende der Durchschnitt über alle Datenpunkte berechnet werden kann.
            # Ich trainiere nicht auf allen Daten gleichzeitig, sondern in Batches. 
            # Daher muss ich die Verluste mit der Anzahl der Datenpunkte im Batch multiplizieren, um die Summe über die Epoche korrekt zu berechnen.
            batch_size = len(x_tab)
            running_total += loss.item() * batch_size
            running_tab += float(loss_tab.item()) * batch_size
            running_str += float(loss_str.item()) * batch_size
            running_ctr += float(loss_contrast.item()) * batch_size
            running_align += float(loss_align.item()) * batch_size

        # epoch_statsberechnet die Durchschnittswerte pro Epoche, dadurch sehe ich, welcher Teil des Trainings besser/schlechter wird
        epoch_stats = {
            "epoch": epoch + 1,
            "loss_total": running_total / len(dataset),
            "loss_tab_recon": running_tab / len(dataset),
            "loss_str_recon": running_str / len(dataset),
            "loss_contrastive": running_ctr / len(dataset),
            "loss_alignment": running_align / len(dataset),
        }
        # Speichern der Trainingshistorie, damit ich später sehen kann, wie sich die Verluste über die Zeit entwickelt haben.
        history.append(epoch_stats)

        print(
            f"[V2] Epoch {epoch+1:02d}/{cfg.epochs} "
            f"total={epoch_stats['loss_total']:.6f} "
            f"tab={epoch_stats['loss_tab_recon']:.6f} "
            f"str={epoch_stats['loss_str_recon']:.6f} "
            f"ctr={epoch_stats['loss_contrastive']:.6f} "
            f"align={epoch_stats['loss_alignment']:.6f}"
        )

    # Trainingsmodus AUS, Evaluationsmodus EIN (Wichtig für: Dropout, BatchNorm, stabile Vorhersagen)
    # Evaluation auf allen Daten ohne Gradienten
    model.eval()
    #deaktiviert Gradientberechnung, da wir hier nur Vorhersagen machen wollen und keine Gewichte aktualisieren. Das spart Speicher und Rechenzeit.
    with torch.no_grad():
        x_tab_t = torch.tensor(X_tab, dtype=torch.float32)
        x_str_t = [torch.tensor(x, dtype=torch.long) for x in X_str_list]
        out_eval = parse_model_output(model(x_tab_t, x_str_t))
    # Modell läuft einmal über ALLE Daten und prodziert die Vorhersagen für die tabellarischen Daten (x_tab_hat), die String-Logits (x_str_logits) und die latenten Repräsentationen (z_shared).
    # x_tab_hat ist die rekonstruierte Tabelle.
    x_tab_hat = out_eval["x_tab_hat"]
    x_str_logits = out_eval["x_str_logits"]
    # z_shared ist der gemeinsame Latent Space.
    z_shared = out_eval["z_shared"]

    X_tab_hat = x_tab_hat.detach().cpu().numpy()
    Z = z_shared.detach().cpu().numpy()

    # Zeilenweiser Rekonstruktionsfehler für Tabellendaten (Großer Fehler: schlecht rekonstruiert, kleiner Fehler: gut rekonstruiert)
    tab_recon_error = ((X_tab_hat - X_tab) ** 2).mean(axis=1)

    # Zeilenweiser Rekonstruktionsfehler für Strings  (Großer Fehler: schlecht rekonstruiert, kleiner Fehler: gut rekonstruiert)
    string_recon_error = compute_per_row_string_reconstruction_error(
        x_str_logits=x_str_logits,
        x_strings=x_str_t,
        pad_idx=0,
    )

    # Diagnose der normierten Rekonstruktionsfehler; keine LOF-Ursachenerklärung.
    # (Großer Wert in tabular_modality_share: Fehler hauptsächlich durch tabellarische Modalität, 
    # großer Wert in string_modality_share: Fehler hauptsächlich durch Stringmodalität)
    modality_attr = compute_modality_attribution(
        tab_error=tab_recon_error,
        string_error=string_recon_error if len(string_recon_error) else np.zeros(len(tab_recon_error)),
    )

    """ Rückgabe	                Bedeutung
    latent                  	    gemeinsamer Latent Space
    reconstruction_error	        Tabellenfehler
    string_reconstruction_error	    Stringfehler
    modality_attribution	        Rekonstruktionsdiagnostik
    model	                        trainiertes Modell
    preprocessor	                Vorverarbeitung
    tokenizer	                    String-Tokenizer
    training_history	            Trainingsverlauf """
    return {
        "df": df,
        "X_tab": X_tab,
        "X_tab_hat": X_tab_hat,
        "latent": Z,
        "reconstruction_error": tab_recon_error,
        "string_reconstruction_error": string_recon_error if len(string_recon_error) else np.zeros(len(tab_recon_error)),
        "used_string_cols": used_string_cols,
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "modality_attribution": modality_attr,
        "training_history": history,
        "model": model,
        "preprocessor": preprocessor,
        "tokenizer": tokenizer,
    }


def compute_hdbscan_raw(Z: np.ndarray, min_cluster_size: int, min_samples: int):
    """
    HDBSCAN-Cluster und Outlier-Scores im 64D-Raum.

    Parameter:
    ----------
    Z : np.ndarray
        Latenter Raum der Form [N, D].
    min_cluster_size : int
        Mindestgröße eines Clusters.
    min_samples : int
        Stärke der Dichteanforderung; höher = konservativer.

    Rückgabe:
    ---------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        labels :
            Cluster-Labels, -1 bedeutet Noise.
        outlier_scores :
            Outlier-Scores von HDBSCAN.
        probabilities :
            Zugehörigkeitswahrscheinlichkeiten zu Clustern.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(Z)
    outlier_scores = getattr(clusterer, "outlier_scores_", np.zeros(len(Z), dtype=float))
    outlier_scores = np.nan_to_num(outlier_scores, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = getattr(clusterer, "probabilities_", np.ones(len(Z), dtype=float))
    probabilities = np.nan_to_num(probabilities, nan=0.0)
    return labels, outlier_scores, probabilities


def compute_centroid_distance_raw(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Distanz zum Cluster-Zentroid als zusätzlicher Outlier-Score.

    Parameter:
    ----------
    Z : np.ndarray
        Latenter Raum der Form [N, D].
    labels : np.ndarray
        Clusterlabels pro Punkt.

    Rückgabe:
    ---------
    np.ndarray
        Distanz zum jeweiligen Clusterzentrum.
        Noise-Punkte erhalten einen hohen Fallback-Wert.
    """
    scores = np.zeros(len(Z), dtype=float)
    valid_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    for lbl in valid_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < 3:
            continue
        cluster_points = Z[idx]
        centroid = cluster_points.mean(axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        scores[idx] = dists

    # Noise-Punkte mit hohem Distanzwert markieren
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) > 0:
        non_noise = scores[labels != -1]
        fallback = np.percentile(non_noise, 95) if len(non_noise) > 0 else 1.0
        scores[noise_idx] = np.maximum(scores[noise_idx], fallback)

    return scores


def project_umap(Z: np.ndarray, cfg) -> np.ndarray:
    """
    UMAP-Projektion nur für Visualisierung und Debug.

    Parameter:
    ----------
    Z : np.ndarray
        Latenter Raum der Form [N, D].
    cfg : object
        Konfiguration mit:
        - umap_n_neighbors
        - umap_min_dist
        - random_state

    Rückgabe:
    ---------
    np.ndarray
        2D-Projektion der Form [N, 2].
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.random_state,
        init="random",
    )
    return reducer.fit_transform(Z)


def main():
    """
    Gesamtablauf:
    ├── lädt Parquet
    ├── entfernt geometry / __index_level_0__
    ├── aktualisiert Delivery-Memory
    ├── ruft train_model(df, CFG) auf
    │   ├── dort: Schema/Typen normalisieren
    │   ├── dort: numerische/kategoriale Matrix bauen
    │   ├── dort: String-Spalten codieren
    │   └── dort: Autoencoder trainieren
    ├── berechnet LOF / HDBSCAN / UMAP
    └── speichert Ergebnisse

    Verwendete Konfigurationswerte aus CFG:
    ---------------------------------------
    - parquet_path : Pfad zu den Eingabedaten
    - output_root_dir : Basisordner für Ergebnisse
    - lof_n_neighbors : LOF-Parameter
    - hdbscan_min_cluster_size : HDBSCAN-Parameter
    - hdbscan_min_samples : HDBSCAN-Parameter
    - latent_dim : Dimension des Latent Space
    - umap_n_neighbors : UMAP-Parameter
    - umap_min_dist : UMAP-Parameter

    Rückgabe:
    ---------
    None
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Eingabedaten laden
    # Positionen für numpy-/Torch-Ausgaben vereinheitlichen; Identität separat erhalten.
    df = with_record_key(pd.read_parquet(CFG.parquet_path).reset_index(drop=True))
    df = df.drop(columns=["geometry", "__index_level_0__"], errors="ignore")

    if getattr(CFG, "enable_delivery_memory", False):

        run_delivery_memory(
            df=df,
            input_parquet=Path(CFG.parquet_path),
            #welcher Lieferfamilie gehört der Datensatz
            delivery_group_id=CFG.delivery_group_id,
            # merken von früheren Lieferungen.
            memory_dir=Path(CFG.delivery_memory_dir),
            #konkrete Lieferordner
            delivery_root_dir=Path(CFG.delivery_root_dir) if CFG.delivery_root_dir else None,
        )
    # Modell trainieren und zentrale Artefakte erzeugen
    result = train_model(df, CFG)
    Z = result["latent"]

    # Outlier-Metriken direkt im Latent Space
    lof_raw = compute_lof_raw(Z, CFG.lof_n_neighbors, CFG.random_state)
    labels, hdbscan_raw, probabilities = compute_hdbscan_raw(
        Z,
        CFG.hdbscan_min_cluster_size,
        CFG.hdbscan_min_samples,
    )
    centroid_raw = compute_centroid_distance_raw(Z, labels)

    # LOF als Hauptscore; andere Kennzahlen bleiben separate Diagnostik
    score_df = build_outlier_scores(
        recon_error=result["reconstruction_error"],
        string_recon_error=result["string_reconstruction_error"],
        lof_raw=lof_raw,
        hdbscan_raw=hdbscan_raw,
        centroid_raw=centroid_raw,
    )

    # 2D-Projektion nur für Visualisierung
    Z_2d = project_umap(Z, CFG)

    # Laufordner mit Zeitstempel anlegen
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(CFG.output_root_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    # Basisergebnisse je Zeile
    results = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "record_key": df["record_key"].to_numpy(),
        "cluster_label": labels,
        "cluster_probability": probabilities,
        "centroid_distance_is_fallback": labels == -1,
        "umap_x": Z_2d[:, 0],
        "umap_y": Z_2d[:, 1],
    })

    # Score-Informationen und Rekonstruktionsdiagnostik ergänzen
    results = pd.concat([results, score_df], axis=1)
    results = pd.concat([results, result["modality_attribution"]], axis=1)

    # Alle Latent-Dimensionen explizit speichern
    for i in range(Z.shape[1]):
        results[f"latent_{i:02d}"] = Z[:, i]

    # Lesbare Vorschau-Spalten aus den Originaldaten dazunehmen
    preview_cols = [
        c for c in ["kachelname", "xml", "tif", "aktualitaet", "bildflugname", "kamera_sensor"]
        if c in df.columns
    ]
    # Alle Originalattribute für Hover/Analyse anhängen
    original_cols = [
        c for c in df.columns
        if c not in results.columns
    ]

    results = results.join(df[original_cols], how="left")

    # Höchste Outlier-Scores nach oben sortieren
    results = rank_by_lof(results)

    # Vorschau der Nachbarn für dieselben LOF-Spitzenplätze wie in CSV und Plot.
    neighbor_preview = build_neighbor_preview(results, top_k=100, k=5)

    # Gesamtergebnis speichern
        #Also die ursprünglichen Daten plus:
        
        #Latent-Space-Informationen
        #Outlier-Scores
        #Clusterinformationen
        #Rekonstruktionsfehler
        #Rekonstruktionsdiagnostik
    results.to_parquet(out_dir / "latent_outlier_results_lof.parquet", index=False)

    # Nur Top-Ausreißer als CSV speichern
    results.head(250).to_csv(out_dir / "top_outliers_lof.csv", index=False, encoding="utf-8")

    # UMAP-Punkte separat speichern
    results[[
        "record_key",
        "row_id",
        "outlier_rank",
        "umap_x",
        "umap_y",
        "lof_raw",
        "cluster_label",
        "dominant_modality",
    ]].to_csv(
        out_dir / "umap_points_lof.csv",
        index=False,
        encoding="utf-8",
    )

    # Visualisierungen speichern
    save_umap_png(results, out_dir / "umap_plot_lof.png")
    umap_html_created = save_umap_html(results, out_dir / "umap_plot_lof.html")

    # Modell und Vorverarbeitung speichern
    torch.save(result["model"].state_dict(), out_dir / "multimodal_autoencoder_v2.pt")
    joblib.dump(result["preprocessor"], out_dir / "tabular_preprocessor_v2.pkl")
    joblib.dump(result["tokenizer"], out_dir / "char_tokenizer_v2.pkl")

    # Nachbarinformationen als JSON speichern
    with open(out_dir / "top_outlier_neighbors_lof.json", "w", encoding="utf-8") as f:
        json.dump(neighbor_preview, f, ensure_ascii=False, indent=2, default=str)

    # Trainingsverlauf speichern
    with open(out_dir / "training_history_v2.json", "w", encoding="utf-8") as f:
        json.dump(result["training_history"], f, ensure_ascii=False, indent=2, default=str)

    # Metadaten des Runs speichern
    with open(out_dir / "run_info_lof.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **lof_metadata(len(results), CFG.lof_n_neighbors, CFG.random_state),
                "source_input": str(Path(CFG.parquet_path).resolve()),
                "run_id": run_id,
                "training_performed": True,
                "umap_html_created": umap_html_created,
                "used_string_cols": result["used_string_cols"],
                "numeric_cols": result["numeric_cols"],
                "categorical_cols": result["categorical_cols"],
                "latent_dim": CFG.latent_dim,
                "top10_cluster_labels": results.head(10)["cluster_label"].tolist(),
                "neighbor_k": neighbor_preview["neighbor_k"],
                "has_modality_attribution": True,
                "loss_weights": get_loss_weights(CFG),
                "ssl_hyperparams": get_ssl_hyperparams(CFG),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # Konsolenausgabe für schnellen Überblick
       # ---------------------------------------------------------
    # Konsolen-Ausgabe: wichtige Pfade
    # ---------------------------------------------------------
    result_parquet = out_dir / "latent_outlier_results_lof.parquet"
    top_csv = out_dir / "top_outliers_lof.csv"
    umap_csv = out_dir / "umap_points_lof.csv"
    umap_png = out_dir / "umap_plot_lof.png"
    umap_html = out_dir / "umap_plot_lof.html"
    neighbors_json = out_dir / "top_outlier_neighbors_lof.json"
    history_json = out_dir / "training_history_v2.json"
    run_info_json = out_dir / "run_info_lof.json"
    model_file = out_dir / "multimodal_autoencoder_v2.pt"
    preprocessor_file = out_dir / "tabular_preprocessor_v2.pkl"
    tokenizer_file = out_dir / "char_tokenizer_v2.pkl"

    print("\n" + "=" * 80)
    print("FERTIG - V2 LATENT SPACE / LOF-RANGLISTE")
    print("=" * 80)

    print("\nRun-Ordner:")
    print(out_dir.resolve())

    print("\nWichtige Ergebnisdateien:")
    print(f"Ergebnisse Parquet: {result_parquet.resolve()}")
    print(f"Top-Outlier CSV:    {top_csv.resolve()}")
    print(f"UMAP CSV:           {umap_csv.resolve()}")
    print(f"UMAP PNG:           {umap_png.resolve()}")
    if umap_html_created:
        print(f"UMAP HTML:          {umap_html.resolve()}")
    print(f"Nachbarn JSON:      {neighbors_json.resolve()}")
    print(f"Training History:   {history_json.resolve()}")
    print(f"Run Info:           {run_info_json.resolve()}")

    print("\nGespeicherte Modellartefakte:")
    print(f"Modell:             {model_file.resolve()}")
    print(f"Preprocessor:       {preprocessor_file.resolve()}")
    print(f"Tokenizer:          {tokenizer_file.resolve()}")

    print("\nFür eine spätere Fehlerinjektionsauswertung ist dies der --results-Pfad:")
    print(result_parquet.resolve())

    print("=" * 80)

    # ---------------------------------------------------------
    # Optional: Explorer und UMAP HTML automatisch öffnen
    # ---------------------------------------------------------
    try:
        os.startfile(out_dir)
    except Exception as e:
        print(f"Explorer konnte nicht geöffnet werden: {e}")

    if umap_html.exists():
        try:
            webbrowser.open(umap_html.resolve().as_uri())
        except Exception as e:
            print(f"UMAP HTML konnte nicht geöffnet werden: {e}")

    print("\nTop 10 LOF-Kandidaten zur fachlichen Prüfung:")
    print(results[[
        "outlier_rank",
        "row_id",
        "lof_raw",
        "reconstruction_error",
        "string_reconstruction_error",
        "lof_z",
        "hdbscan_outlier_z",
        "centroid_distance_z",
        "cluster_label",
        "tabular_modality_share",
        "string_modality_share",
        "dominant_modality",
    ] + preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
