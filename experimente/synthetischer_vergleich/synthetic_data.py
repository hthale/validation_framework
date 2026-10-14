"""Festgelegte künstliche Attributprodukte; Wahrheit strikt getrennt von Merkmalen."""
from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

TARGET = "pruefcode"
CONTROL = "kontrolle"
SIGNALS = ["messwert_a", "messwert_b", "messwert_c", "messwert_d"]
SCENARIOS = ("basis", "stoermerkmale")
PROFILES = ("A", "B", "C")
WEIGHTS = (.70, .25, .05)
CENTERS = {"A": (-4., -4., -4., -4.), "B": (0., 0., 0., 0.), "C": (4., 4., 4., 4.)}
NOISE_COLUMNS = [f"zusatz_{i:02d}" for i in range(12)]


def digest(*values):
    payload = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def allocate(n, weights):
    raw = np.asarray(weights)*n
    counts = np.floor(raw).astype(int)
    for index in sorted(range(len(weights)), key=lambda i: (-(raw[i]-counts[i]), i))[:n-int(counts.sum())]:
        counts[index] += 1
    return counts.tolist()


def create_population(seed=42, units=1200):
    """Zwei Beobachtungen je Einheit; Kontrolle paarweise ausgeglichen.

    Beide Szenarien erhalten dieselben Signalwerte, Gruppen und Aufteilungen.
    Seltene Gruppe C: 5 %. Gruppenquoten sind eine bewusste Benchmarkentscheidung.
    """
    rng = np.random.default_rng(seed)
    data, truth = [], []
    counts = allocate(units, WEIGHTS)
    for profile, count in zip(PROFILES, counts):
        partitions = np.array(["train"]*allocate(count, (.6, .2, .2))[0] +
                              ["validation"]*allocate(count, (.6, .2, .2))[1] +
                              ["test"]*allocate(count, (.6, .2, .2))[2])
        rng.shuffle(partitions)
        for number in range(count):
            group = "unit13:"+digest(seed, profile, number)
            # Kompakte Gruppen; innerhalb einer Gruppe sind alle vier Messwerte
            # frei innerhalb +/-0.6 zulässig. Keine räumlichen Koordinaten.
            values = np.asarray(CENTERS[profile]) + rng.uniform(-.6, .6, len(SIGNALS))
            nuisance = rng.uniform(-1., 1., len(NOISE_COLUMNS))
            for observation in (0, 1):
                key = "rk13:"+digest(group, observation)
                data.append({"record_key": key, **dict(zip(SIGNALS, values)),
                             **dict(zip(NOISE_COLUMNS, nuisance)), TARGET: profile,
                             CONTROL: f"K{observation}"})
                truth.append({"record_key": key, "group_id": group, "partition": partitions[number],
                              "profile": profile, "rare_normal": profile == "C", "observation": observation})
    attributes, metadata = pd.DataFrame(data), pd.DataFrame(truth)
    attributes = attributes.sort_values("record_key").reset_index(drop=True)
    metadata = metadata.set_index("record_key").loc[attributes.record_key].reset_index()
    return attributes, metadata


def scenario_frame(all_attributes, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Unbekanntes Szenario.")
    features = SIGNALS + (NOISE_COLUMNS if scenario == "stoermerkmale" else [])
    return all_attributes[["record_key", *features, TARGET, CONTROL]].copy(), features


def normal_mask(frame):
    """Referenzdefinition: Code und sämtliche Signalintervalle müssen passen."""
    good = np.zeros(len(frame), dtype=bool)
    for profile in PROFILES:
        center = np.asarray(CENTERS[profile])
        within = (np.abs(frame[SIGNALS].to_numpy(dtype=float)-center) <= .6+1e-12).all(axis=1)
        good |= within & frame[TARGET].eq(profile).to_numpy()
    for col in NOISE_COLUMNS:
        if col in frame:
            good &= frame[col].between(-1., 1.).to_numpy()
    return good


def make_test_variants(train, test, metadata, seed, fraction=.20):
    """20 % der Testeinheiten: erste Beobachtung ändern, zweite bleibt Kontrolle.

    Damit etwa 10 % fehlerhafte Zeilen im Prüfbestand. Derselbe Bestand und
    dieselben betroffenen Kennungen für alle Vergleichsmethoden und Fehlerarten.
    """
    meta = metadata.set_index("record_key").loc[test.record_key].reset_index()
    if not normal_mask(test).all() or not normal_mask(train).all():
        raise ValueError("Referenzbestand ist nicht gültig gemäß Generator.")
    selected = set()
    for profile in PROFILES:
        group_ids = sorted(meta.loc[meta.profile.eq(profile), "group_id"].unique(),
                           key=lambda g: digest("inject13", seed, g))
        count = max(1, int(round(len(group_ids)*fraction))) if group_ids else 0
        selected.update(group_ids[:count])
    errors = (meta.group_id.isin(selected) & meta.observation.eq(0)).to_numpy()
    code_changed, signal_changed = test.copy(deep=True), test.copy(deep=True)
    next_profile = {"A": "B", "B": "C", "C": "A"}
    for pos in np.flatnonzero(errors):
        profile = test[TARGET].iloc[pos]
        alternative = next_profile[profile]
        code_changed.loc[code_changed.index[pos], TARGET] = alternative
        # Ein tatsächlich im Training vorhandener Messwert einer anderen Gruppe.
        donors = train.loc[train[TARGET].eq(alternative)].sort_values("record_key")
        pick = int(digest("donor13", seed, test.record_key.iloc[pos]), 16) % len(donors)
        signal_changed.loc[signal_changed.index[pos], "messwert_b"] = donors["messwert_b"].iloc[pick]
    labels = meta.copy()
    labels["is_error"] = errors
    variants = {"clean": test.copy(deep=True), "code_swap": code_changed, "signal_swap": signal_changed}
    for name, changed in variants.items():
        expected = np.zeros(len(test), dtype=bool) if name == "clean" else errors
        if not np.array_equal(~normal_mask(changed), expected):
            raise AssertionError("Erzeugte Fehler stimmen nicht mit unabhängiger Referenzdefinition überein.")
        if changed.record_key.tolist() != test.record_key.tolist():
            raise AssertionError("Kennungen durch Fehlererzeugung verändert.")
    return variants, labels
