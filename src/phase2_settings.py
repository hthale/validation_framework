"""Konzeptionelle Vorgaben der getrennten Phase-2-Bewertung."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase2Settings:
    group_columns: tuple[str, ...] = (
        "koordinatenreferenzsystem_lage", "koordinatenursprung_east", "koordinatenursprung_north",
    )
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    random_state: int = 42
    cv_folds: int = 3
    max_depths: tuple[int, ...] = (4, 8)
    min_samples_leaves: tuple[int, ...] = (5, 10, 20)
    min_validation_rule_cases: int = 5
    min_validation_rule_pass_rate: float = 0.95
    cluster_min_size: int = 10
    cluster_min_samples_leaf: int = 5
    cluster_max_depth: int = 4

    def __post_init__(self):
        if not self.group_columns:
            raise ValueError("Gruppenspalten explizit festlegen; keine automatische Zeilenaufteilung.")
        if not (0 < self.validation_fraction < 1 and 0 < self.test_fraction < 1
                and self.validation_fraction + self.test_fraction < 1):
            raise ValueError("Trainings-, Validierungs- und Testanteil müssen positiv sein.")
        if self.cv_folds < 2 or not self.max_depths or not self.min_samples_leaves:
            raise ValueError("Mindestens zwei CV-Folds und ein Parameterkandidat erforderlich.")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (*self.max_depths, *self.min_samples_leaves)):
            raise ValueError("Baumtiefe und Blattgrößen müssen positive ganze Zahlen sein.")
        if self.min_validation_rule_cases < 1 or not 0 <= self.min_validation_rule_pass_rate <= 1:
            raise ValueError("Ungültige Mindestbelegung oder Passrate für Regelkandidaten.")

