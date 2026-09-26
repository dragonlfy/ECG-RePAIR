"""Clinical action surface for the first ECG Agent prototype."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ClaimAction


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    protocol_step: str
    owner_expert: str
    allowed_experts: tuple[str, ...]
    required_evidence: tuple[str, ...]
    legal_actions: tuple[ClaimAction, ...]
    aliases: tuple[str, ...]
    positive_template: str
    importance: float = 0.5

    @property
    def certified_for_withhold(self) -> bool:
        return ClaimAction.WITHHOLD in self.legal_actions


OBSERVE_ONLY = (ClaimAction.KEEP, ClaimAction.ABSTAIN)
CERTIFIED = (ClaimAction.KEEP, ClaimAction.WITHHOLD, ClaimAction.ABSTAIN)


def _spec(
    claim_id: str,
    step: str,
    owner: str,
    evidence: tuple[str, ...],
    aliases: tuple[str, ...],
    template: str,
    *,
    certified: bool = False,
    importance: float = 0.5,
    extra_experts: tuple[str, ...] = (),
) -> ClaimSpec:
    return ClaimSpec(
        claim_id=claim_id,
        protocol_step=step,
        owner_expert=owner,
        allowed_experts=(owner, "quality", *extra_experts),
        required_evidence=evidence,
        legal_actions=CERTIFIED if certified else OBSERVE_ONLY,
        aliases=aliases,
        positive_template=template,
        importance=importance,
    )


CLAIM_REGISTRY: dict[str, ClaimSpec] = {
    "axis_left": _spec(
        "axis_left",
        "conduction_axis",
        "axis",
        ("qrs_axis",),
        ("left axis deviation", "leftward axis"),
        "The measured QRS axis supports left axis deviation.",
        certified=True,
        importance=0.75,
    ),
    "axis_right": _spec(
        "axis_right",
        "conduction_axis",
        "axis",
        ("qrs_axis",),
        ("right axis deviation", "rightward axis"),
        "The measured QRS axis supports right axis deviation.",
        certified=True,
        importance=0.75,
    ),
    "axis_extreme": _spec(
        "axis_extreme",
        "conduction_axis",
        "axis",
        ("qrs_axis",),
        ("extreme axis deviation", "northwest axis"),
        "The measured QRS axis lies in the extreme-axis range.",
        importance=0.8,
    ),
    "pr_prolonged": _spec(
        "pr_prolonged",
        "conduction_axis",
        "interval",
        ("pr_ms",),
        ("prolonged pr", "first degree av block", "first-degree av block"),
        "The measured PR interval is prolonged.",
        certified=True,
        importance=0.8,
    ),
    "pr_short": _spec(
        "pr_short",
        "conduction_axis",
        "interval",
        ("pr_ms",),
        ("short pr", "pre-excitation", "wpw"),
        "The measured PR interval is short.",
        importance=0.8,
    ),
    "qrs_prolonged": _spec(
        "qrs_prolonged",
        "conduction_axis",
        "interval",
        ("qrs_ms",),
        ("wide qrs", "prolonged qrs", "bundle branch block"),
        "The measured QRS duration is prolonged.",
        importance=0.85,
        extra_experts=("morphology",),
    ),
    "qrs_intermediate": _spec(
        "qrs_intermediate",
        "conduction_axis",
        "interval",
        ("qrs_ms",),
        ("incomplete bundle branch block", "intraventricular conduction delay"),
        "The measured QRS duration is in the intermediate range.",
        importance=0.65,
        extra_experts=("morphology",),
    ),
    "low_voltage": _spec(
        "low_voltage",
        "hypertrophy_voltage",
        "morphology",
        ("low_voltage",),
        ("low qrs voltage", "low voltage"),
        "The measured limb-lead or precordial amplitudes meet low-QRS-voltage criteria.",
        certified=True,
        importance=0.6,
    ),
    "poor_r_progression": _spec(
        "poor_r_progression",
        "hypertrophy_voltage",
        "morphology",
        ("r_progression",),
        ("poor r wave progression", "poor r-wave progression"),
        "Precordial morphology supports poor R-wave progression.",
        importance=0.55,
    ),
    "pathologic_q": _spec(
        "pathologic_q",
        "ischemia_infarction",
        "morphology",
        ("pathologic_q",),
        ("pathologic q", "pathological q", "q waves"),
        "Pathologic Q waves are present in anatomically related leads.",
        importance=0.95,
    ),
    "t_wave_inversion": _spec(
        "t_wave_inversion",
        "ischemia_infarction",
        "st_t",
        ("t_wave",),
        ("t wave inversion", "t-wave inversion", "inverted t waves"),
        "T-wave inversion is measured in the stated leads.",
        importance=0.9,
    ),
    "rv_infarction": _spec(
        "rv_infarction",
        "ischemia_infarction",
        "st_t",
        ("st_v1", "st_v2"),
        ("right ventricular infarction", "right ventricular involvement"),
        "The V1-to-V2 ST pattern supports right ventricular involvement.",
        importance=1.0,
    ),
    "rhythm_irregular": _spec(
        "rhythm_irregular",
        "rate_rhythm",
        "rhythm",
        ("rr_cv",),
        ("irregular rhythm", "irregularly irregular", "atrial fibrillation"),
        "Beat-to-beat intervals support rhythm irregularity.",
        importance=1.0,
    ),
    "bradycardia": _spec(
        "bradycardia",
        "rate_rhythm",
        "rhythm",
        ("heart_rate_bpm",),
        ("bradycardia", "bradycardic", "slow rate"),
        "The measured ventricular rate is below 60 beats per minute.",
        importance=0.75,
    ),
    "tachycardia": _spec(
        "tachycardia",
        "rate_rhythm",
        "rhythm",
        ("heart_rate_bpm",),
        ("tachycardia", "tachycardic", "rapid rate"),
        "The measured ventricular rate is above 100 beats per minute.",
        importance=0.8,
    ),
}


CERTIFIED_CLAIMS = frozenset(
    claim_id for claim_id, spec in CLAIM_REGISTRY.items() if spec.certified_for_withhold
)
