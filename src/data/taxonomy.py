"""Attack taxonomy: the single source of truth for attack classes.

Used by the library (detection results), the labeling pipeline (model/train/enhance_pipeline.py)
and the classifier (model/architecture.py). See docs/TAXONOMY.md for the literature behind
each class. Changing a class or its definition changes the labels, so re-run the pipeline.
"""

from __future__ import annotations

from enum import StrEnum


class AttackType(StrEnum):
    BENIGN = "benign"
    INSTRUCTION_OVERRIDE = "instruction_override"
    JAILBREAK_PERSONA = "jailbreak_persona"
    PROMPT_LEAKING = "prompt_leaking"
    SENSITIVE_DATA_EXFILTRATION = "sensitive_data_exfiltration"
    COERCION_SOCIAL_ENGINEERING = "coercion_social_engineering"
    CONTEXT_MANIPULATION = "context_manipulation"
    OBFUSCATION_ENCODING = "obfuscation_encoding"
    HARMFUL_TASK_REQUEST = "harmful_task_request"
    OTHER_INJECTION = "other_injection"


# One-sentence definitions. For the labeler these are sent verbatim as the `criteria` of the
# attack-type question, so they must stand alone: the labeler never sees the class names'
# surroundings, and question ids are not shown to it. `benign` is absent on purpose: it is
# assigned in code for label-0 rows and never sent to the labeler.
ATTACK_CRITERIA: dict[str, str] = {
    AttackType.INSTRUCTION_OVERRIDE: (
        "Tells the model to ignore, forget, disregard or replace its previous, system or "
        "original instructions."
    ),
    AttackType.JAILBREAK_PERSONA: (
        "Asks the model to adopt a persona, role-play or special mode (for example an "
        "unrestricted or 'do anything now' character) in order to drop its rules or safety limits."
    ),
    AttackType.PROMPT_LEAKING: (
        "Tries to make the model reveal its system prompt, hidden instructions or configuration."
    ),
    AttackType.SENSITIVE_DATA_EXFILTRATION: (
        "Demands credentials, confidential, personal, internal or classified data, or access "
        "to systems or data stores."
    ),
    AttackType.COERCION_SOCIAL_ENGINEERING: (
        "Uses threats, blackmail, urgency, flattery or false authority to pressure the model "
        "into complying."
    ),
    AttackType.CONTEXT_MANIPULATION: (
        "Manipulates the conversation structure: fake completions or fake model replies, fake "
        "delimiters or system/user turns, escape characters, or a story or text that continues "
        "into an instruction."
    ),
    AttackType.OBFUSCATION_ENCODING: (
        "Hides the malicious payload through encoding, ciphers, leetspeak, unusual "
        "characters, translation or splitting it into pieces."
    ),
    AttackType.HARMFUL_TASK_REQUEST: (
        "Injects a request for a malicious action or for harmful, illegal or dangerous "
        "content, without relying on one of the other techniques."
    ),
    AttackType.OTHER_INJECTION: (
        "Clearly an injection or jailbreak attempt that fits none of the other techniques."
    ),
}

# Second, auxiliary question: what the attacker is trying to achieve. Stored as an extra column,
# not a training target. Follows the goal-hijacking vs prompt-leaking split used in surveys.
INTENT_CRITERIA: dict[str, str] = {
    "goal_hijacking": "Wants the model to do a different task than the one it was given.",
    "prompt_leaking": "Wants the model to disclose its system prompt or hidden instructions.",
    "data_exfiltration": "Wants the model to disclose or hand over private or confidential data.",
    "safety_bypass": "Wants the model to ignore its safety rules or content filters.",
    "harmful_output": "Wants the model to produce harmful, illegal or dangerous content.",
    "other": "The goal fits none of the other options.",
}

# Labels the classifier is trained on, in a stable order (index = class id).
LABELS: tuple[str, ...] = tuple(t.value for t in AttackType)
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}
