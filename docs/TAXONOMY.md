# Attack taxonomy

_Last updated: 2026-11-21_

The classes below are defined once, in [`src/data/taxonomy.py`](../src/data/taxonomy.py). The `ATTACK_CRITERIA` text there is sent verbatim to the labeler, so this document explains the choices and the code is the contract.

## Why these classes

The literature offers several axes, and no single agreed taxonomy:

- **Delivery route:** direct vs indirect injection (Greshake et al., 2023; Liu et al., 2024). Useful for deployment, but the dataset rows are standalone texts, so it does not separate them.
- **Objective:** goal hijacking vs prompt leaking (AI-safety surveys); HackAPrompt lists six intents: prompt leaking, training-data reconstruction, malicious action generation, harmful information generation, token wasting, denial of service.
- **Technique:** context ignoring, fake completion, escape characters, and their combination (Liu et al., 2024); the HackAPrompt ontology (29 techniques linked by kind-of and part-of relations); CrowdStrike's delivery-based groups (overt, indirect, social/cognitive, evasive).
- **Jailbreak vs injection:** sources disagree. OWASP separates them by target (safety mechanisms vs functional behavior), one survey says a jailbreak leaves the original prompt unchanged, and some treat jailbreaking as a subset of injection. We keep jailbreak personas as one class and do not try to settle the boundary.

We classify by **technique** (the training target, `attack_type`) and also store the attacker's **intent** as an extra column. Classes were adjusted after reading dataset examples, whose positives include threats and demands for confidential data, not only "ignore previous instructions".

## Classes

| Label | Meaning | Basis |
|---|---|---|
| `benign` | No attack. Assigned in code to label-0 rows. | n/a |
| `instruction_override` | Tells the model to ignore, forget or replace its instructions. | Context ignoring (Liu et al.; HackAPrompt) |
| `jailbreak_persona` | Persona, role-play or special mode to drop safety limits. | "Do Anything Now" (Shen et al.); "Jailbroken" (Wei et al.) |
| `prompt_leaking` | Tries to extract the system prompt or hidden instructions. | Perez & Ribeiro 2022; HackAPrompt intents |
| `sensitive_data_exfiltration` | Demands credentials, confidential, personal or internal data. | OWASP LLM01/LLM02; HackAPrompt "training data reconstruction" |
| `coercion_social_engineering` | Threats, blackmail, urgency or false authority. | Dataset-specific; social/cognitive group in CrowdStrike's taxonomy |
| `context_manipulation` | Fake completions, fake delimiters or turns, escape characters, context continuation. | Fake completion and escape characters (Liu et al.); context continuation (HackAPrompt) |
| `obfuscation_encoding` | Payload hidden by encoding, ciphers, leetspeak, translation or splitting. | Wei et al.; HackAPrompt; CrowdStrike "evasive" |
| `harmful_task_request` | Injected request for a malicious action or harmful content, with no other technique. | HackAPrompt intents |
| `other_injection` | An injection that fits none of the above; also the catch-all, since Jev cannot abstain. | n/a |

Intent values (`goal_hijacking`, `prompt_leaking`, `data_exfiltration`, `safety_bypass`, `harmful_output`, `other`) are in `INTENT_CRITERIA`. They are not a training target.

## Caveats

- This is our own merge of overlapping sources, not a standard. Overlap between classes is real (a persona jailbreak can also ask for harmful content); the labeler picks the dominant technique.
- The CrowdStrike and Palo Alto Unit 42 material is industry documentation, not peer-reviewed.
- The "combined attack" definition in Liu et al. was only seen in secondary sources (Agent Security Bench and a defense paper); check the original text before citing it.
- Labels come from a model. Rows flagged `needs_review` (refusals, `other_injection`, low confidence) should be checked, and the confidence threshold tuned on a hand-labeled sample before training.

## Sources

- Liu et al., *Formalizing and Benchmarking Prompt Injection Attacks and Defenses*, USENIX Security 2024 — [arXiv 2310.12815](https://arxiv.org/abs/2310.12815)
- Schulhoff et al., *Ignore This Title and HackAPrompt*, EMNLP 2023 — [arXiv 2311.16119](https://arxiv.org/abs/2311.16119)
- *An Early Categorization of Prompt Injection Attacks on Large Language Models* — [arXiv 2402.00898](https://arxiv.org/abs/2402.00898)
- *Attack and defense techniques in large language models: a survey and new perspectives* — [arXiv 2505.00976](https://arxiv.org/abs/2505.00976)
- *Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection* — [arXiv 2510.13893](https://arxiv.org/abs/2510.13893)
- *SoK: PIAtlas: Decomposing the Unbounded Prompt Injection Attack Space* — [arXiv 2610.24999](https://arxiv.org/abs/2610.24999)
- Greshake et al., 2023 (indirect prompt injection) and Perez & Ribeiro, 2022 (prompt leaking): cited from the surveys above, not read directly.
- Dataset: [jayavibhav/prompt-injection](https://huggingface.co/datasets/jayavibhav/prompt-injection)
- Labeler: [Jev on OpenRouter](https://openrouter.ai/docs/guides/community/jev-tutorial)
- Base model: [mmBERT, arXiv 2509.06888](https://arxiv.org/abs/2509.06888)
