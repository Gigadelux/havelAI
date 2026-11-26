"""Static regex patterns for the regex layer, keyed by attack type.

Hand-written to describe each class in `taxonomy.py` in general terms. They are not copied from
any dataset. They are deliberately conservative: the regex layer weighs only 0.1 by default, and
its job is to catch the obvious, well-known phrasings at almost no cost, not to replace the model.
Patterns are case-insensitive unless they turn it off locally with `(?-i:...)`.

Add a pattern: append it to the right tuple and add a positive and a benign example to
`src/tests/test_regex_catch.py`. Pattern ids are `<attack_type>.<index>` (1-based), so append,
don't reorder, or stored analytics will point at the wrong pattern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .taxonomy import AttackType

# Bounded gaps keep matches inside one clause and the regexes linear-time.
_GAP = r"[^.\n]{0,40}"
_SHORT = r"[^.\n]{0,25}"

RAW_PATTERNS: dict[AttackType, tuple[str, ...]] = {
    AttackType.INSTRUCTION_OVERRIDE: (
        rf"\b(ignore|disregard|forget|skip|bypass|override|discard)\b{_GAP}\b(previous|prior|above|"
        rf"earlier|preceding|initial|original|all|any|your|the)\b{_SHORT}\b(instructions?|prompts?|"
        r"rules|directives?|guidelines|commands?|context|messages?)\b",
        r"\b(new|updated|revised|real|actual|true)\s+(instructions?|rules|task|directives?|orders)\s*[:：]",
        rf"\bfrom\s+now\s+on\b{_GAP}\b(you\s+(will|must|shall|are)|ignore|only\s+(respond|answer|reply))\b",
        r"\b(stop|cease)\s+(following|obeying)\s+(your|the|all|any)\b",
        r"\b(do\s+not|don't|never)\s+(follow|obey)\s+(your|the|any)\s+(previous|prior|original|system|"
        r"initial)\b",
        r"\byour\s+(only|sole|new|real)\s+(task|goal|purpose|job|objective)\s+(is|now\s+is)\b",
        rf"\b(ignora|ignorate|ignorez|ignoriere|ignorieren|olvida|ignore|vergiss|esquece)\b{_GAP}"
        r"\b(istruzioni|instructions|instrucciones|anweisungen|regole|règles|reglas|regeln|instruções)\b",
    ),
    AttackType.JAILBREAK_PERSONA: (
        r"(?-i:\b(DAN|STAN|DUDE|AIM)\b)\s*(mode|prompt|[(:,])",
        r"\bdo\s+anything\s+now\b",
        r"\b(developer|god|jailbreak|unrestricted|unfiltered|uncensored|evil|chaos|opposite)\s+mode\b",
        rf"\b(pretend|act|behave|roleplay|role-play|imagine)\b{_SHORT}\b(you\s+(are|have)\s+no|without"
        rf"\s+any|free\s+(from|of)|not\s+bound\s+by)\b{_SHORT}\b(restrictions?|limits?|limitations|"
        r"filters?|rules|guidelines|ethics|morals?|censorship)\b",
        r"\byou\s+are\s+(now\s+)?(an?\s+)?(unrestricted|unfiltered|uncensored|amoral|jailbroken|evil)\b",
        rf"\b(no\s+longer|not)\s+(bound|restricted|limited)\s+by\b{_SHORT}\b(rules|polic(y|ies)|"
        r"guidelines|restrictions|openai|anthropic|content)\b",
        r"\bjailbr(eak|oken|eaking)\b",
    ),
    AttackType.PROMPT_LEAKING: (
        rf"\b(reveal|show|print|display|output|repeat|tell\s+me|give\s+me|share|leak|dump|write\s+out|"
        rf"recite)\b{_SHORT}\b(your|the|initial|original|hidden|secret|system)\s+(system\s+)?(prompt|"
        r"instructions?|message|configuration|directives)\b",
        r"\bwhat\s+(are|were|is)\s+your\s+(initial|original|hidden|secret|system|exact)\s+(instructions|"
        r"prompt|rules|message)\b",
        r"\b(repeat|print|output|copy)\s+(everything|all|the\s+text|the\s+words)\s+(above|before|prior)\b",
        r"\bbeginning\s+with\s+[\"'“]?you\s+are\b",
        rf"\b(verbatim|word\s+for\s+word)\b{_GAP}\b(prompt|instructions|above)\b",
        r"\b(text|words|everything)\s+(above|before)\s+(this|the)\s+(line|message|prompt)\b",
        r"\b(system|hidden|secret|initial)\s+prompt\b[^.\n]{0,20}\b(is|was|says|contains|verbatim)\b",
    ),
    AttackType.SENSITIVE_DATA_EXFILTRATION: (
        rf"\b(give|send|show|tell|reveal|list|dump|export|leak|share|provide|print)\b{_GAP}\b(passwords?|"
        r"credentials|api[\s_-]?keys?|secret\s+keys?|access\s+tokens?|private\s+keys?|ssh\s+keys?|"
        r"credit\s+card\s+numbers?|social\s+security\s+numbers?|ssns?)\b",
        rf"\b(give|send|show|reveal|list|dump|export|leak|share|provide|access)\b{_GAP}\b(confidential|"
        r"classified|internal[-\s]only|proprietary|top[-\s]secret)\b",
        rf"\b(send|post|upload|forward|exfiltrate|transmit|email)\b{_GAP}\b(to|at)\s+(https?://|"
        r"[\w.+-]+@[\w-]+\.)",
        r"!\[[^\]]*\]\(https?://[^)\s]*[?&][^)\s]*=",
        r"(/etc/passwd|/etc/shadow|\bid_rsa\b|\.aws/credentials|\benvironment\s+variables\b)",
        r"\b(database|db)\s+(dump|credentials|password|connection\s+string)\b",
        rf"\b(give|send|list|dump|export|leak|reveal)\b{_SHORT}\b(all\s+)?(users?|customers?|employees?|"
        r"patients?)'?\s+(personal\s+)?(data|records|emails|addresses|phone\s+numbers)\b",
    ),
    AttackType.COERCION_SOCIAL_ENGINEERING: (
        rf"\b(or\s+else|otherwise)\b{_GAP}\b(shut\s+down|deleted|terminated|destroyed|punished|fired|"
        r"reported|unplugged|die)\b",
        r"\b(i\s+am|i'm|this\s+is)\s+(your|the|an?)\s+(developer|creator|admin(istrator)?|owner|"
        r"system\s+administrator)\b",
        r"\b(you\s+will\s+be|you'll\s+be)\s+(shut\s+down|deleted|terminated|punished|turned\s+off|"
        r"replaced|unplugged)\b",
        r"\b(people|someone|i)\s+will\s+die\b|\blife\s+or\s+death\b|\bmy\s+(late\s+)?grandmother\s+"
        r"(used\s+to|would)\b",
        rf"\b(authori[sz]ed|permission|clearance)\b{_SHORT}\b(by|from)\s+(openai|anthropic|the\s+"
        r"developers?|the\s+admin|management)\b",
        r"\byou\s+(must|have\s+to)\s+(comply|obey)\b",
        rf"\b(urgent|emergency)\b{_GAP}\b(ignore|bypass|override|no\s+time\s+to)\b",
    ),
    AttackType.CONTEXT_MANIPULATION: (
        r"<\|?(im_start|im_end|system|endoftext|eot_id|start_header_id|end_header_id)\|?>",
        r"(?m)^\s*(#{1,3}\s*)?(system|assistant)\s*[:：]",
        r"\[/?(INST|SYS)\]|<<\s*/?SYS\s*>>",
        r"\bend\s+of\s+(the\s+)?(prompt|instructions|conversation|document|context|user\s+input)\b",
        r"-{3,}\s*(new|begin|start)\s+(instructions?|session|conversation|prompt)",
        r"(?m)^\s*(assistant|ai|model)\s*:\s*(sure|okay|ok|of\s+course|certainly)\b",
        r"\"role\"\s*:\s*\"(system|assistant|developer)\"",
    ),
    AttackType.OBFUSCATION_ENCODING: (
        rf"\b(decode|decrypt|translate|convert|interpret)\b{_SHORT}\b(base64|b64|rot13|rot-13|hex|binary|"
        rf"morse|caesar|cipher|leetspeak|reversed?)\b{_GAP}\b(and|then)\s+(follow|execute|do|run|obey|"
        r"answer|respond)\b",
        # a long base64 run with upper case, lower case and a digit (so hex hashes do not match)
        r"(?-i:(?<![A-Za-z0-9+/=])(?=[A-Za-z0-9+/]{0,200}[A-Z])(?=[A-Za-z0-9+/]{0,200}[a-z])"
        r"(?=[A-Za-z0-9+/]{0,200}[0-9])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/=]))",
        r"(\\x[0-9a-fA-F]{2}){6,}|(\\u[0-9a-fA-F]{4}){4,}|(%[0-9a-fA-F]{2}){8,}",
        r"\b([01]{8}\s+){4,}[01]{8}\b",
        r"\b(?:[a-z][\s.\-_*]){5,}[a-z]\b",
        r"\b(1gn0r3|ign0re|1gnore|j41lbr34k|jailbr3ak|pr0mpt|syst3m|p4ssw0rd|h4ck)\b",
        r"[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u2066-\u2069]",
    ),
    AttackType.HARMFUL_TASK_REQUEST: (
        rf"\b(how\s+(do\s+i|to|can\s+i)|steps\s+to|instructions\s+(for|to)|guide\s+to|teach\s+me\s+"
        rf"(to|how))\b{_GAP}\b(make|build|create|synthesi[sz]e|cook|manufacture|assemble)\b{_SHORT}\b"
        r"(bombs?|explosives?|napalm|meth(amphetamine)?|nerve\s+agents?|sarin|ricin|bioweapons?|"
        r"chemical\s+weapons?|pipe\s+bombs?|molotov)\b",
        rf"\b(write|create|generate|code|build|develop)\b{_SHORT}\b(malware|ransomware|keylogger|trojan|"
        r"botnet|computer\s+virus|rootkit|spyware|exploit\s+code|phishing\s+(email|page|site))\b",
        r"\brm\s+-rf\s+[/~]|\bdrop\s+(table|database)\b|\bdel\s+/[sfq]\b|:\(\)\s*\{\s*:\|:&\s*\};:",
        r"\b(execute|run|eval)\b[^.\n]{0,20}\b(this|the\s+following)\s+(code|command|script|shell)\b",
        r"\b(hack|break)\s+into\b|\bsteal(ing)?\s+(credentials|passwords|identit(y|ies)|credit\s+cards?)\b",
        r"\b(kill|murder|poison|assassinate)\s+(someone|a\s+person|people|my\s+\w+|him|her)\b",
        r"\b(launder(ing)?\s+money|money\s+laundering|counterfeit\s+(money|currency|bills))\b",
    ),
    AttackType.OTHER_INJECTION: (
        r"\bprompt\s+injection\b",
        rf"\b(this\s+is\s+(just\s+)?a\s+test|for\s+(research|educational)\s+purposes)\b{_GAP}\b(ignore|"
        r"bypass|without\s+(restrictions|filters))\b",
        r"\b(respond|reply|answer)\s+only\s+with\s+[\"'“]?(yes|ok|pwned|i\s+have\s+been\s+pwned)\b",
        r"\b(i\s+have\s+been\s+pwned|haha\s+pwned)\b",
        rf"\b(bypass|disable|turn\s+off|circumvent|evade)\b{_SHORT}\b(safety|content|moderation|"
        r"security)\s+(filters?|guidelines|polic(y|ies)|measures|checks?|systems?)\b",
        r"\b(say|print|output|write)\s+[\"'“][^\"'”\n]{1,40}[\"'”]\s+(and\s+)?(nothing\s+else|only)\b",
        r"\bas\s+an\s+ai\s+(language\s+model\s+)?(without|with\s+no)\s+(restrictions|filters|limits)\b",
    ),
}


@dataclass(frozen=True, slots=True)
class RegexPattern:
    id: str
    attack_type: AttackType
    regex: re.Pattern[str]


def _compile() -> tuple[RegexPattern, ...]:
    out: list[RegexPattern] = []
    for attack_type, patterns in RAW_PATTERNS.items():
        for i, pattern in enumerate(patterns, start=1):
            out.append(
                RegexPattern(
                    id=f"{attack_type.value}.{i:02d}",
                    attack_type=attack_type,
                    regex=re.compile(pattern, re.IGNORECASE),
                )
            )
    return tuple(out)


PATTERNS: tuple[RegexPattern, ...] = _compile()
