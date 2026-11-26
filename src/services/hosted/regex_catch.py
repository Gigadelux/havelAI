"""Regex layer for the hosted service: identical to the local one (it runs on this machine)."""

from ..local.regex_catch import NAME, RegexCatcher, dominant_type, regex_catch, regex_matches

__all__ = ["NAME", "RegexCatcher", "dominant_type", "regex_catch", "regex_matches"]
