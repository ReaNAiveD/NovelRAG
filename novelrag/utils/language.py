"""Language policy utilities for unified language control across all agent components.

The project uses a single `language` config field that governs:
- Prompt template selection (fallback to English if no translation exists)
- Resource content language (IDs, URIs, descriptions, narrative content)
- Intermediate reasoning language (analysis, search queries)
- Aspect schema descriptions (keys stay English)
- User interaction language (follows user input; autonomous mode uses content language)

When `language` is not configured, the LLM is instructed to follow the language
used in agent_beliefs. If beliefs are also empty, English is used as default.
"""

# Human-readable language names used in directives
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese (中文)",
    "cn": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "es": "Spanish (Español)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
}


def _lang_name(lang: str) -> str:
    """Get the human-readable name for a language code."""
    return LANGUAGE_NAMES.get(lang, lang)


def content_directive(language: str | None, beliefs: list[str] | None = None) -> str:
    """Build a directive for content generation language.

    This directive is prepended to system prompts for any LLM call that
    generates resource content, IDs, URIs, descriptions, or intermediate
    reasoning (analysis, search queries, etc.).

    Args:
        language: The configured content language code (e.g. "zh", "en"),
                  or None if not configured.
        beliefs: The agent beliefs list, used as language signal when
                 language is not configured.

    Returns:
        A directive string to prepend to the system prompt.
    """
    if language:
        lang_name = _lang_name(language)
        return (
            f"[Language Policy] All generated content — including resource IDs, URIs, "
            f"field values, descriptions, analysis, and search queries — "
            f"MUST be written in {lang_name}. "
            f"Schema/metadata keys (e.g. field names in JSON) must remain in English."
        )
    if beliefs:
        return (
            "[Language Policy] The content language is not explicitly configured. "
            "You MUST follow the language used in the agent's beliefs below for all "
            "generated content — including resource IDs, URIs, field values, "
            "descriptions, analysis, and search queries. "
            "Schema/metadata keys (e.g. field names in JSON) must remain in English."
        )
    # No language configured, no beliefs — default to English
    return (
        "[Language Policy] All generated content — including resource IDs, URIs, "
        "field values, descriptions, analysis, and search queries — "
        "MUST be written in English. "
        "Schema/metadata keys (e.g. field names in JSON) must remain in English."
    )


def schema_directive(language: str | None, beliefs: list[str] | None = None) -> str:
    """Build a directive for aspect schema/metadata generation.

    Aspect schema keys must stay in English, but descriptions and definitions
    follow the content language.

    Args:
        language: The configured content language code, or None.
        beliefs: The agent beliefs list.

    Returns:
        A directive string to prepend to the system prompt.
    """
    if language:
        lang_name = _lang_name(language)
        return (
            f"[Language Policy] Schema/metadata keys (e.g. JSON field names, "
            f"constraint names) MUST remain in English. "
            f"All descriptions, definitions, and human-readable text "
            f"MUST be written in {lang_name}."
        )
    if beliefs:
        return (
            "[Language Policy] Schema/metadata keys (e.g. JSON field names, "
            "constraint names) MUST remain in English. "
            "All descriptions, definitions, and human-readable text "
            "MUST follow the language used in the agent's beliefs."
        )
    return (
        "[Language Policy] Schema/metadata keys and all descriptions "
        "MUST be in English."
    )


def interaction_directive(language: str | None, is_autonomous: bool = False) -> str:
    """Build a directive for user-facing response language.

    For user-initiated requests: respond in the same language as the user's input.
    For autonomous goals (no user input): respond in the content language.

    Args:
        language: The configured content language code, or None.
        is_autonomous: Whether this is an autonomous goal (no direct user request).

    Returns:
        A directive string to prepend to the system prompt.
    """
    if is_autonomous:
        if language:
            lang_name = _lang_name(language)
            return (
                f"[Language Policy] This is an autonomous action with no direct user request. "
                f"Respond and generate all content in {lang_name}."
            )
        return (
            "[Language Policy] This is an autonomous action with no direct user request. "
            "Respond and generate all content in the same language as the agent's beliefs."
        )
    # User-initiated: follow user's language
    return (
        "[Language Policy] Respond in the same language as the user's input. "
        "If the user writes in a specific language, provide your entire response "
        "in that language."
    )
