"""
LLM provider factory for abm-gods-eye.

Supports Anthropic, OpenAI, and Google (Gemini) via their respective
LangChain integrations. Provider packages are optional extras — only
the one you use needs to be installed.

Usage::

    from abm_gods_eye.llm import make_llm

    llm = make_llm("anthropic", "claude-sonnet-4-6")
    llm = make_llm("openai", "gpt-4o")
    llm = make_llm("google", "gemini-2.0-flash")
"""

from langchain_core.language_models import BaseChatModel

# Default models per provider — used when caller omits the model name
_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash",
}

SUPPORTED_PROVIDERS: list[str] = list(_DEFAULTS.keys())


def make_llm(provider: str, model: str | None = None) -> BaseChatModel:
    """
    Instantiate a LangChain chat model for the given provider and model name.

    Parameters
    ----------
    provider:
        One of ``"anthropic"``, ``"openai"``, or ``"google"``.
        Case-insensitive.
    model:
        The model identifier to use. If omitted, a sensible default is chosen
        for the provider (e.g. ``"claude-sonnet-4-6"`` for Anthropic).

    Returns
    -------
    BaseChatModel
        A ready-to-use LangChain chat model. API keys are read from environment
        variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
        ``GOOGLE_API_KEY``) — load a ``.env`` file before calling this if
        needed.

    Raises
    ------
    ValueError
        If the provider is not recognised.
    ImportError
        If the required provider package is not installed.
    """
    provider = provider.strip().lower()
    resolved_model = model or _DEFAULTS.get(provider)

    if resolved_model is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    match provider:
        case "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "langchain-anthropic is not installed. "
                    "Run: uv add langchain-anthropic"
                )
            return ChatAnthropic(model=resolved_model)  # type: ignore[call-arg]

        case "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai is not installed. "
                    "Run: uv add langchain-openai  (or: uv add 'abm-gods-eye[openai]')"
                )
            return ChatOpenAI(model=resolved_model)

        case "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is not installed. "
                    "Run: uv add langchain-google-genai  (or: uv add 'abm-gods-eye[google]')"
                )
            return ChatGoogleGenerativeAI(model=resolved_model)

        case _:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
            )
