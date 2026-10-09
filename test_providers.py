"""Which AI provider the app talks to, chosen from environment variables."""
import anthropic
import pytest

from pashuplan import providers
from pashuplan.llm import OpenAICompatClient


def test_nothing_configured_means_no_provider_and_a_helpful_message():
    assert providers.resolve({}) is None
    assert "GEMINI_API_KEY" in providers.NOT_CONFIGURED and "ANTHROPIC_API_KEY" in providers.NOT_CONFIGURED
    assert "aistudio.google.com" in providers.NOT_CONFIGURED


def test_a_gemini_key_selects_gemini_with_free_tier_friendly_settings():
    p = providers.resolve({"GEMINI_API_KEY": "g-key"})
    assert p.name == "gemini"
    assert p.model == providers.DEFAULT_MODELS["gemini"]
    assert p.max_parallel == providers.GEMINI_MAX_PARALLEL == 1  # free tiers allow only a few requests a minute
    client = p.make_client()
    assert isinstance(client, OpenAICompatClient)
    assert client.base_url == providers.GEMINI_BASE_URL.rstrip("/")
    assert client.min_max_tokens >= 4096 and client.extra_body == {"reasoning_effort": "low"}


def test_google_api_key_is_accepted_as_a_second_name_for_the_gemini_key():
    assert providers.resolve({"GOOGLE_API_KEY": "g-key"}).name == "gemini"


def test_an_anthropic_key_selects_anthropic():
    p = providers.resolve({"ANTHROPIC_API_KEY": "a-key"})
    assert p.name == "anthropic" and p.model == providers.DEFAULT_MODELS["anthropic"]
    assert p.max_parallel >= 5
    assert isinstance(p.make_client(), anthropic.Anthropic)


def test_with_both_keys_anthropic_wins_unless_the_provider_is_named():
    both = {"ANTHROPIC_API_KEY": "a", "GEMINI_API_KEY": "g"}
    assert providers.resolve(both).name == "anthropic"
    assert providers.resolve({**both, "PASHUPLAN_PROVIDER": "gemini"}).name == "gemini"


def test_naming_a_provider_without_its_key_means_not_configured():
    assert providers.resolve({"PASHUPLAN_PROVIDER": "gemini", "ANTHROPIC_API_KEY": "a"}) is None


def test_an_openai_compatible_server_such_as_ollama_needs_no_key():
    env = {"PASHUPLAN_BASE_URL": "http://localhost:11434/v1", "PASHUPLAN_MODEL": "qwen2.5:7b"}
    p = providers.resolve(env)
    assert p.name == "openai-compatible" and p.model == "qwen2.5:7b" and p.max_parallel == 1
    client = p.make_client()
    assert isinstance(client, OpenAICompatClient) and client.base_url == "http://localhost:11434/v1"


def test_an_openai_compatible_server_must_name_its_model():
    with pytest.raises(ValueError, match="PASHUPLAN_MODEL"):
        providers.resolve({"PASHUPLAN_BASE_URL": "http://localhost:11434/v1"})


def test_the_model_and_parallelism_can_be_overridden():
    p = providers.resolve({"GEMINI_API_KEY": "g", "PASHUPLAN_MODEL": "gemini-3.5-flash-lite",
                           "PASHUPLAN_MAX_PARALLEL": "1"})
    assert p.model == "gemini-3.5-flash-lite" and p.max_parallel == 1


def test_a_bad_parallelism_value_is_rejected_clearly():
    with pytest.raises(ValueError, match="PASHUPLAN_MAX_PARALLEL"):
        providers.resolve({"GEMINI_API_KEY": "g", "PASHUPLAN_MAX_PARALLEL": "lots"})


def test_an_unknown_provider_name_is_rejected_clearly():
    with pytest.raises(ValueError, match="PASHUPLAN_PROVIDER"):
        providers.resolve({"PASHUPLAN_PROVIDER": "skynet", "GEMINI_API_KEY": "g"})


def test_compact_is_the_default_mode_and_agentic_can_be_chosen():
    assert providers.resolve({"GEMINI_API_KEY": "g"}).mode == "compact"
    assert providers.resolve({"GEMINI_API_KEY": "g", "PASHUPLAN_MODE": "agentic"}).mode == "agentic"
    assert providers.resolve({"ANTHROPIC_API_KEY": "a", "PASHUPLAN_MODE": " Compact "}).mode == "compact"


def test_an_unknown_mode_is_rejected_clearly():
    with pytest.raises(ValueError, match="PASHUPLAN_MODE"):
        providers.resolve({"GEMINI_API_KEY": "g", "PASHUPLAN_MODE": "turbo"})


def test_the_key_is_never_part_of_the_provider_description():
    p = providers.resolve({"GEMINI_API_KEY": "very-secret"})
    assert "very-secret" not in repr(p)
