"""Provider failure recovery stays bounded at the actual HTTP request boundary."""

import socket
import threading
from unittest.mock import Mock, patch

import httpx
import pytest
from google import genai

from idea.llm import (
    GenerationCancelledError,
    GenerationResponseDeadlineError,
    Ideator,
    MAX_GENERATION_ATTEMPTS,
    _is_response_timeout_error,
    _is_transient_generation_error,
)


def make_wrapper(**kwargs):
    with patch("idea.llm.LLMWrapper._setup_provider"):
        return Ideator(**kwargs)


def run_without_sleep(wrapper):
    return wrapper.generate_text.retry_with(sleep=lambda _: None)(wrapper, "test prompt")


def http_error(status, message="provider failed"):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    return httpx.HTTPStatusError(message, request=response.request, response=response)


@pytest.mark.parametrize("error", [
    socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
    httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known"),
    http_error(429),
    http_error(500),
    http_error(503),
])
@pytest.mark.parametrize("provider,method", [
    ("google_generative_ai", "_generate_content"),
    ("openai", "_generate_openai_response"),
])
def test_transient_generation_failure_recovers_for_both_providers(error, provider, method):
    wrapper = make_wrapper(provider=provider)
    with patch.object(wrapper, method, side_effect=[error, "recovered"]) as generate:
        assert run_without_sleep(wrapper) == "recovered"
    assert generate.call_count == 2
    event = wrapper.get_diagnostic_events()[-1]
    assert event["key"] == "generation_retries"
    assert event["metadata"]["next_attempt"] == 2
    assert 1 <= event["metadata"]["delay_seconds"] <= 2
    assert "test prompt" not in str(event)


@pytest.mark.parametrize("error", [
    http_error(400, "invalid request timed out"),
    http_error(401),
    http_error(402, "resource exhausted"),
    http_error(403),
    http_error(429, '{"code": "insufficient_quota"}'),
    http_error(429, '{"code": "billing_hard_limit_reached"}'),
    ValueError("invalid argument"),
    GenerationCancelledError("cancelled"),
    GenerationResponseDeadlineError("accepted response timed out"),
])
def test_permanent_or_unsafe_failures_are_not_retried(error):
    wrapper = make_wrapper()
    with patch.object(wrapper, "_generate_content", side_effect=error) as generate:
        with pytest.raises(type(error)):
            run_without_sleep(wrapper)
    assert generate.call_count == 1
    assert wrapper.get_diagnostics().get("generation_retries", 0) == 0
    assert not _is_transient_generation_error(error)


def test_cancellation_during_retry_wait_prevents_another_provider_attempt():
    wrapper = make_wrapper()
    wrapper.cancel_event = threading.Event()
    generate = Mock(side_effect=httpx.ConnectError("DNS unavailable"))
    retry = wrapper.generate_text.retry_with(sleep=lambda _: wrapper.cancel_event.set())
    with patch.object(wrapper, "_generate_content", generate):
        with pytest.raises(GenerationCancelledError):
            retry(wrapper, "test prompt")
    assert generate.call_count == 1


@pytest.mark.parametrize("error,accepted_request_uncertain", [
    (httpx.ConnectError("DNS unavailable"), False),
    (httpx.ConnectTimeout("Could not connect"), False),
    (socket.gaierror(socket.EAI_AGAIN, "DNS unavailable"), False),
    (httpx.ReadTimeout("No response yet"), True),
    (socket.timeout("No response yet"), True),
    (http_error(408), True),
    (http_error(504), True),
])
def test_response_timeout_policy_distinguishes_connection_failures(error, accepted_request_uncertain):
    assert _is_response_timeout_error(error) is accepted_request_uncertain


@pytest.mark.parametrize("status,expected_attempts", [(503, MAX_GENERATION_ATTEMPTS), (429, MAX_GENERATION_ATTEMPTS), (403, 1)])
def test_sdk_does_not_multiply_application_retries(status, expected_attempts):
    wrapper = make_wrapper(model_name="gemini-4-argon")
    assert wrapper._http_options.retry_options.attempts == 1
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {
            "code": status,
            "message": "Temporary provider failure" if status != 403 else "Permission denied",
            "status": "RESOURCE_EXHAUSTED" if status == 429 else "UNAVAILABLE",
        }})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        options = wrapper._http_options.model_copy(update={"httpx_client": http_client})
        with genai.Client(api_key="unit-test-key", http_options=options) as client:
            with patch.object(wrapper, "_get_client", return_value=client):
                with pytest.raises(Exception):
                    run_without_sleep(wrapper)

    assert len(calls) == expected_attempts
    assert wrapper.get_diagnostics().get("generation_retries", 0) == expected_attempts - 1
    assert wrapper.get_token_usage_summary()["missing_usage_requests"] == expected_attempts
