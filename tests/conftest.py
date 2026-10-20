"""Unit tests use explicit service fakes and never ambient cloud credentials."""

import socket
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def offline_services(monkeypatch, request):
    if request.node.get_closest_marker('live'):
        return
    from idea import database

    def unexpected_io(*args, **kwargs):
        raise AssertionError('External I/O in unit test: provide an explicit service fake')

    monkeypatch.setattr(database, 'get_db', unexpected_io)
    # The standard offline user has no remote run overlay. Tests exercising
    # coordination explicitly override this with their own run-state fake.
    monkeypatch.setattr(database, 'get_active_run', AsyncMock(return_value=None))
    monkeypatch.setattr(socket.socket, 'connect', unexpected_io)
    monkeypatch.setattr(socket.socket, 'connect_ex', unexpected_io)
    monkeypatch.setattr(socket, 'create_connection', unexpected_io)
    for name in ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS'):
        monkeypatch.delenv(name, raising=False)
