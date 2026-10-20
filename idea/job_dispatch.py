"""Opt-in Cloud Run Job dispatch with a credential-free, fenced work payload."""

import json
import re
from dataclasses import asdict

from idea.blocking_io import run_blocking
from idea.run_service import RunJob


_JOB_NAME = re.compile(r"projects/[A-Za-z0-9-]+/locations/[A-Za-z0-9-]+/jobs/[A-Za-z0-9-]+\Z")


class CloudJobDispatcher:
    def __init__(self, job_name: str, session=None):
        if not _JOB_NAME.fullmatch(job_name):
            raise ValueError("IDEA_RUN_JOB must be projects/<project>/locations/<region>/jobs/<job>")
        self.job_name = job_name
        self.session = session

    def _dispatch_sync(self, job: RunJob):
        session = self.session
        owned_session = session is None
        if session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            session = AuthorizedSession(credentials)
        try:
            response = session.post(
                f"https://run.googleapis.com/v2/{self.job_name}:run",
                json={"overrides": {"containerOverrides": [{"args": [
                    "--payload", json.dumps(asdict(job), separators=(",", ":")),
                ]}]}}, timeout=20,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or not result.get("name"):
                raise RuntimeError("Job dispatcher returned no operation name")
            return result["name"]
        finally:
            if owned_session:
                session.close()

    async def dispatch(self, job: RunJob) -> str:
        return await run_blocking(self._dispatch_sync, job, timeout=30)


def parse_job_payload(raw: str) -> RunJob:
    payload = json.loads(raw)
    fields = {"user_id", "evolution_id", "run_token", "mode", "additional_generations"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Invalid job payload fields")
    for name in ("user_id", "evolution_id", "run_token"):
        value = payload[name]
        if not isinstance(value, str) or not value.strip() or len(value) > 256 or "/" in value or "\\" in value:
            raise ValueError(f"Invalid job {name}")
    if payload["mode"] not in {"new", "resume", "continue"}:
        raise ValueError("Invalid job mode")
    count = payload["additional_generations"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > 500:
        raise ValueError("Invalid job additional_generations")
    if payload["mode"] != "continue" and count:
        raise ValueError("Only continuation jobs may request additional generations")
    return RunJob(**payload)
