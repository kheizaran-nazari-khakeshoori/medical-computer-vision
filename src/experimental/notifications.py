"""Notification system for completed analysis."""

import logging

logger = logging.getLogger(__name__)


def notify_completion(patient_id: str, result: dict):
    msg = f"analysis complete for {patient_id}: {result.get('label')} ({result.get('confidence',0):.2%})"
    logger.info(msg)
    return msg


def send_email_stub(to: str, subject: str, body: str):
    print(f"[email stub] to={to} subject={subject}")
    return True
