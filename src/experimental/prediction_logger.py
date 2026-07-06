"""Integrating audit logging into prediction flow."""

import logging

from src.audit import log_action

logger = logging.getLogger(__name__)


def logged_predict(predict_fn, image, user="anonymous"):
    result = predict_fn(image)
    log_action(user, "predict", str(result.get("label")))
    logger.info(f"user {user} predicted {result.get('label')}")
    return result
