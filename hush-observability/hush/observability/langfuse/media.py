"""Media utilities for Langfuse tracing.

This module provides utilities for resolving media attachments
and injecting them into Langfuse trace data.
"""
from typing import Any, Dict
import base64


def resolve_media_for_langfuse(
    trace_data: Dict[str, Any],
    media_attachments: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve media attachments and inject into trace data for Langfuse.

    This function runs in the flush subprocess where LangfuseMedia is available.

    Args:
        trace_data: The trace data dict (input, output, metadata)
        media_attachments: Serialized media attachments

    Returns:
        Modified trace_data with LangfuseMedia objects injected
    """
    try:
        from langfuse.media import LangfuseMedia
    except ImportError:
        # If langfuse is not installed, return trace_data unchanged
        return trace_data

    # Group attachments by target location
    by_location: Dict[str, Dict[str, Any]] = {
        "input": {},
        "output": {},
        "metadata": {},
    }

    for key, attachment in media_attachments.items():
        attach_to = attachment.get("attach_to", "metadata")
        content_type = attachment["content_type"]

        # Get content bytes
        if "base64" in attachment:
            content_bytes = base64.b64decode(attachment["base64"])
        elif "path" in attachment:
            with open(attachment["path"], "rb") as f:
                content_bytes = f.read()
        else:
            continue

        # Create LangfuseMedia object
        media_obj = LangfuseMedia(
            content_bytes=content_bytes,
            content_type=content_type
        )

        by_location[attach_to][key] = media_obj

    # Inject into trace_data
    result = trace_data.copy()

    if by_location["input"]:
        if isinstance(result.get("input"), dict):
            result["input"] = {**result["input"], **by_location["input"]}
        else:
            # If input is not a dict, wrap it
            result["input"] = {"_original": result.get("input"), **by_location["input"]}

    if by_location["output"]:
        if isinstance(result.get("output"), dict):
            result["output"] = {**result["output"], **by_location["output"]}
        else:
            result["output"] = {"_original": result.get("output"), **by_location["output"]}

    if by_location["metadata"]:
        if isinstance(result.get("metadata"), dict):
            result["metadata"] = {**result["metadata"], **by_location["metadata"]}
        else:
            result["metadata"] = by_location["metadata"]

    return result