"""Trace models for observability.

Based on legacy beegen trace models but updated for hush-core.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional, Union

from hush.core.utils.yaml_model import YamlModel
from pydantic import ConfigDict, field_serializer, field_validator


class BaseTraceInfo(YamlModel):
    """Base class for trace information in workflow and system monitoring.

    This class provides a common structure for capturing execution traces,
    including timing information, inputs/outputs, and metadata.

    Attributes:
        message_id: Unique identifier for the associated message
        message_data: Raw data associated with the message
        inputs: Input data for the traced operation (str, dict, or list)
        outputs: Output data from the traced operation (str, dict, or list)
        start_time: Timestamp when the operation began
        end_time: Timestamp when the operation completed
        metadata: Additional key-value pairs for custom trace information
        trace_id: Unique identifier for this trace instance
    """
    message_id: Optional[str] = None
    message_data: Optional[Any] = None
    inputs: Optional[Union[str, dict[str, Any], list]] = None
    outputs: Optional[Union[str, dict[str, Any], list]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: dict[str, Any] = {}
    trace_id: Optional[str] = None

    @field_validator("inputs", "outputs")
    @classmethod
    def ensure_type(cls, v):
        """Validate inputs and outputs are of supported types."""
        if v is None:
            return None
        if isinstance(v, str | dict | list):
            return v
        return ""

    model_config = ConfigDict(protected_namespaces=())

    @field_serializer("start_time", "end_time")
    def serialize_datetime(self, dt: datetime | None) -> str | None:
        """Serialize datetime fields to ISO format strings."""
        if dt is None:
            return None
        return dt.isoformat()


class WorkflowTraceInfo(BaseTraceInfo):
    """Extended trace information for workflow execution monitoring.

    Captures comprehensive information about workflow runs including
    performance metrics, execution status, and workflow-specific identifiers.

    Attributes:
        workflow_data: Raw workflow configuration or execution data
        conversation_id: Identifier linking to a conversation context
        workflow_app_log_id: Application-level log identifier
        workflow_id: Unique identifier for the workflow definition
        tenant_id: Identifier for the tenant/organization
        workflow_run_id: Unique identifier for this execution instance
        workflow_run_elapsed_time: Total execution time in seconds
        workflow_run_status: Current status (completed, failed, running)
        workflow_run_inputs: Input parameters provided to the workflow
        workflow_run_outputs: Output values produced by the workflow
        workflow_run_version: Version identifier of the workflow
        error: Error message if execution failed
        total_tokens: Total tokens consumed during execution
        file_list: List of files processed or generated
        query: Original query that initiated the workflow
        metadata: Additional workflow-specific metadata
    """
    workflow_data: Any
    conversation_id: Optional[str] = None
    workflow_app_log_id: Optional[str] = None
    workflow_id: str
    tenant_id: str
    workflow_run_id: str
    workflow_run_elapsed_time: Union[int, float]
    workflow_run_status: str
    workflow_run_inputs: Mapping[str, Any]
    workflow_run_outputs: Mapping[str, Any]
    workflow_run_version: str
    error: Optional[str] = None
    total_tokens: int = 0
    file_list: list[str] = []
    query: str = ""
    metadata: dict[str, Any] = {}


class MessageTraceInfo(BaseTraceInfo):
    """Trace information for individual message processing.

    Captures detailed information about message-level operations within
    conversations, including token usage, model information, and file handling.

    Attributes:
        conversation_model: AI model used (e.g., "gpt-4", "claude-3")
        message_tokens: Number of tokens in the input message
        answer_tokens: Number of tokens generated in response
        total_tokens: Total token count for the exchange
        error: Error message if processing failed
        file_list: Files associated with this message
        message_file_data: Raw file data or metadata
        conversation_mode: Mode or type of conversation (chat, completion)
    """
    conversation_model: str
    message_tokens: int = 0
    answer_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    file_list: Optional[Union[str, dict[str, Any], list]] = None
    message_file_data: Optional[Any] = None
    conversation_mode: str = "chat"
    trace_id: Optional[str] = None
