"""Test StreamNode with WorkflowEngine"""
import asyncio
import uuid
import logging
from hush.core import WorkflowEngine, StreamNode, CodeNode, START, END, INPUT, OUTPUT, STREAM_SERVICE

# Enable logging (set to WARNING to reduce noise)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Only show StreamNode logs
logging.getLogger('hush.core.nodes.flow.stream_node').setLevel(logging.INFO)


def process_chunk(content: str, timestamp: float) -> dict:
    """Process a single chunk

    Returns:
        processed_text (str): Processed chunk text
        out_timestamp (float): Original timestamp
    """
    processed = f"Processed: {content}"
    return {
        "processed_text": processed,
        "out_timestamp": timestamp
    }


async def stream_producer(session_id: str, request_id: str, channel: str, data: list):
    """Simulate streaming data to STREAM_SERVICE"""
    for item in data:
        # Push dict data to STREAM_SERVICE
        chunk_data = {"content": item, "timestamp": asyncio.get_event_loop().time()}
        await STREAM_SERVICE.push(request_id, channel, chunk_data, session_id=session_id)
        await asyncio.sleep(0.01)
    await STREAM_SERVICE.end(request_id, channel, session_id=session_id)


async def main():
    """StreamNode usage example with WorkflowEngine"""

    # Define workflow with StreamNode
    with WorkflowEngine(
        name="stream-workflow",
        description="Real-time streaming data processing workflow"
    ) as workflow:

        # StreamNode with inner graph
        with StreamNode(
            name="stream_processor",
            description="Process chunks as they stream",
            inputs={
                "input_channel": INPUT,
            },
            outputs={
                "output_channel": "output_stream",
            }
        ) as stream_node:
            processor = CodeNode(
                name="process",
                code_fn=process_chunk,
                inputs=INPUT,
                outputs=OUTPUT
            )
            START >> processor >> END

        START >> stream_node >> END

    workflow.compile()

    # Generate IDs
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    # Start streaming data in background
    test_data = ["Hello", "Stream", "Processing", "World"]
    producer = asyncio.create_task(
        stream_producer(session_id, request_id, "input_stream", test_data)
    )

    # Run workflow (will consume stream)
    result = await workflow.run(
        inputs={"input_channel": "input_stream"},
        user_id=user_id,
        session_id=session_id,
        request_id=request_id
    )

    await producer

    print(f"\n{'='*60}")
    print(f"Workflow completed successfully!")
    print(f"{'='*60}")
    print(f"Request ID: {request_id}")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
