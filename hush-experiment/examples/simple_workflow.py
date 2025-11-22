"""Simplest possible workflow example.

This demonstrates the absolute basics of creating and running a workflow
without needing external services or API keys.
"""

import asyncio
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
from hush.core.nodes import code_node


@code_node
def greet_user(name: str):
    """Generate a greeting message."""
    greeting = f"Hello, {name}! Welcome to Hush workflows."
    return {"greeting": greeting}


@code_node
def to_uppercase(text: str):
    """Convert text to uppercase."""
    result = text.upper()
    return {"result": result}


async def main():
    print("="*60)
    print("Simple Workflow Example")
    print("="*60)

    # Create a simple workflow with code nodes (no external dependencies)
    with WorkflowEngine(name="simple_workflow") as workflow:
        # Node 1: Generate greeting
        greet = greet_user(
            name="greet",
            inputs={"name": INPUT}
        )

        # Node 2: Convert to uppercase
        uppercase = to_uppercase(
            name="uppercase",
            inputs={"text": greet["greeting"]}
        )

        # Define flow: START -> greet -> uppercase -> END
        START >> greet >> uppercase >> END

    # Compile the workflow
    workflow.compile()
    print("\n✓ Workflow compiled successfully")

    # Run the workflow
    print("\nRunning workflow with input: {'name': 'Hush User'}")
    result = await workflow.run(inputs={"name": "Hush User"})

    # Display results
    print("\n" + "-"*60)
    print("Workflow Results:")
    print("-"*60)
    if result:
        for key, value in result.items():
            print(f"  {key}: {value}")
    print("\n" + "="*60)
    print("✓ Workflow completed successfully!")
    print("="*60)

    return result


if __name__ == "__main__":
    asyncio.run(main())
