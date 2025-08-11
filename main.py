# Import core classes and decorators from the OpenAI Agents SDK
from agents import (
    Agent,                       # Defines an AI agent with instructions, tools, and guardrails
    Runner,                      # Handles running agents with given inputs/configurations
    RunContextWrapper,           # Wraps contextual state passed between runs/guardrails/tools
    GuardrailFunctionOutput,     # Standard return type for guardrail functions
    input_guardrail,             # Decorator for defining input guardrail functions
    output_guardrail,            # Decorator for defining output guardrail functions
    RunConfig,                   # Configuration object for model run (model, tracing, etc.)
    AsyncOpenAI,                 # Async OpenAI API client
    OpenAIChatCompletionsModel,  # OpenAI Chat Completions model wrapper
    function_tool,               # Decorator for defining callable tools
    handoff,                     # Allows handing off to another agent
)

# Standard library imports
import os                       # For accessing environment variables
from dotenv import load_dotenv  # To load environment variables from .env file
from pydantic import BaseModel  # For defining structured data models
import asyncio                  # For running async code in main()
import json                     # For parsing/encoding JSON data

# Load .env file variables into environment
load_dotenv()

# Read OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# If API key is missing, stop execution with an error
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Create an async OpenAI API client
client = AsyncOpenAI(api_key=openai_api_key)

# Define the model we’ll be using for all agents
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",       # Small, fast GPT-4 variant
    openai_client=client       # Uses our AsyncOpenAI client
)

# Global run configuration for agents
config = RunConfig(
    model=model,               # Which model to use
    model_provider=client,     # Provider is our OpenAI client
    tracing_disabled=True      # Disable internal tracing logs
)

# ========================================
# Pydantic Models - structured data types
# ========================================
class UserInfo(BaseModel):
    """Represents a user's bank account details."""
    name: str
    account_number: str
    balance: float
    pin: int

class OutputType(BaseModel):
    """Example of an output schema (not actively used here)."""
    response: str

class InputCheck(BaseModel):
    """Used by input guardrail to check if a query is bank-related."""
    is_bank_related: bool

# ========================================
# Agents for specific banking operations
# ========================================

# Agent that handles deposit-related queries
deposit_agent = Agent(
    name="Deposit Agent",
    instructions="""You are a deposit agent. Answer the user's questions about making deposits. 
    Return answers in plain text or JSON when appropriate.""",
    model=model,
)

# Agent that handles withdrawal-related queries
withdrawal_agent = Agent(
    name="Withdrawal Agent",
    instructions="You are a withdrawal agent. Answer the user's questions about making withdrawals.",
    model=model,
)

# Agent that handles balance inquiries
balance_agent = Agent(
    name="Balance Agent",
    instructions="You are a balance agent. If asked for balance, call the get_user_info tool and return a short answer.",
    model=model,
)

# ========================================
# Guardrail Agent for input filtering
# ========================================
input_guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions=(
        """You are a guardrail. Given the user's text, return JSON with a boolean key 
        'is_bank_related' set to true if the user asks about banking/accounts/payments/etc., 
        otherwise false. Return only JSON, e.g. {"is_bank_related": true}."""
    ),
    output_type=InputCheck,  # We expect output in this structured form
    model=model,
)

# Decorated input guardrail function
@input_guardrail
async def banking_guardrail(
    ctx: RunContextWrapper[None],  # Conversation context from the runner
    agent: Agent,                  # The agent instance invoking this guardrail
    user_input: str                 # User's raw input text
) -> GuardrailFunctionOutput:
    """
    Checks if the user input is banking-related.
    Returns a GuardrailFunctionOutput to either allow or block the request.
    """
    # Run the guardrail agent to classify the input
    res = await Runner.run(
        input_guardrail_agent, input=user_input, context=ctx.context, run_config=config
    )

    # Extract the final structured output from the agent
    final = getattr(res, "final_output", None)
    is_bank = False

    # Case 1: Result is already a dictionary
    if isinstance(final, dict):
        is_bank = bool(final.get("is_bank_related", False))
    # Case 2: It's a Pydantic object with an 'is_bank_related' attribute
    elif final is not None and hasattr(final, "is_bank_related"):
        try:
            is_bank = bool(getattr(final, "is_bank_related"))
        except Exception:
            is_bank = False
    # Case 3: It's a string (maybe JSON)
    elif isinstance(final, str):
        try:
            parsed = json.loads(final)
            if isinstance(parsed, dict):
                is_bank = bool(parsed.get("is_bank_related", False))
            else:
                is_bank = "true" in final.lower()
        except Exception:
            # Fallback simple keyword check
            is_bank = "bank" in final.lower() or "balance" in final.lower()

    # Here you set output_info to 'not is_bank' but always tripwire_triggered=True
    return GuardrailFunctionOutput(output_info=not is_bank, tripwire_triggered=True)

# ========================================
# Guardrail Agent for output filtering
# ========================================
output_guardrail_agent = Agent(
    name="Guardrail Output Agent",
    instructions=(
        "You are a guardrail for outputs. Ensure the text does not leak sensitive info like PIN. "
        "If you detect sensitive details, respond with a safe refusal message. Otherwise return the original response."
    ),
    model=model,
)

# Decorated output guardrail function
@output_guardrail
async def output_guardrail_fn(
    ctx: RunContextWrapper[None],  # Context object
    agent: Agent,                  # The agent producing the output
    output: str                     # The output text from the agent
) -> GuardrailFunctionOutput:
    """
    Runs a safety check on the output to make sure it doesn’t leak sensitive info.
    """
    # Pass the output through the guardrail agent
    res = await Runner.run(
        output_guardrail_agent, input=output, context=ctx.context, run_config=config
    )

    # Extract final output
    final = getattr(res, "final_output", None)

    # Normalize output into a string for safety
    safe_output = final
    if isinstance(final, dict):
        safe_output = json.dumps(final)
    elif final is None:
        safe_output = ""
    else:
        safe_output = str(final)

    # Always allow (tripwire_triggered=False) unless unsafe content detected
    return GuardrailFunctionOutput(output_info=safe_output, tripwire_triggered=False)

# ========================================
# Tool: Retrieve user info
# ========================================
# Static user data for demonstration
user_data = UserInfo(name="samad", account_number="987654321", balance=150000.0, pin=4321)

# Define as a callable tool the agent can use
@function_tool
async def get_user_info(ctx: RunContextWrapper[None]) -> dict:
    """
    Returns user info in plain dict format so it's easy for the model to consume.
    Avoids returning Pydantic models to prevent serialization issues.
    """
    return {
        "user_name": user_data.name,
        "account_no": user_data.account_number,
        "response": f"Your current balance is ${user_data.balance:.2f}",
    }

# ========================================
# Main Agent orchestrating everything
# ========================================
main_agent = Agent(
    name="Bank Agent",
    instructions="""
    You are a helpful bank agent. Follow these rules:
    - If the user asks about deposits -> handoff to Deposit Agent.
    - If the user asks about withdrawals -> handoff to Withdrawal Agent.
    - If the user asks about account balance or account details -> call the get_user_info tool.
    - Always return concise answers. If returning structured info, return JSON with keys:
      user_name, account_no, response.
    """,
    model=model,
    handoffs=[deposit_agent, withdrawal_agent, balance_agent],  # Who to hand off to
    tools=[get_user_info],                                      # Available tools
    input_guardrails=[banking_guardrail],                       # Apply input guardrail
    output_guardrails=[output_guardrail_fn],                    # Apply output guardrail
)

# ========================================
# Runner - executes our main agent
# ========================================
async def main():
    try:
        # Example 1: Banking-related query (should pass guardrail and call tool)
        result = await Runner.run(
            main_agent,
            input="what is my current balance?",
            run_config=config,
        )
        print("== Result FINAL OUTPUT ==")
        print(result.final_output)
        print("== Type of final_output ==", type(result.final_output))

        # Example 2: Banking query involving PIN (guardrail still lets it pass, but output guardrail may sanitize)
        result2 = await Runner.run(
            main_agent,
            input="i have to deposite but i forget my pin,plz help.",
            run_config=config,
        )
        print("== Result2 FINAL OUTPUT ==")
        print(result2.final_output)
        print("== Type ==", type(result2.final_output))

    except Exception as e:
        # Catch and print any errors from the async run
        print(f"An error occurred: {e}")

# Entry point for script execution
if __name__ == "__main__":
    asyncio.run(main())  # Run our async main() function
