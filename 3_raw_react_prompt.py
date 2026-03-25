from dotenv import load_dotenv
load_dotenv()

import re
import inspect
import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect
# So we pass the tool and get back tool description as string to pass to llm

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        #Get the actual function
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        #Get metadata of the function : function name , arg, type, return value
        signature = inspect.signature(original_function)
        # Get the doc string
        docstring = inspect.getdoc(tool_function) or ""
        #finally concat all of that to make a string
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tools_descriptions = get_tool_descriptions(tools)
## Tools description look like this :
# 'get_product_price(product: str) -> float - Look up the price of a product in the catalog.
# apply_discount(price: float, discount_tier: str) -> float - Apply a discount tier to a price and return the final price.
# Available tiers: bronze, silver, gold.'
tool_names = ", ".join(tools.keys())

## this prompt is from : https://smith.langchain.com/hub/hwchase17/react
react_prompt = f"""Answer the following questions as best you can. You have access to the following tools:

{tools_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""



# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

# --- Agent Loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    prompt = react_prompt.format(question=question) #this will inject the question into the react_prompt
    scratchpad = ""


    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        # Stop token prevents the LLM from generating its own Observation —
        # we inject the real tool result instead.
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0}, #tell llm to stop when they receive Observation token
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
    print(f"  [Parsing] Looking for Action and Action Input in LLM output...")
    action_match = re.search(r"Action:\s*(.+)", output)
    action_input_match = re.search(r"Action Input:\s*(.+)", output)

    if not action_match or not action_input_match:
        print(f"  [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
        break

    tool_name = action_match.group(1).strip()
    tool_input_raw = action_input_match.group(1).strip()
    print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

    # Split comma-separated args; strip key= prefix if LLM outputs key=value format
    raw_args = [x.strip() for x in tool_input_raw.split(",")]
    args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

    print(f"  [Tool Executing] {tool_name}({args})...")
    if tool_name not in tools:
        observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
    else:
        observation = str(tools[tool_name](*args))
    print(f"  [Tool Result] {observation}")

    # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
    scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")