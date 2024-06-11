# Function Calling System Design

To successfully implement function calling with LLM, the structure of the prompt and response is very important.

Unfortunately, there's no guide that we can follow at this moment as these are secrets to openAI. However, after we played around with openAI GTP-4 to ask it reveal system prompt and other stuff, we soon figured out what's the best practice from openAI (assume GPT-4 was telling us the real thing.) So we will follow this "openAI best practice" to design the structure for function calling.

Here's a simplified workflow of function calling based on openAI system design:

- User provide a proper prompt with one or more functions attached as 'tools', the prompt may ask LLM some question that might need to first call one or more of these function tools to get a proper answer before the LLM can generate a response
- The LLM takes the properly formatted prompt (including function metadata in the system prompt), then process and decided whether it should initiate a function calling, the response is typically an json object if need to initiate function calling
- User received the response from LLM, and checks if the response contains json object to initiate function calling, if it does, the user will use the function along with processed parameters to call the function (this happens outside the control of LLM)
- Once user finished calling the function and got the results back, user add the results to the message history, with role 'tool'
- LLM takes in the message contains the results from function calling, and generate an answer to the original question

In general, we need to handle the following three main design challenges when building a LLM capable of function calling:

- Design a robust and controlled way to provide function metadata to the LLM
- Design a proper structure for LLM to initiating function calling
- Design a proper way to provide results to the LLM, once client have made the call the the function and got the results

## Function Metadata

In order for the model to generate function calling, we need to tell the model what functions it can use. This is often referred to as passing the function metadata to the model. Typically, these function metadata are injected into the system prompt, where we can combine then with other system prompt.

For example, we can start with a **Tools** section in the system prompt, and introduce a namespace concept, which could be useful if our model need to support tools other than function.

Here's an typical example of the system prompt for functions adapted from openAI's practice:

```
# Tools

## functions

namespace functions {

function 1...

function 2...

} // namespace functions
```

Next, we need to define a structure for the function metadata. It should at least contains a name, description, and a list of arguments.

For example, this is an simple function for get current weather for a given location, adapted from openAI API documentation https://platform.openai.com/docs/guides/function-calling

```
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

```

One can directly use the raw string of above function metadata and inject into the system prompt. However, the problem with this solution is it takes too much space (in terms of token size) in the prompt. In case we have more complex or multiple functions, the size would grow even further.

To address this issue, we adapt best practice from openAI, where we first serialize the function metadata. This solution achieves an ~45% of token reduction.

For example, for the same get weather function, we can construct the function string as the following, it would only take 51 tokens instead of 96 tokens.

```bash
// Get the current weather in a given location
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
unit?: "celsius" | "fahrenheit",
}) => any;
```

So now the system prompt with the `get_current_weather` function would become:

```
# Tools

## functions

namespace functions {

// Get the current weather in a given location
type get_current_weather = (_: {
// The city and state, e.g. San Francisco, CA
location: string,
unit?: "celsius" | "fahrenheit",
}) => any;

} // namespace functions
```

In practice, we often need to add additional text descriptions to tell the model how and when to utilize these functions. A more comprehensive example is provided inside the `llama3/core/prompt.py` module.

## Function Calling Structure

Now we know how to add function metadata as part of input to the prompt. Next, we will design a structure for the LLM to initiate the function calling. Ideally, structure should be a json object, this is because we need to be able to easily parse the response from LLM for API design. And we might also want to support initiating multiple function callings in a single response from LLM, and an easy to extend to support other tools in the future if required.

After careful review openAI's response, we ended with this json structure:

```
{'tool_uses': [{'recipient_name': 'functions.xxxx', 'parameters': {...}}]}
```

More specifically:

- `tool_uses`: this contain a list of tools to call, each call have the following structure:
  - `recipient_name`: the name of the tool, with namespace as part of the name, in our case the get weather function would be for example `functions.get_current_weather`
  - `parameters`: a dictionary contains the arguments to the tool mentioned in the `recipient_name`.

This design have the following benefit:

- It's an valid json structure with specific key, which is easy to parse for API design
- It's clean and simple, which help reduce token usage
- We can initiate multiple function callings in a single response
- We can support other tools in the future, as we only need to add new namespace

## Tools response

Once the client have received these function calling instructions, it should call the function and provide the results to the LLM.

We decide to use a straightforward design, which closely follow the LLM initiate function calling.

In our case, the tool response is just a list of json objects contains the results from the function calls made on the client side. Similar to the above design, this is clean and simple, and help reduce token usage. It can also handle multiple function calls in a single response.

```

[{'location': 'San Francisco', 'temperature': '72', 'unit': None}, {'location': 'Tokyo', 'temperature': '10', 'unit': None}]

```

In addition to the 'user', 'assistant' roles in a normal chat history, we decided to introduce a new role called 'tool', this represents the results from calling the corresponding tool. For example, the above response can be added to the chat history as this:

```
{'role': 'tool', 'content': [{'location': 'San Francisco', 'temperature': '72', 'unit': None}, {'location': 'Tokyo', 'temperature': '10', 'unit': None}]}
```

## Put it all together

You can run the `analyze_dataset.ipynb` inside `ideas` module to check the datasets. We include some examples here for an overview. Notice here the last turn is supposed to be the training labels.

**Example - Calculate tip**

In this case, when given the system and user prompt, we would expect the LLM to generate the a response close to last chat turn with the 'assistant' role.

```
[
    {
        'role': 'system',
        'content': "\n# Tools\n\n## functions\n\nnamespace functions {\n\n// Calculate the tip amount for a given bill\ntype calculate_tip = (_: {\n// The total bill amount\nbill_amount: number,\n// The tip percentage\ntip_percentage: number,\n}) => any;\n\n} // namespace functions\n\n## multi_tool_use\n\n// This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections. Only tools in the functions namespace are permitted.\n// Ensure that the parameters provided to each tool are valid according to that tool's specification.\nnamespace multi_tool_use {\n\n// Use this function to run multiple tools simultaneously, but only if they can operate in parallel. Do this even if the prompt suggests using the tools sequentially.\ntype parallel = (_: {\n// The tools to be executed in parallel. NOTE: only functions tools are permitted\ntool_uses: {\n// The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.\nrecipient_name: string,\n// The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.\nparameters: object,\n}[],\n}) => any;\n\n} // namespace multi_tool_use\n",
    },
    {'role': 'user', 'content': 'Hi, I need help with calculating a tip. My bill amount is $50 and I want to leave a 20% tip.'},
    {'role': 'assistant', 'content': "{'tool_uses': [{'recipient_name': 'functions.calculate_tip', 'parameters': {'bill_amount': 50, 'tip_percentage': 20}}]}"}
]
```

**Example - Ask followup question**

In some case, the user may forgot provide essential information. So it's LLM's job to ensure it gets all information required before initiate a function calling.

For example, when user asking LLM to check weather, but the user didn't tell the location. In this case, the LLM should ask the user where before initiate a function calling.

```
[
    {
        'role': 'system',
        'content': "\n# Tools\n\n## functions\n\nnamespace functions {\n\n// Search for books based on keywords\ntype search_books = (_: {\n// The keywords to search for in books\nkeywords: array,\n}) => any;\n\n} // namespace functions\n\n## multi_tool_use\n\n// This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections. Only tools in the functions namespace are permitted.\n// Ensure that the parameters provided to each tool are valid according to that tool's specification.\nnamespace multi_tool_use {\n\n// Use this function to run multiple tools simultaneously, but only if they can operate in parallel. Do this even if the prompt suggests using the tools sequentially.\ntype parallel = (_: {\n// The tools to be executed in parallel. NOTE: only functions tools are permitted\ntool_uses: {\n// The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.\nrecipient_name: string,\n// The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.\nparameters: object,\n}[],\n}) => any;\n\n} // namespace multi_tool_use\n",
    },
    {'role': 'user', 'content': 'I am looking for some books to read. Can you help me find some based on my interests?'},
    {'role': 'assistant', 'content': 'Of course! Could you please tell me some keywords related to your interests?'},
]
```

**Example - Answer question based on results from tool**

The LLM should also be able to answer the user question based on the results of the function calls, which is provided to LLM as an additional role 'tool'.

```
[
    {
        'role': 'system',
        'content': "\n# Tools\n\n## functions\n\nnamespace functions {\n\n// Search for books based on keywords\ntype search_books = (_: {\n// The keywords to search for in books\nkeywords: array,\n}) => any;\n\n} // namespace functions\n\n## multi_tool_use\n\n// This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections. Only tools in the functions namespace are permitted.\n// Ensure that the parameters provided to each tool are valid according to that tool's specification.\nnamespace multi_tool_use {\n\n// Use this function to run multiple tools simultaneously, but only if they can operate in parallel. Do this even if the prompt suggests using the tools sequentially.\ntype parallel = (_: {\n// The tools to be executed in parallel. NOTE: only functions tools are permitted\ntool_uses: {\n// The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.\nrecipient_name: string,\n// The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.\nparameters: object,\n}[],\n}) => any;\n\n} // namespace multi_tool_use\n",
    },
    {'role': 'user', 'content': 'I am looking for some books to read. Can you help me find some based on my interests?'},
    {'role': 'assistant', 'content': 'Of course! Could you please tell me some keywords related to your interests?'},
    {'role': 'user', 'content': 'I am interested in history, biographies and science fiction.'},
    {'role': 'assistant', 'content': "{'tool_uses': [{'recipient_name': 'functions.search_books', 'parameters': {'keywords': ['history', 'biographies', 'science fiction']}}]}"},
    {
        'role': 'tool',
        'content': "[{'results': [{'title': 'Sapiens: A Brief History of Humankind', 'author': 'Yuval Noah Harari'}, {'title': 'Elon Musk: Tesla, SpaceX, and the Quest for a Fantastic Future', 'author': 'Ashlee Vance'}, {'title': 'Dune', 'author': 'Frank Herbert'}]}]",
    },
    {
        'role': 'assistant',
        'content': 'Based on your interests, here are some books you might enjoy: \n1. "Sapiens: A Brief History of Humankind" by Yuval Noah Harari\n2. "Elon Musk: Tesla, SpaceX, and the Quest for a Fantastic Future" by Ashlee Vance\n3. "Dune" by Frank Herbert.',
    },
]
```

**Example - answer question outside capabilities**

In special case where the user might ask LLM to do something that is outside of the capabilities (not functions can do that), so the model should be able to state this clearly to the user.

```
[
    {
        'role': 'system',
        'content': "\n# Tools\n\n## functions\n\nnamespace functions {\n\n// Calculate the monthly mortgage payment\ntype calculate_mortgage_payment = (_: {\n// The loan amount\nloan_amount: number,\n// The annual interest rate\ninterest_rate: number,\n// The loan term in years\nloan_term: integer,\n}) => any;\n\n} // namespace functions\n\n## multi_tool_use\n\n// This tool serves as a wrapper for utilizing multiple tools. Each tool that can be used must be specified in the tool sections. Only tools in the functions namespace are permitted.\n// Ensure that the parameters provided to each tool are valid according to that tool's specification.\nnamespace multi_tool_use {\n\n// Use this function to run multiple tools simultaneously, but only if they can operate in parallel. Do this even if the prompt suggests using the tools sequentially.\ntype parallel = (_: {\n// The tools to be executed in parallel. NOTE: only functions tools are permitted\ntool_uses: {\n// The name of the tool to use. The format should either be just the name of the tool, or in the format namespace.function_name for plugin and function tools.\nrecipient_name: string,\n// The parameters to pass to the tool. Ensure these are valid according to the tool's own specifications.\nparameters: object,\n}[],\n}) => any;\n\n} // namespace multi_tool_use\n",
    },
    {'role': 'user', 'content': 'Can you please book a flight for me from New York to London?'},
    {'role': 'assistant', 'content': "I'm sorry, but I'm unable to assist with that. My current capabilities are limited to calculating mortgage payments. I don't have the functionality to book flights."},
]

```
