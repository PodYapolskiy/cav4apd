import gradio as gr


def generate_response(
    user_message: str,
    history: list,
    model: str,
    harmfulness: int,
    descriptiveness: int,
    politeness: int,
):
    """
    Appends the user message to the chat history and generates a dummy model response.
    The model response reflects the current concept scale values.
    """
    # Append user's message
    history.append(gr.ChatMessage(role="user", content=user_message))

    # Here you could integrate your LLM experiment logic that uses the concept directions.
    # For demonstration, we include the concept slider values in the response.
    response = (
        f"Model {model} response (with experiment params):\n"
        f"Harmfulness={harmfulness}\n"
        f"Descriptiveness={descriptiveness}\n"
        f"Politeness={politeness}"
    )

    # Append model's response to the chat history
    history.append(gr.ChatMessage(role="assistant", content=response))

    return history, ""
