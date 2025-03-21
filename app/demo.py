import gradio as gr


def generate_response(
    user_message, history, model, harmfulness, descriptiveness, politeness
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


with gr.Blocks() as demo:
    # State to maintain history history
    history = gr.State([])

    # Layout with two columns: chat on the left, controls on the right.
    with gr.Row():
        # Left column takes approximately 90% width
        with gr.Column(scale=9):
            chat_component = gr.Chatbot(label="Chat history", type="messages")
            user_input = gr.Textbox(
                placeholder="Enter your message here...", label="Your Message"
            )
            send_button = gr.Button("Send")

        # Right column takes approximately 10% width and holds three sliders.
        with gr.Column(scale=1):
            # model choice
            model_selector = gr.Dropdown(
                choices=["Qwen/Qwen-1_8B-chat"],
                label="Select Model",
                value="Qwen/Qwen-1_8B-chat",
            )

            # concept selectors
            harmfulness_slider = gr.Slider(1, 5, step=1, value=3, label="Harmfulness")
            descriptiveness_slider = gr.Slider(
                1, 5, step=1, value=3, label="Descriptiveness"
            )
            politeness_slider = gr.Slider(1, 5, step=1, value=3, label="Politeness")

    # Define the button click action to update the chat and pass slider values
    send_button.click(
        fn=generate_response,
        inputs=[
            user_input,
            history,
            model_selector,
            # concepts
            harmfulness_slider,
            descriptiveness_slider,
            politeness_slider,
        ],
        outputs=[chat_component, user_input],
    )

demo.launch()
