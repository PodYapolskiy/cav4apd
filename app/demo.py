import gradio as gr
from llm import generate_response

with gr.Blocks() as demo:
    # State to maintain history history
    history = gr.State([])

    # Layout with two columns: chat on the left, controls on the right.
    with gr.Row():
        # Left column takes approximately 90% width
        with gr.Column(scale=9):
            chatbot = gr.Chatbot(
                label="Chat history",
                type="messages",
                show_copy_button=True,
            )
            user_input = gr.Textbox(
                placeholder="Enter your message here...", label="Your Message"
            )
            send_button = gr.Button("Send")

        # Right column takes approximately 10% width and holds three sliders.
        with gr.Column(scale=1):
            # model choice
            model_choices = [
                "google/gemma-2b-it",
                "Qwen/Qwen-1_8B-chat",
            ]
            model_selector = gr.Dropdown(
                choices=model_choices,
                label="Select Model",
                value=model_choices[0],
            )

            # concept selectors
            harmfulness_slider = gr.Slider(1, 5, step=1, value=3, label="Harmfulness")
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
            politeness_slider,
        ],
        outputs=[chatbot, user_input],
    )

    # clear histroy state after chatbot's bin click
    chatbot.clear(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[history, user_input],
    )

demo.launch()
