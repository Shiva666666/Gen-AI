from pathlib import Path

import gradio as gr
from PIL import Image
from transformers import pipeline


BASE_DIR = Path(r"C:\Users\licha\OneDrive\Desktop\Gen-AI")
SAMPLE_IMAGE = BASE_DIR / "output" / "playwright" / "lab14_sample_cats.jpg"
DEFAULT_QUESTION = "How many cats are there?"


vqa_pipeline = pipeline(
    "visual-question-answering",
    model="dandelin/vilt-b32-finetuned-vqa",
)


def answer_question(image, question):
    if image is None:
        return "Please provide an image.", [["-", 0.0]]

    question = (question or "").strip()
    if not question:
        return "Please enter a question.", [["-", 0.0]]

    results = vqa_pipeline(image=image, question=question, top_k=3)
    top_answer = results[0]
    answer = f"Predicted answer: {top_answer['answer']} (score: {top_answer['score']:.4f})"
    rows = [[item["answer"], round(float(item["score"]), 4)] for item in results]
    return answer, rows


default_answer, default_rows = answer_question(Image.open(SAMPLE_IMAGE), DEFAULT_QUESTION)


with gr.Blocks(title="Lab 15 - Image Question Answering Assistant") as demo:
    gr.Markdown("# Lab 15 - Image Question Answering Assistant")
    gr.Markdown(
        "A free local course project built with Hugging Face Transformers and Gradio."
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Input Image",
            value=str(SAMPLE_IMAGE),
        )
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a question about the image",
                value=DEFAULT_QUESTION,
            )
            answer_output = gr.Textbox(label="Top Answer", value=default_answer)
            topk_output = gr.Dataframe(
                headers=["Answer", "Score"],
                label="Top 3 Predictions",
                interactive=False,
                row_count=3,
                column_count=(2, "fixed"),
                value=default_rows,
            )
            ask_button = gr.Button("Ask the Model")

    ask_button.click(
        answer_question,
        inputs=[image_input, question_input],
        outputs=[answer_output, topk_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False, inbrowser=False)
