import json
import re
import subprocess
from pathlib import Path

import nbformat as nbf
import ollama
import requests
from agno.agent import Agent
from agno.models.ollama import Ollama
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


BASE_DIR = Path(r"C:\Users\licha\OneDrive\Desktop\Gen-AI")
NOTEBOOK_DIR = BASE_DIR / "output" / "jupyter-notebook"
LAB12_DIR = BASE_DIR / "output" / "lab12_notebook"
LAB13_DIR = BASE_DIR / "output" / "lab13_notebook"
LAB14_DIR = BASE_DIR / "output" / "lab14_notebook"
PLAYWRIGHT_DIR = BASE_DIR / "output" / "playwright"
TMP_DIR = BASE_DIR / "tmp"

LAB12_NOTEBOOK = NOTEBOOK_DIR / "lab_12_agno_agent_428.ipynb"
LAB13_NOTEBOOK = NOTEBOOK_DIR / "lab_13_code_bug_detection_428.ipynb"
LAB14_NOTEBOOK = NOTEBOOK_DIR / "lab_14_vqa_428.ipynb"
SAMPLE_IMAGE_PATH = PLAYWRIGHT_DIR / "lab14_sample_cats.jpg"


def ensure_dirs():
    for path in [NOTEBOOK_DIR, LAB12_DIR, LAB13_DIR, LAB14_DIR, PLAYWRIGHT_DIR, TMP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def notebook_metadata():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }


def save_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata=notebook_metadata())
    path.write_text(nbf.writes(nb), encoding="utf-8")


def stream_output(text: str):
    return nbf.v4.new_output("stream", name="stdout", text=text)


def pick_font(size: int, mono: bool = False):
    if mono:
        candidates = [
            Path(r"C:\Windows\Fonts\consola.ttf"),
            Path(r"C:\Windows\Fonts\cour.ttf"),
        ]
    else:
        candidates = [
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\calibri.ttf"),
        ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            width = draw.textbbox((0, 0), trial, font=font)[2]
            if width <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def create_output_image(title: str, body: str, out_path: Path):
    width = 1500
    margin = 70
    line_gap = 10
    title_font = pick_font(34, mono=False)
    body_font = pick_font(23, mono=True)
    dummy = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(dummy)
    lines = wrap_text(draw, body, body_font, width - (2 * margin))
    line_height = draw.textbbox((0, 0), "Ag", font=body_font)[3] + line_gap
    title_height = draw.textbbox((0, 0), title, font=title_font)[3]
    height = margin * 2 + title_height + 40 + max(1, len(lines)) * line_height + 40

    image = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (25, 25, width - 25, height - 25),
        radius=24,
        fill="white",
        outline="#cfd8e3",
        width=3,
    )
    draw.text((margin, margin), title, font=title_font, fill="#17324d")
    y = margin + title_height + 40
    for line in lines:
        draw.text((margin, y), line, font=body_font, fill="#1f2937")
        y += line_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def extract_code_block(text: str):
    match = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def download_sample_image():
    if SAMPLE_IMAGE_PATH.exists():
        return SAMPLE_IMAGE_PATH

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    SAMPLE_IMAGE_PATH.write_bytes(response.content)
    return SAMPLE_IMAGE_PATH


def run_lab12():
    agent = Agent(
        model=Ollama(id="qwen2.5:3b"),
        name="Campus Study Assistant",
        role="Helpful assistant for Gen AI lab learners",
        description="A study-focused agent that explains concepts clearly and briefly.",
        instructions=[
            "Answer for an engineering student preparing lab records and viva questions.",
            "Keep the tone friendly and practical.",
            "Use short bullet points when listing steps or examples.",
        ],
        markdown=True,
    )

    prompts = [
        "Explain what an LLM agent is in simple terms for a lab record.",
        "Give three practical uses of AI agents in education.",
        "Provide a four-step plan to prepare for a Gen AI viva tomorrow.",
    ]

    records = []
    for prompt in prompts:
        response = agent.run(prompt)
        text = response.content.strip()
        records.append({"prompt": prompt, "response": text})

    images = []
    for idx, record in enumerate(records, start=1):
        out_path = LAB12_DIR / f"lab12_agent_response_{idx}.png"
        images.append(
            str(
                create_output_image(
                    f"Lab 12 Output {idx}: Agno Agent Response",
                    f"Prompt: {record['prompt']}\n\nResponse:\n{record['response']}",
                    out_path,
                )
            )
        )

    cells = [
        nbf.v4.new_markdown_cell(
            "# Lab 12 - Building a Simple LLM Agent Using the Phi Data Framework\n"
            "This notebook uses Agno, the current successor to Phi Data, with Ollama and the `qwen2.5:3b` model."
        ),
        nbf.v4.new_code_cell(
            source=(
                "from agno.agent import Agent\n"
                "from agno.models.ollama import Ollama\n\n"
                "agent = Agent(\n"
                "    model=Ollama(id='qwen2.5:3b'),\n"
                "    name='Campus Study Assistant',\n"
                "    role='Helpful assistant for Gen AI lab learners',\n"
                "    description='A study-focused agent that explains concepts clearly and briefly.',\n"
                "    instructions=[\n"
                "        'Answer for an engineering student preparing lab records and viva questions.',\n"
                "        'Keep the tone friendly and practical.',\n"
                "        'Use short bullet points when listing steps or examples.',\n"
                "    ],\n"
                "    markdown=True,\n"
                ")\n"
                "print('Agno agent initialized with qwen2.5:3b')"
            ),
            execution_count=1,
            outputs=[stream_output("Agno agent initialized with qwen2.5:3b\n")],
        ),
    ]

    for count, record in enumerate(records, start=2):
        text = f"Prompt: {record['prompt']}\n\n{record['response']}\n"
        cells.append(
            nbf.v4.new_code_cell(
                source=(
                    f"prompt = {record['prompt']!r}\n"
                    "response = agent.run(prompt)\n"
                    "print(response.content)"
                ),
                execution_count=count,
                outputs=[stream_output(text)],
            )
        )

    save_notebook(LAB12_NOTEBOOK, cells)

    result_path = LAB12_DIR / "lab12_results.json"
    result_path.write_text(
        json.dumps(
            {
                "framework": "Agno (successor to Phi Data)",
                "model": "qwen2.5:3b",
                "notebook": str(LAB12_NOTEBOOK),
                "records": records,
                "images": images,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_lab13():
    generation_prompt = (
        "Write a concise Python program for a StudentRecordManager class.\n"
        "Requirements:\n"
        "- add_student(name, marks)\n"
        "- class_average()\n"
        "- passed_students() for marks >= 40\n"
        "- include a short sample usage block\n"
        "Return only Python code inside one fenced python block."
    )
    generated_text = ollama.chat(
        model="qwen2.5-coder:3b",
        messages=[{"role": "user", "content": generation_prompt}],
    )["message"]["content"].strip()

    buggy_code = (
        "def average_marks(marks):\n"
        "    total = 0\n"
        "    for i in range(len(marks) + 1):\n"
        "        total += marks[i]\n"
        "    return total / len(marks)\n\n"
        "print(average_marks([10, 20, 30]))"
    )
    analysis_prompt = (
        "You are reviewing Python code.\n"
        "Find the bug in this program, explain why it fails, and then provide a corrected version.\n\n"
        f"{buggy_code}\n\n"
        "Use this format:\n"
        "Bug:\nWhy:\nFix:\n```python\n# corrected code\n```"
    )
    analysis_text = ollama.chat(
        model="qwen2.5-coder:3b",
        messages=[{"role": "user", "content": analysis_prompt}],
    )["message"]["content"].strip()

    fix_prompt = (
        "Return only the corrected Python code for this buggy program in one fenced python block.\n\n"
        f"{buggy_code}"
    )
    fixed_text = ollama.chat(
        model="qwen2.5-coder:3b",
        messages=[{"role": "user", "content": fix_prompt}],
    )["message"]["content"].strip()

    fixed_code = extract_code_block(fixed_text)
    fixed_script = TMP_DIR / "lab13_fixed_average.py"
    fixed_script.write_text(fixed_code + "\n", encoding="utf-8")

    run_result = subprocess.run(
        ["python", str(fixed_script)],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        timeout=60,
        check=True,
    )
    verification_output = run_result.stdout.strip()

    image_paths = [
        str(
            create_output_image(
                "Lab 13 Output 1: Code Generation",
                generated_text,
                LAB13_DIR / "lab13_generated_code.png",
            )
        ),
        str(
            create_output_image(
                "Lab 13 Output 2: Bug Detection and Fix",
                analysis_text,
                LAB13_DIR / "lab13_bug_analysis.png",
            )
        ),
        str(
            create_output_image(
                "Lab 13 Output 3: Corrected Code Verification",
                f"Corrected code:\n{fixed_code}\n\nProgram output:\n{verification_output}",
                LAB13_DIR / "lab13_fixed_output.png",
            )
        ),
    ]

    cells = [
        nbf.v4.new_markdown_cell(
            "# Lab 13 - Using LLMs for Code Generation and Bug Detection in Software Development\n"
            "This notebook uses Ollama with the `qwen2.5-coder:3b` model."
        ),
        nbf.v4.new_code_cell(
            source=(
                "import ollama\n"
                "print('Using local model: qwen2.5-coder:3b')"
            ),
            execution_count=1,
            outputs=[stream_output("Using local model: qwen2.5-coder:3b\n")],
        ),
        nbf.v4.new_code_cell(
            source=(
                f"generation_prompt = {generation_prompt!r}\n"
                "response = ollama.chat(model='qwen2.5-coder:3b', messages=[{'role': 'user', 'content': generation_prompt}])\n"
                "print(response['message']['content'])"
            ),
            execution_count=2,
            outputs=[stream_output(generated_text + "\n")],
        ),
        nbf.v4.new_code_cell(
            source=(
                f"buggy_code = {buggy_code!r}\n"
                f"analysis_prompt = {analysis_prompt!r}\n"
                "response = ollama.chat(model='qwen2.5-coder:3b', messages=[{'role': 'user', 'content': analysis_prompt}])\n"
                "print(response['message']['content'])"
            ),
            execution_count=3,
            outputs=[stream_output(analysis_text + "\n")],
        ),
        nbf.v4.new_code_cell(
            source=(
                f"fixed_code = {fixed_code!r}\n"
                "print(fixed_code)\n"
                "print('\\nProgram output:')\n"
                f"print({verification_output!r})"
            ),
            execution_count=4,
            outputs=[stream_output(f"{fixed_code}\n\nProgram output:\n{verification_output}\n")],
        ),
    ]
    save_notebook(LAB13_NOTEBOOK, cells)

    result_path = LAB13_DIR / "lab13_results.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "qwen2.5-coder:3b",
                "notebook": str(LAB13_NOTEBOOK),
                "generation_prompt": generation_prompt,
                "generated_text": generated_text,
                "buggy_code": buggy_code,
                "analysis_text": analysis_text,
                "fixed_code": fixed_code,
                "verification_output": verification_output,
                "images": image_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_lab14():
    image_path = download_sample_image()
    image = Image.open(image_path)
    vqa_pipe = pipeline(
        "visual-question-answering",
        model="dandelin/vilt-b32-finetuned-vqa",
    )

    questions = [
        "How many cats are there?",
        "What are the cats sitting on?",
        "What color is the couch?",
    ]

    answers = []
    for question in questions:
        result = vqa_pipe(image=image, question=question, top_k=3)
        top_answer = result[0]
        answers.append(
            {
                "question": question,
                "answer": top_answer["answer"],
                "score": round(float(top_answer["score"]), 4),
                "top_k": result,
            }
        )

    summary_lines = [f"Image: {image_path.name}", ""]
    for idx, item in enumerate(answers, start=1):
        summary_lines.append(f"Q{idx}: {item['question']}")
        summary_lines.append(f"Answer: {item['answer']} (score: {item['score']})")
        summary_lines.append("")

    result_image = create_output_image(
        "Lab 14 Output: Visual Question Answering Results",
        "\n".join(summary_lines).strip(),
        LAB14_DIR / "lab14_vqa_results.png",
    )

    cells = [
        nbf.v4.new_markdown_cell(
            "# Lab 14 - Using Multimodal LLMs for Visual Question Answering\n"
            "This notebook uses Hugging Face Transformers with the `dandelin/vilt-b32-finetuned-vqa` model."
        ),
        nbf.v4.new_code_cell(
            source=(
                "from transformers import pipeline\n"
                "from PIL import Image\n\n"
                f"image = Image.open(r'{image_path}')\n"
                "vqa_pipe = pipeline('visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa')\n"
                "print('VQA model loaded successfully')"
            ),
            execution_count=1,
            outputs=[stream_output("VQA model loaded successfully\n")],
        ),
        nbf.v4.new_code_cell(
            source=(
                "questions = [\n"
                "    'How many cats are there?',\n"
                "    'What are the cats sitting on?',\n"
                "    'What color is the couch?',\n"
                "]\n"
                "for question in questions:\n"
                "    result = vqa_pipe(image=image, question=question, top_k=3)\n"
                "    print(question)\n"
                "    print(result[0])\n"
                "    print()"
            ),
            execution_count=2,
            outputs=[
                stream_output(
                    "\n".join(
                        [
                            f"{item['question']}\n{{'answer': '{item['answer']}', 'score': {item['score']}}}\n"
                            for item in answers
                        ]
                    )
                    + "\n"
                )
            ],
        ),
    ]
    save_notebook(LAB14_NOTEBOOK, cells)

    result_path = LAB14_DIR / "lab14_results.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "dandelin/vilt-b32-finetuned-vqa",
                "notebook": str(LAB14_NOTEBOOK),
                "image_path": str(image_path),
                "answers": answers,
                "result_image": str(result_image),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    ensure_dirs()
    download_sample_image()
    run_lab12()
    run_lab13()
    run_lab14()
    print("Artifacts created for labs 12 to 14.")


if __name__ == "__main__":
    main()
