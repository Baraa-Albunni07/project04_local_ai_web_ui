import gradio as gr
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="lokal-kein-key-noetig",
)


def chat_with_local_llm(message: str, history: list, system_prompt: str, model_name: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    for entry in history:
        if isinstance(entry, dict):
            messages.append({"role": entry["role"], "content": entry["content"]})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, bot_msg = entry
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Verbindungsfehler zum lokalen Modell-Server ({e}). Läuft Ollama/LM Studio/Lemonade?"


demo = gr.ChatInterface(
    fn=chat_with_local_llm,
    title="Local AI Web UI",
    description="Lokale Chat-Oberfläche für OpenAI-kompatible Modell-Server (Ollama, LM Studio, AMD Lemonade).",
    additional_inputs=[
        gr.Textbox(
            value="Du bist ein hilfreicher, präziser lokaler KI-Assistent.",
            label="System Prompt",
        ),
        gr.Textbox(
            value="llama3.2",
            label="Modellname",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()