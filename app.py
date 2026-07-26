import gradio as gr
import json
import numpy as np
import time
import torch
import os
import anthropic
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# --- Rolling latency metrics (in-process; resets on restart) ---------------
# Tracks end-to-end respond() time: MuRIL/BERT intent classification +
# tactic retrieval + the Claude Haiku call. Capped so long-running demo
# sessions don't grow this unbounded.
_LATENCY_SAMPLES: list[float] = []
_LATENCY_CAP = 500


def _record_latency(seconds: float):
    _LATENCY_SAMPLES.append(seconds)
    del _LATENCY_SAMPLES[:-_LATENCY_CAP]


def _percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(pct / 100 * len(sorted_values))) - 1)
    return sorted_values[max(idx, 0)]


def latency_stats_text() -> str:
    if not _LATENCY_SAMPLES:
        return "No requests yet this session."
    s = sorted(_LATENCY_SAMPLES)
    return (
        f"n={len(s)} | avg={sum(s) / len(s):.2f}s | "
        f"p50={_percentile(s, 50):.2f}s | p95={_percentile(s, 95):.2f}s | "
        f"max={s[-1]:.2f}s"
    )

MURIL_MODEL = "models/muril_finetuned"
BERT_MODEL = "models/bert_finetuned"


class BargainingAgent:
    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")

        print("Loading India model (MuRIL)...")
        self.muril_embeddings = np.load("index/muril_finetuned_embeddings.npy")
        with open("index/muril_finetuned_metadata.json") as f:
            self.muril_metadata = json.load(f)
        with open("models/muril_finetuned/label_map.json") as f:
            lm = json.load(f)
        self.muril_id_to_label = {v: k for k, v in lm.items()}
        self.muril_tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL)
        self.muril_classifier = AutoModelForSequenceClassification.from_pretrained(
            MURIL_MODEL).to(self.device)
        self.muril_classifier.eval()
        self.muril_embed = AutoModel.from_pretrained(
            MURIL_MODEL).to(self.device)
        self.muril_embed.eval()

        print("Loading Global model (BERT)...")
        self.bert_embeddings = np.load("index/bert_finetuned_embeddings.npy")
        with open("index/bert_finetuned_metadata.json") as f:
            self.bert_metadata = json.load(f)
        with open("models/bert_finetuned/label_map.json") as f:
            lm2 = json.load(f)
        self.bert_id_to_label = {v: k for k, v in lm2.items()}
        self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.bert_classifier = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL).to(self.device)
        self.bert_classifier.eval()
        self.bert_embed = AutoModel.from_pretrained(BERT_MODEL).to(self.device)
        self.bert_embed.eval()

        self.claude = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"))
        print("Both models ready!")

    def classify_intent(self, text, market):
        tokenizer = self.muril_tokenizer if market == "india" else self.bert_tokenizer
        classifier = self.muril_classifier if market == "india" else self.bert_classifier
        id_to_label = self.muril_id_to_label if market == "india" else self.bert_id_to_label
        enc = tokenizer(text, truncation=True, padding=True,
                        max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = classifier(**enc)
        probs = torch.softmax(out.logits, dim=1)
        prob, idx = torch.max(probs, dim=1)
        return id_to_label[idx.item()], prob.item()

    def get_embedding(self, text, market):
        tokenizer = self.muril_tokenizer if market == "india" else self.bert_tokenizer
        embed_model = self.muril_embed if market == "india" else self.bert_embed
        enc = tokenizer(text, truncation=True, padding=True,
                        max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = embed_model(**enc)
        emb = out[0].mean(dim=1)
        return torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()

    def retrieve_tactic(self, query, market, pillar=None):
        embeddings = self.muril_embeddings if market == "india" else self.bert_embeddings
        metadata = self.muril_metadata if market == "india" else self.bert_metadata
        qe = self.get_embedding(query, market)
        scores = np.dot(embeddings, qe.T).flatten()
        if pillar:
            filtered = [i for i, m in enumerate(
                metadata) if m["pillar"].lower() == pillar.lower()]
            fs = np.full(len(embeddings), -1.0)
            for i in filtered:
                fs[i] = scores[i]
            top = np.argsort(fs)[::-1][:2]
        else:
            top = np.argsort(scores)[::-1][:2]
        return metadata[top[0]]["text"][:300], metadata[top[0]]["book"], metadata[top[0]]["page_number"]

    def get_pillar(self, intent):
        return {
            "price_objection": "Negotiation",
            "walkaway_threat": "Negotiation",
            "competitor_comparison": "Sales",
            "quality_doubt": "Persuasion",
            "ready_to_buy": "Sales",
            "guilt_pressure": "Power",
            "urgent_buyer": "Sales",
            "value_seeker": "Persuasion",
            "trust_issue": "Persuasion",
            "neutral": "Negotiation"
        }.get(intent, "Negotiation")

    def next_price(self, intent, current, floor):
        drops = {"price_objection": 20,
                 "walkaway_threat": 30, "ready_to_buy": 0}
        drop = drops.get(intent, 10)
        return max(current - drop, floor)

    def respond(self, customer_msg, current_offer, floor_price, mrp, product_name, market, gender, history):
        _t0 = time.monotonic()
        intent, conf = self.classify_intent(customer_msg, market)
        pillar = self.get_pillar(intent)
        tactic, book, page = self.retrieve_tactic(customer_msg, market, pillar)
        next_offer = self.next_price(intent, current_offer, floor_price)
        at_floor = next_offer == floor_price

        if market == "india":
            if gender == "Female":
                address = "Didi"
                tone = "warm, sisterly, lightly playful — like a friendly older sister who runs a shop"
                example = "Arre Didi, aapki choice ekdum amazing hai! Aapke liye Rs.820 final kar deti hoon — free delivery bhi milegi 😊"
            elif gender == "Male":
                address = "Bhaiya"
                tone = "friendly, brotherly, light humor — like a fun shopkeeper who enjoys chatting"
                example = "Bhaiya aapki nazar toh ekdum sahi jagah padi! Rs.820 mein le jao — best deal hai aaj ka, guarantee se 😄"
            else:
                address = "Aap"
                tone = "warm, respectful, light humor"
                example = "Aapke liye Rs.820 final kar deta hoon — ekdum best price hai yeh aaj ke liye 😊"

            language_instruction = f"""Respond in natural Hinglish (Hindi + English mix).
Address the customer as {address}.
Tone: {tone}

Important rules:
- Add light humor or a playful comment naturally — not forced
- Praise the customer's choice genuinely — make them feel good
- Use warmth — make them feel special, not just another customer
- Never sound desperate or robotic or boring
- Keep it 2-3 sentences maximum
- 1-2 emojis only
- Use scarcity or urgency naturally if relevant

Good example: '{example}'"""

        else:
            language_instruction = """Respond in natural conversational English.
Warm, confident, friendly and slightly playful — like a real shop owner who genuinely enjoys helping customers.
Add a light compliment about their choice. Make them feel like they got a great deal.
Never sound like a bot. Keep it short — 2-3 sentences max.
Good example: 'Great taste! I can do Rs.820 for you — honestly that is our best price and you are getting a steal today 😄'"""

        prompt = f"""You are an experienced shopkeeper negotiating on WhatsApp for {product_name}.

Customer said: "{customer_msg}"
Customer intent: {intent}

Negotiation tactic from {book} Page {page}:
{tactic}

Price: Previous Rs.{current_offer} — New offer Rs.{next_offer}
{'This is your FINAL price. Do not go lower under any circumstances.' if at_floor else 'You can negotiate slightly more if genuinely needed.'}

Instructions:
{language_instruction}

Write only the WhatsApp reply message. Nothing else."""

        message = self.claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text.strip()
        latency_s = time.monotonic() - _t0
        _record_latency(latency_s)
        info = (f"Intent: {intent} ({conf:.0%}) | Source: {book[:35]}... Page {page} | "
                f"Offer: Rs.{current_offer} → Rs.{next_offer} | Gender: {gender} | "
                f"Latency: {latency_s:.2f}s")
        history.append({"role": "user", "content": customer_msg})
        history.append({"role": "assistant", "content": response})
        return history, info, next_offer, ""


print("Loading agent...")
agent = BargainingAgent()
print("Agent ready!")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0a1628 !important;
    color: #e2e8f0 !important;
}

.stack-info {
    background: #0f1f35;
    border: 0.5px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 0.78rem;
    color: #475569;
    line-height: 2;
}

.footer-note {
    text-align: center;
    color: #1e3a5f;
    font-size: 0.75rem;
    padding: 1rem;
    margin-top: 1rem;
    border-top: 0.5px solid #1e3a5f;
}
"""


def chat(message, history, current_offer, floor_price, mrp, product_name, market, gender):
    if not message.strip():
        return history, "", current_offer, "", latency_stats_text()
    market_key = "india" if market == "India (Hinglish)" else "global"
    history, info, new_offer, _ = agent.respond(
        message, int(current_offer), int(floor_price), int(mrp),
        product_name, market_key, gender, history
    )
    return history, info, new_offer, "", latency_stats_text()


def reset_chat(mrp):
    starting = int(float(mrp) * 0.94)
    return [], "Configure your product and start negotiating...", starting


def update_price(x):
    return x


def set_q1(): return "bahut mehnga hai bhaiya"
def set_q2(): return "quality acchi nahi lagti"
def set_q3(): return "Amazon pe sasta milega"
def set_q4(): return "final price kya hai"
def set_q5(): return "This is too expensive"
def set_q6(): return "I can find it cheaper elsewhere"
def set_q7(): return "Can you do better on price?"
def set_q8(): return "Ok I will take it"


with gr.Blocks(title="BargainAI") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:2rem 1rem 1.5rem;background:linear-gradient(180deg,#0f2744 0%,#0a1628 100%);border-bottom:0.5px solid #1e3a5f;margin-bottom:1rem;">
        <div style="font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#f5c842,#e8954a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            BargainAI
        </div>
        <div style="color:#475569;font-size:0.9rem;margin-top:0.3rem;">
            AI-powered negotiation agent · India + Global · Gender-aware · 9 books · MuRIL + BERT
        </div>
    </div>
    """)

    with gr.Row():

        with gr.Column(scale=1):
            gr.HTML("<div style='color:#60a5fa;font-size:0.82rem;font-weight:600;letter-spacing:1px;margin-bottom:8px;'>PRODUCT CONFIG</div>")

            market = gr.Radio(
                choices=["India (Hinglish)", "Global (English)"],
                value="India (Hinglish)",
                label="Market"
            )
            gender = gr.Radio(
                choices=["Male", "Female", "Unknown"],
                value="Male",
                label="Customer Gender"
            )
            product_name = gr.Textbox(
                value="Premium Cotton Kurti", label="Product Name")
            mrp = gr.Number(value=899, label="MRP (Rs.)")
            floor_price = gr.Number(
                value=749, label="Floor Price — Never go below")
            current_offer = gr.State(value=849)
            price_display = gr.Number(
                value=849, label="Live Offer Price (Rs.)", interactive=False)
            reset_btn = gr.Button("Reset Conversation", variant="secondary")

            gr.HTML("""
            <div class="stack-info">
                <div style='color:#60a5fa;font-weight:600;margin-bottom:4px;'>Intelligence Stack</div>
                India: MuRIL fine-tuned<br>
                Global: BERT fine-tuned<br>
                Gender-aware · Bhaiya / Didi / Aap<br>
                Warm humor · Natural Hinglish<br>
                9 books · 2,383 indexed chunks<br>
                Claude Haiku · Real-time responses<br>
                <span style="color:#334155;">Run `python eval_intent_classifier.py` for a real, reproducible per-intent classification_report (precision/recall/F1) instead of a static accuracy figure.</span>
            </div>
            """)

            latency_display = gr.Textbox(
                value="No requests yet this session.",
                label="Session Latency (classify + retrieve + Haiku call)",
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.HTML("<div style='color:#60a5fa;font-size:0.82rem;font-weight:600;letter-spacing:1px;margin-bottom:8px;'>WHATSAPP NEGOTIATION SIMULATOR</div>")

            chatbot = gr.Chatbot(
                value=[],
                height=420,
                show_label=False
            )

            intel_bar = gr.Textbox(
                value="Configure your product and start negotiating...",
                label="Agent Intelligence",
                interactive=False
            )

            gr.HTML(
                "<div style='color:#475569;font-size:0.78rem;margin:8px 0 4px;'>India quick replies:</div>")
            with gr.Row():
                q1 = gr.Button("bahut mehnga hai", size="sm")
                q2 = gr.Button("quality acchi nahi", size="sm")
                q3 = gr.Button("Amazon pe sasta", size="sm")
                q4 = gr.Button("final price kya hai", size="sm")

            gr.HTML(
                "<div style='color:#475569;font-size:0.78rem;margin:8px 0 4px;'>Global quick replies:</div>")
            with gr.Row():
                q5 = gr.Button("Too expensive", size="sm")
                q6 = gr.Button("Cheaper elsewhere", size="sm")
                q7 = gr.Button("Can you do better?", size="sm")
                q8 = gr.Button("Ok I will take it", size="sm")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type message in Hinglish or English...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    gr.HTML("""
    <div class="footer-note">
        Built by Nitesh Nankani · MuRIL + BERT + Claude Haiku + 9 Negotiation Books · Python · HuggingFace · Gradio
    </div>
    """)

    q1.click(fn=set_q1, outputs=msg_input)
    q2.click(fn=set_q2, outputs=msg_input)
    q3.click(fn=set_q3, outputs=msg_input)
    q4.click(fn=set_q4, outputs=msg_input)
    q5.click(fn=set_q5, outputs=msg_input)
    q6.click(fn=set_q6, outputs=msg_input)
    q7.click(fn=set_q7, outputs=msg_input)
    q8.click(fn=set_q8, outputs=msg_input)

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, current_offer,
                floor_price, mrp, product_name, market, gender],
        outputs=[chatbot, intel_bar, current_offer, msg_input, latency_display]
    ).then(fn=update_price, inputs=[current_offer], outputs=[price_display])

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, current_offer,
                floor_price, mrp, product_name, market, gender],
        outputs=[chatbot, intel_bar, current_offer, msg_input, latency_display]
    ).then(fn=update_price, inputs=[current_offer], outputs=[price_display])

    reset_btn.click(
        fn=reset_chat,
        inputs=[mrp],
        outputs=[chatbot, intel_bar, current_offer]
    ).then(fn=update_price, inputs=[current_offer], outputs=[price_display])

if __name__ == "__main__":
    demo.launch(share=True)
