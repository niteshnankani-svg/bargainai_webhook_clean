from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from gradio_client import Client
import traceback
import os

app = Flask(__name__)

conversations = {}

# Load from environment variable — never hardcode
HF_TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_FLOOR = 749
DEFAULT_MRP = 899

print("Connecting to BargainAI Space...")
client = Client(
    "nitz0219/BargainAI",
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
)
print("Connected!")


def get_last_message(history):
    if not history or len(history) == 0:
        return "Ek minute bhaiya!"

    last_msg = history[-1]

    if isinstance(last_msg, dict):
        content = last_msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "Ek minute bhaiya!")
            return "Ek minute bhaiya!"
        else:
            return str(content)
    else:
        return str(last_msg)


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get("Body", "").strip()
    user_number = request.values.get("From", "")

    print(f"\nFrom: {user_number}")
    print(f"Message: {incoming_msg}")

    if user_number not in conversations:
        conversations[user_number] = {
            "history": []
        }

    state = conversations[user_number]

    try:
        result = client.predict(
            message=incoming_msg,
            history=state["history"],
            floor_price=DEFAULT_FLOOR,
            mrp=DEFAULT_MRP,
            product_name="Premium Cotton Kurti",
            market="India (Hinglish)",
            gender="Unknown",
            api_name="/chat"
        )

        new_history = result[0]
        state["history"] = new_history

        agent_reply = get_last_message(new_history)
        print(f"Agent reply: {agent_reply}")

    except Exception as e:
        print(f"FULL ERROR: {traceback.format_exc()}")
        agent_reply = "Arre bhaiya thoda technical issue hai — ek minute mein try karein! 😊"

    resp = MessagingResponse()
    resp.message(agent_reply)
    return str(resp)


@app.route("/health")
def health():
    return "BargainAI WhatsApp Webhook running!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
