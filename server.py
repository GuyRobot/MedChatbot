from flask import Flask, request
import requests
import os
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

HfFolder.save_token('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')

model_checkpoint = "facebook/bart-base"
model_name = model_checkpoint.split("/")[-1]
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_name}-finetuned-xsum")


def generate_summary(question, model):
    inputs = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


app = Flask(__name__)

FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
VERIFY_TOKEN = '5rApTs/BRm6jtiwApOpIdjBHe73ifm6mNGZOsYkwwAw='
PAGE_ACCESS_TOKEN = os.environ['PAGE_ACCESS_TOKEN']  # paste your page access token here>"


def get_bot_response(message):
    return generate_summary(message, model)[1][0]


def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"


def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = get_bot_response(message)
    send_message(sender, response)
    return response


def is_user_message(message):
    """Check if the message is a message from the user"""
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))


@app.route("/webhook", methods=['GET', 'POST'])
def listen():
    """This is the main function flask uses to
    listen at the `/webhook` endpoint"""
    if request.method == 'GET':
        return verify_webhook(request)

    if request.method == 'POST':
        payload = request.json
        event = payload['entry'][0]['messaging']
        res = "ok"
        for x in event:
            if is_user_message(x):
                text = x['message']['text']
                sender_id = x['sender']['id']
                res = respond(sender_id, text)

        return res


def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {
        'message': {
            'text': text
        },
        'recipient': {
            'id': recipient_id
        },
        'notification_type': 'regular'
    }

    auth = {
        'access_token': PAGE_ACCESS_TOKEN
    }

    response = requests.post(
        FB_API_URL,
        params=auth,
        json=payload
    )

    return response.json()
