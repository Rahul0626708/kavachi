from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_response

app = Flask(__name__)

@app.route("/")
def home():
    return "<h2>🤖 TensorFlow Chatbot Running</h2>"

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")
    response = chatbot_response(user_msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)