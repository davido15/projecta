
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from uuid import uuid4
from dotenv import load_dotenv
import logging

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# In-memory storage for projects and messages
projects = []  # Each project: {id, name}
project_messages = {}  # project_id: [messages]

# --- Project Management Endpoints ---
@app.route('/projects', methods=['GET'])
def get_projects():
    return jsonify(projects)

@app.route('/projects', methods=['POST'])
def create_project():
    data = request.get_json()
    name = data.get('name', f"Project {len(projects) + 1}")
    project_id = str(uuid4())
    project = {'id': project_id, 'name': name}
    projects.append(project)
    project_messages[project_id] = []
    return jsonify(project)

@app.route('/projects/<project_id>/messages', methods=['GET'])
def get_project_messages(project_id):
    return jsonify(project_messages.get(project_id, []))

# --- Message Formatting (already exists) ---
@app.route('/format', methods=['POST'])
def format_update():
    data = request.get_json()
    raw_update = data.get('update')
    project_id = data.get('project_id')
    project_name = data.get('project_name', '')
    prompt = f"""
You are a Project Management Assistant. Reformat the following update into 5 structured sections:
✅ Current Phase
🔜 Next Stage
⛔ Blockers
🛠 Actions Required
📅 Timeline or Deadline

For each section:
- Use bullet points for each item.
- Highlight key points using bold text or emojis.
- Make the update easy to read and visually clear.

---
Update: {raw_update}
"""
    logging.info(f"[FORMAT] project_id={project_id} | user_input={raw_update}\nprompt={prompt}")
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful project assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    formatted = response.choices[0].message.content
    logging.info(f"[FORMAT] project_id={project_id} | ai_response={formatted}")
    # Save message to project if project_id is provided
    if project_id and project_id in project_messages:
        project_messages[project_id].append({'role': 'user', 'content': raw_update})
        project_messages[project_id].append({'role': 'bot', 'content': formatted})
    return jsonify({"formatted": formatted})

# --- Summarize Project (already exists) ---
@app.route('/summarize', methods=['POST'])
def summarize_project():
    data = request.get_json()
    raw_notes = data.get('update')
    project_id = data.get('project_id')
    project_name = data.get('project_name', '')
    summary_prompt = f"""
Act as a professional project manager. Given the following project notes, summarize the current project status clearly and concisely for stakeholders. Include:
- Project progress
- Current blockers
- Next steps
- Timeline and any deadlines

Use clear bullet points and structure.

---
Notes: {raw_notes}
"""
    logging.info(f"[SUMMARIZE] project_id={project_id} | user_input={raw_notes}\nprompt={summary_prompt}")
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a senior project manager assistant."},
            {"role": "user", "content": summary_prompt}
        ]
    )
    summary = response.choices[0].message.content
    logging.info(f"[SUMMARIZE] project_id={project_id} | ai_response={summary}")
    return jsonify({"summary": summary})

# --- Extract Action Items ---
@app.route('/action-items', methods=['POST'])
def extract_action_items():
    data = request.get_json()
    updates = data.get('update')
    project_id = data.get('project_id')
    project_name = data.get('project_name', '')
    prompt = f"""
Extract all specific action items from the following project updates. For each action item, write a single, clear sentence starting with a numbered markdown bullet (e.g., 1.), then the action title in bold (using markdown, e.g., **Title**), followed by a colon and a concise description. Keep each action item to a maximum of 1-2 lines. List each action item on a new line as a separate numbered bullet. Do not use JSON or field labels. Only return the list of action items. If there are no specific action items, reply with: No specific action items found in the provided updates.

Example:
1. **Finalize Vendor Selection**: Follow up with Finance to review the latest proposal for the reporting module.
2. **Resolve Merge Conflict**: Dev team to address the merge conflict in the notification service update.
3. **Test User Role Flow Changes**: QA to test the user role flow changes once staging env stabilizes.

---
Project: {project_name}
Updates: {updates}
"""
    logging.info(f"[ACTION-ITEMS] project_id={project_id} | user_input={updates}\nprompt={prompt}")
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a project management AI that extracts action items as a concise numbered markdown list with bolded titles."},
            {"role": "user", "content": prompt}
        ]
    )
    action_items = response.choices[0].message.content.strip()
    logging.info(f"[ACTION-ITEMS] project_id={project_id} | ai_response={action_items}")
    return jsonify({"action_items": action_items})

# --- Sentiment Analysis ---
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    updates = data.get('update')
    project_id = data.get('project_id')
    project_name = data.get('project_name', '')
    prompt = f"""
Analyze the overall sentiment of the following project updates. Respond in the following format:

AI
Overall sentiment: <Positive/Neutral/Negative>

Explanation: <A brief explanation of the tone and key points in the updates.>

---
Project: {project_name}
Updates: {updates}
"""
    logging.info(f"[SENTIMENT] project_id={project_id} | user_input={updates}\nprompt={prompt}")
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis AI."},
            {"role": "user", "content": prompt}
        ]
    )
    sentiment = response.choices[0].message.content.strip()
    logging.info(f"[SENTIMENT] project_id={project_id} | ai_response={sentiment}")
    return jsonify({"sentiment": sentiment})

# --- Generate Email ---
@app.route('/generate-email', methods=['POST'])
def generate_email():
    data = request.get_json()
    updates = data.get('update')
    project_id = data.get('project_id')
    project_name = data.get('project_name', '')
    sentiment = data.get('sentiment', '')
    prompt = f"""
Write a professional project update email for the following project. Use the sentiment: {sentiment}.

---
Project: {project_name}
Updates: {updates}
"""
    logging.info(f"[EMAIL] project_id={project_id} | user_input={updates}\nprompt={prompt}")
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert project manager and email writer."},
            {"role": "user", "content": prompt}
        ]
    )
    email = response.choices[0].message.content
    logging.info(f"[EMAIL] project_id={project_id} | ai_response={email}")
    return jsonify({"email": email})

# --- Subscription (Stub) ---
@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    # Stub: In production, integrate with Stripe or payment provider
    return jsonify({"url": "https://example.com/checkout"})

if __name__ == '__main__':
    app.run(debug=True)