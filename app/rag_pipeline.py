import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path, override=True)


def generate_answer(query, context):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "API key not configured"

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
You are a professional medical AI assistant.

Instructions:
- Answer ONLY using the given context
- Do NOT hallucinate
- If answer is not found, say: "Not available in provided documents"
- Keep answer clear and medically accurate

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a safe and factual medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating answer: {str(e)}"