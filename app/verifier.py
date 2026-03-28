import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path, override=True)


def verify_answer(answer, context):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "ERROR"

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
You are a strict medical evaluator.

Check if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Rules:
- Reply ONLY with YES or NO
- Do not explain
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You verify factual correctness."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip().upper()

        if "YES" in result:
            return "YES"
        elif "NO" in result:
            return "NO"
        else:
            return "UNKNOWN"

    except Exception:
        return "ERROR"