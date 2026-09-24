import os
import json

from langchain_groq import ChatGroq

from src.schemas import AnswerResponse



MOCK_LLM = os.getenv("MOCK_LLM", "1")


SUPPORT_PROMPT = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
You will receive policy information retrieved from Zepto's
official policy document corpus.

TASK:
Answer the customer's question using only the provided context.

FORMAT:
Return a JSON object with exactly these fields:
{
  "answer": "string",
  "sources": ["document or chunk IDs"],
  "confidence": 0.0
}

LENGTH:
Keep the answer concise and directly address the customer's question.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, timings, or rules.

FEW-SHOT EXAMPLE:

Customer question:
"When can I cancel my Zepto order?"

Retrieved context:
"Orders can be cancelled free of cost any time before the order status
changes to 'Packed'."

Expected JSON:
{
  "answer": "An order can be cancelled free of cost before its status changes to 'Packed'.",
  "sources": ["doc_05.txt"],
  "confidence": 1.0
}

Now answer the following customer question.

Customer question:
{query}

Retrieved context:
{context}
"""



def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key,
    )


def generate_real_answer(
    query: str,
    context: str,
    sources: list[str],
) -> AnswerResponse:

    llm = get_llm()

    prompt = SUPPORT_PROMPT.format(
        query=query,
        context=context,
    )

    last_error = None

    for attempt in range(3):

        try:

            if attempt == 0:

                response = llm.invoke(prompt)

            else:

                corrective_prompt = f"""
Your previous response failed validation.

Return ONLY valid JSON matching this exact schema:

{{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}}

Do not include Markdown.
Do not include ```json.
Do not add any fields.

Customer question:
{query}

Retrieved context:
{context}

Your previous response:
{response.content}

Validation error:
{last_error}
"""

                response = llm.invoke(corrective_prompt)

            raw_output = response.content.strip()


            if raw_output.startswith("```json"):
                raw_output = raw_output[7:]

            if raw_output.startswith("```"):
                raw_output = raw_output[3:]

            if raw_output.endswith("```"):
                raw_output = raw_output[:-3]

            raw_output = raw_output.strip()

            data = json.loads(raw_output)
            data["sources"] = sources

            validated = AnswerResponse.model_validate(data)

            return validated

        except Exception as exc:

            last_error = str(exc)

    raise ValueError(
        f"LLM response failed validation after 3 attempts: {last_error}"
    )