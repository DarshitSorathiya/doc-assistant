# ============================================================
# Document Classification (Rule-Based + LLM Hybrid)
# ============================================================
# Classifies documents into 12 standardized categories:
# Insurance, Banking, Government ID, Property, Tax, Medical,
# Education, Employment, Legal, Financial, Utility, Other.
# ============================================================

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field

from app.models import DocumentCategory
from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    category: DocumentCategory = Field(..., description="Assigned document category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    method: str = Field(..., description="Classification method: rule | llm | hybrid")
    reasoning: Optional[str] = Field(None, description="Explanation or top matched keywords")


# Keyword signals with relative weights for rule-based classification
CATEGORY_KEYWORDS: Dict[DocumentCategory, Dict[str, float]] = {
    DocumentCategory.INSURANCE: {
        "insurance": 2.0, "policy schedule": 3.0, "policy number": 3.0, "sum insured": 3.0,
        "premium paid": 2.5, "premium amount": 2.5, "claim helpline": 2.0, "nominee": 2.0,
        "third party administrator": 2.5, "tpa": 1.5, "cashless hospital": 2.0, "deductible": 2.0,
        "star health": 3.0, "hdfc ergo": 3.0, "lic of india": 3.0, "max life": 3.0, "icici prudential": 3.0,
        "health insurance": 3.0, "life insurance": 3.0, "motor insurance": 3.0,
    },
    DocumentCategory.BANKING: {
        "account statement": 3.0, "bank statement": 3.0, "savings account": 2.5, "current account": 2.5,
        "ifsc code": 2.5, "ifsc": 2.0, "branch": 1.0, "account balance": 2.5, "closing balance": 2.5,
        "opening balance": 2.0, "debit": 1.5, "credit": 1.5, "withdrawal": 1.5, "deposit": 1.5,
        "cheque": 2.0, "passbook": 2.5, "hdfc bank": 2.5, "state bank of india": 2.5, "icici bank": 2.5,
        "axis bank": 2.5, "transaction date": 2.0,
    },
    DocumentCategory.GOVERNMENT_ID: {
        "income tax department": 3.5, "permanent account number": 3.5, "pan card": 3.5,
        "unique identification authority of india": 3.5, "uidai": 3.0, "aadhaar": 3.5, "myaadhaar": 3.0,
        "republic of india": 2.0, "passport": 3.5, "driving licence": 3.5, "driving license": 3.5,
        "election commission of india": 3.5, "voter identity card": 3.5, "epic no": 3.0,
        "transport department": 2.5, "govt of india": 2.0,
    },
    DocumentCategory.PROPERTY: {
        "sale deed": 3.5, "conveyance deed": 3.5, "title deed": 3.0, "property tax": 3.0,
        "possession letter": 3.0, "allotment letter": 3.0, "khata certificate": 3.0, "patta": 2.5,
        "survey number": 2.5, "plot no": 2.0, "floor plan": 2.0, "municipal corporation": 2.0,
        "sub-registrar": 2.5, "registration and stamps": 2.5, "schedule property": 2.5,
    },
    DocumentCategory.TAX: {
        "income tax return": 3.5, "itr-v": 3.5, "itr acknowledgment": 3.5, "form 16": 3.5,
        "form 26as": 3.5, "assessment year": 2.5, "financial year": 2.0, "tax deduction at source": 2.5,
        "tds certificate": 3.0, "central board of direct taxes": 3.0, "cbdt": 2.5, "total income tax": 2.5,
        "gross total income": 2.5, "taxable income": 2.5, "goods and services tax": 3.0, "gstin": 3.0,
    },
    DocumentCategory.MEDICAL: {
        "discharge summary": 3.5, "prescription": 3.0, "doctor": 2.0, "patient name": 2.5,
        "diagnosis": 3.0, "hospital": 2.0, "clinic": 2.0, "pathology report": 3.0, "radiology": 2.5,
        "blood test": 2.5, "dosage": 2.0, "pharmacy": 2.0, "opd": 2.0, "admission date": 2.5,
        "attending physician": 3.0,
    },
    DocumentCategory.EDUCATION: {
        "marksheet": 3.5, "statement of marks": 3.5, "degree certificate": 3.5, "provisional certificate": 3.0,
        "university": 2.5, "semester": 2.5, "grade card": 3.0, "academic transcript": 3.5, "gpa": 2.0,
        "cgpa": 2.5, "roll number": 2.0, "school": 1.5, "college": 2.0, "enrollment number": 2.5,
    },
    DocumentCategory.EMPLOYMENT: {
        "offer letter": 3.5, "appointment letter": 3.5, "payslip": 3.5, "salary slip": 3.5,
        "employee code": 2.5, "employee id": 2.5, "designation": 2.0, "human resources": 2.0,
        "experience letter": 3.5, "relieving letter": 3.5, "compensation": 2.0, "basic salary": 2.5,
        "house rent allowance": 2.0, "provident fund": 2.0,
    },
    DocumentCategory.LEGAL: {
        "affidavit": 3.5, "power of attorney": 3.5, "non-disclosure agreement": 3.5,
        "memorandum of understanding": 3.5, "notary public": 3.0, "sworn before me": 3.0,
        "court of": 2.5, "arbitration": 2.5, "indemnity bond": 3.5, "stamp duty": 2.0,
        "executant": 2.5, "whereas": 1.5,
    },
    DocumentCategory.FINANCIAL: {
        "mutual fund": 3.5, "folio number": 3.5, "demat account": 3.5, "holding statement": 3.0,
        "cas statement": 3.0, "systematic investment plan": 3.0, "sip": 2.0, "net asset value": 3.0,
        "nav": 2.0, "dividend": 2.0, "securities": 2.0, "portfolio value": 2.5,
    },
    DocumentCategory.UTILITY: {
        "electricity bill": 3.5, "water bill": 3.5, "gas bill": 3.5, "broadband bill": 3.5,
        "telephone bill": 3.5, "consumer number": 3.0, "ca number": 3.0, "meter number": 2.5,
        "units consumed": 3.0, "billing period": 2.5, "due date": 2.0, "discom": 2.5,
    },
}


class DocumentClassifier:
    """Classifies document text using weighted keyword matching with LLM fallback."""

    @classmethod
    def classify_rule_based(cls, text: str) -> Tuple[DocumentCategory, float, List[str]]:
        """Run rule-based classification on the text.

        Returns:
            Tuple of (best_category, confidence_score, matched_keywords).
        """
        if not text or not text.strip():
            return DocumentCategory.OTHER, 0.0, []

        lower_text = text.lower()
        category_scores: Dict[DocumentCategory, float] = {}
        category_matches: Dict[DocumentCategory, List[str]] = {}

        for category, keyword_weights in CATEGORY_KEYWORDS.items():
            score = 0.0
            matches = []
            for kw, weight in keyword_weights.items():
                # Count occurrences (cap count effect to prevent keyword stuffing)
                count = lower_text.count(kw)
                if count > 0:
                    score += weight * min(count, 3)
                    matches.append(kw)

            category_scores[category] = score
            category_matches[category] = matches

        # Find best category
        best_category = DocumentCategory.OTHER
        max_score = 0.0
        for cat, score in category_scores.items():
            if score > max_score:
                max_score = score
                best_category = cat

        if max_score == 0:
            return DocumentCategory.OTHER, 0.0, []

        # Normalize confidence (score 6.0+ is deemed very confident >= 0.85)
        raw_confidence = max_score / (max_score + 3.0)
        confidence = round(min(max(raw_confidence, 0.1), 0.99), 3)

        return best_category, confidence, category_matches.get(best_category, [])

    @classmethod
    def classify_llm(cls, text: str) -> Optional[Tuple[DocumentCategory, float, str]]:
        """Classify document using LLM for ambiguous cases."""
        settings = get_settings()
        if settings.LLM_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
            return None
        if settings.LLM_PROVIDER == "groq" and not settings.GROQ_API_KEY:
            return None
        if settings.LLM_PROVIDER not in {"openai", "ollama", "groq"}:
            return None

        prompt = (
            "You are an expert document classifier. Classify the following document excerpt into "
            "EXACTLY one of these categories:\n"
            "Insurance, Banking, Government ID, Property, Tax, Medical, Education, Employment, Legal, Financial, Utility, Other.\n\n"
            f"Document Excerpt:\n\"\"\"{text[:1500]}\"\"\"\n\n"
            "Respond ONLY in format: Category: <category> | Confidence: <0.0 to 1.0> | Reason: <short reason>"
        )

        try:
            response_text = ""
            if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
                from openai import OpenAI
                client = OpenAI(api_key=settings.OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=60,
                )
                response_text = resp.choices[0].message.content or ""
            elif settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
                import urllib.request
                payload = {
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 60,
                }
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    response_text = data["choices"][0]["message"]["content"]
            elif settings.LLM_PROVIDER == "ollama":
                import httpx
                resp = httpx.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    response_text = resp.json().get("response", "")

            # Parse LLM response
            for cat in DocumentCategory:
                if cat.value.lower() in response_text.lower():
                    # Extract confidence if present
                    conf_match = re.search(r"confidence:\s*([0-9.]+)", response_text, re.IGNORECASE)
                    conf = float(conf_match.group(1)) if conf_match else 0.85
                    return cat, min(max(conf, 0.1), 1.0), response_text

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")

        return None

    @classmethod
    def classify(cls, text: str, min_rule_confidence: float = 0.65) -> ClassificationResult:
        """Classify a document using hybrid rule-based + LLM approach."""
        rule_cat, rule_conf, matches = cls.classify_rule_based(text)

        # If rule-based confidence is solid, return immediately
        if rule_conf >= min_rule_confidence:
            reason = f"Matched keywords: {', '.join(matches[:5])}" if matches else "Rule-based pattern"
            return ClassificationResult(
                category=rule_cat,
                confidence=rule_conf,
                method="rule",
                reasoning=reason,
            )

        # Otherwise attempt LLM classification
        llm_res = cls.classify_llm(text)
        if llm_res:
            llm_cat, llm_conf, llm_reason = llm_res
            return ClassificationResult(
                category=llm_cat,
                confidence=llm_conf,
                method="llm" if rule_conf < 0.3 else "hybrid",
                reasoning=f"LLM: {llm_reason}",
            )

        # Fallback to rule result
        return ClassificationResult(
            category=rule_cat,
            confidence=rule_conf,
            method="rule",
            reasoning=f"Low-confidence rule match: {', '.join(matches)}" if matches else "Fallback to Other",
        )
