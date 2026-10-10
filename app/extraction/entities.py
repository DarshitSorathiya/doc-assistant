# ============================================================
# Entity Extraction — Indian Identifiers & Document Entities
# ============================================================
# High-precision regex and rule-based extractors for:
# PAN, Aadhaar, IFSC, Passport, Driving License, Policy Numbers,
# Bank Accounts, Dates/Expiry Dates, Amounts, Emails, Phone Numbers.
# ============================================================

import re
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

from app.ocr.base import PageResult


class ExtractedEntity(BaseModel):
    """Structured representation of an extracted entity."""

    entity_type: str = Field(..., description="Type: PAN, Aadhaar, IFSC, Passport, Policy Number, etc.")
    value: str = Field(..., description="Extracted entity string value")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0, description="Extraction confidence score")
    source_page: Optional[int] = Field(default=1, description="1-indexed source page")
    context: Optional[str] = Field(default=None, description="Snippet or surrounding text context")


# ------------------------------------------------------------------
# Verhoeff Algorithm for Aadhaar Validation
# ------------------------------------------------------------------
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def validate_verhoeff(num_str: str) -> bool:
    """Validate 12-digit number using Verhoeff checksum."""
    if not num_str.isdigit() or len(num_str) != 12:
        return False
    c = 0
    for i, item in enumerate(reversed(num_str)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


# ------------------------------------------------------------------
# Entity Extractor Class
# ------------------------------------------------------------------
class EntityExtractor:
    """Extracts structured entities from document text or page lists."""

    # Pre-compiled Regex Patterns
    RE_PAN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
    RE_AADHAAR = re.compile(r"\b([2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4})\b")
    RE_IFSC = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")
    RE_PASSPORT = re.compile(r"\b([A-PR-WYa-pr-wy][1-9]\d\s?\d{4}[1-9]|[A-Z][0-9]{7})\b")
    RE_DRIVING_LICENSE = re.compile(r"\b([A-Z]{2}[ -]?[0-9]{2}[ -]?(?:19|20)[0-9]{2}[ -]?[0-9]{7}|[A-Z]{2}[0-9]{13,14})\b")

    RE_POLICY_NUMBER = re.compile(
        r"(?:policy\s*(?:no|number|#)?[:\s-]*|\bPOL[-_]?)([A-Za-z0-9\/-]{6,25})\b",
        re.IGNORECASE,
    )
    RE_ACCOUNT_NUMBER = re.compile(
        r"(?:account\s*(?:no|number|#)?|a\/c\s*(?:no|number)?|acct\s*#?)[:\s-]*([0-9]{9,18})\b",
        re.IGNORECASE,
    )

    RE_EMAIL = re.compile(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
    RE_PHONE = re.compile(r"(?:(?:\+91|0)[\-\s]?)?([6-9]\d{9})\b")

    RE_AMOUNT = re.compile(
        r"(?:Rs\.?|INR|₹|USD|\$)\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)\b",
        re.IGNORECASE,
    )

    RE_DATE = re.compile(
        r"\b((?:0?[1-9]|[12][0-9]|3[01])[\/\-\.](?:0?[1-9]|1[012])[\/\-\.](?:19|20)\d\d|"
        r"(?:19|20)\d\d[\/\-\.](?:0?[1-9]|1[012])[\/\-\.](?:0?[1-9]|[12][0-9]|3[01])|"
        r"(?:0?[1-9]|[12][0-9]|3[01])[\s\-\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-\.](?:19|20)\d\d)\b",
        re.IGNORECASE,
    )

    RE_EXPIRY_CONTEXT = re.compile(
        r"(?:expir(?:y|es|ation)|valid\s*(?:till|upto|to|through)|due\s*date|renew(?:al)?\s*by)[:\s-]*"
        r"((?:0?[1-9]|[12][0-9]|3[01])[\/\-\.](?:0?[1-9]|1[012])[\/\-\.](?:19|20)\d\d|"
        r"(?:19|20)\d\d[\/\-\.](?:0?[1-9]|1[012])[\/\-\.](?:0?[1-9]|[12][0-9]|3[01])|"
        r"(?:0?[1-9]|[12][0-9]|3[01])[\s\-\.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-\.](?:19|20)\d\d)",
        re.IGNORECASE,
    )

    RE_PERSON_NAME = re.compile(
        r"(?:insured\s*name|holder\s*name|patient\s*name|employee\s*name|student\s*name|name\s*of\s*holder|mr\.|mrs\.|ms\.)[:\s-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        re.IGNORECASE,
    )

    @classmethod
    def extract_from_text(cls, text: str, page_num: int = 1) -> List[ExtractedEntity]:
        """Extract all entities from a given text string for a specific page."""
        entities: List[ExtractedEntity] = []
        seen_keys: Set[Tuple[str, str]] = set()

        def add_entity(etype: str, val: str, conf: float = 0.95, ctx: Optional[str] = None):
            clean_val = val.strip()
            if not clean_val or len(clean_val) < 2:
                return
            key = (etype, clean_val.upper())
            if key not in seen_keys:
                seen_keys.add(key)
                entities.append(
                    ExtractedEntity(
                        entity_type=etype,
                        value=clean_val,
                        confidence=conf,
                        source_page=page_num,
                        context=ctx,
                    )
                )

        # 1. PAN
        for m in cls.RE_PAN.finditer(text):
            pan = m.group(1).upper()
            conf = 0.99 if pan[3] in "PCHFATBLJGD" else 0.95
            add_entity("PAN", pan, conf=conf)

        # 2. Aadhaar
        for m in cls.RE_AADHAAR.finditer(text):
            raw_aadhaar = re.sub(r"\s+", "", m.group(1))
            if len(raw_aadhaar) == 12 and not raw_aadhaar.startswith("0") and not raw_aadhaar.startswith("1"):
                is_valid = validate_verhoeff(raw_aadhaar)
                conf = 0.98 if is_valid else 0.80
                formatted_aadhaar = f"{raw_aadhaar[:4]} {raw_aadhaar[4:8]} {raw_aadhaar[8:]}"
                add_entity("Aadhaar", formatted_aadhaar, conf=conf)

        # 3. IFSC Code
        for m in cls.RE_IFSC.finditer(text):
            add_entity("IFSC", m.group(1).upper(), conf=0.98)

        # 4. Passport
        for m in cls.RE_PASSPORT.finditer(text):
            passport_num = re.sub(r"\s+", "", m.group(1)).upper()
            add_entity("Passport", passport_num, conf=0.92)

        # 5. Driving License
        for m in cls.RE_DRIVING_LICENSE.finditer(text):
            add_entity("Driving License", m.group(1).upper(), conf=0.92)

        # 6. Policy Number
        for m in cls.RE_POLICY_NUMBER.finditer(text):
            pol_num = m.group(1).strip()
            if len(pol_num) >= 5 and not pol_num.isdigit():
                add_entity("Policy Number", pol_num, conf=0.90)
            elif pol_num.isdigit() and len(pol_num) >= 7:
                add_entity("Policy Number", pol_num, conf=0.85)

        # 7. Bank Account Number
        for m in cls.RE_ACCOUNT_NUMBER.finditer(text):
            acc_num = m.group(1).strip()
            if 9 <= len(acc_num) <= 18:
                add_entity("Bank Account Number", acc_num, conf=0.88)

        # 8. Expiry Dates
        for m in cls.RE_EXPIRY_CONTEXT.finditer(text):
            add_entity("Expiry Date", m.group(1).strip(), conf=0.95, ctx=m.group(0))

        # 9. General Dates
        for m in cls.RE_DATE.finditer(text):
            date_val = m.group(1).strip()
            add_entity("Date", date_val, conf=0.85)

        # 10. Financial Amounts
        for m in cls.RE_AMOUNT.finditer(text):
            amt_val = m.group(1).strip()
            # Filter out standalone integers that look like years or small integers
            if amt_val not in {"2020", "2021", "2022", "2023", "2024", "2025", "2026"}:
                add_entity("Amount", amt_val, conf=0.90, ctx=m.group(0))

        # 11. Emails & Phone Numbers
        for m in cls.RE_EMAIL.finditer(text):
            add_entity("Email", m.group(1).lower(), conf=0.99)

        for m in cls.RE_PHONE.finditer(text):
            phone_num = m.group(1).strip()
            add_entity("Phone Number", phone_num, conf=0.90)

        # 12. Person Names
        for m in cls.RE_PERSON_NAME.finditer(text):
            name_val = m.group(1).strip()
            add_entity("Person Name", name_val, conf=0.85)

        return entities

    @classmethod
    def extract_from_pages(cls, pages: List[PageResult]) -> List[ExtractedEntity]:
        """Extract all entities across all pages of a document."""
        all_entities: List[ExtractedEntity] = []
        for page in pages:
            page_entities = cls.extract_from_text(page.text, page_num=page.page_num)
            all_entities.extend(page_entities)
        return all_entities
