from fastapi import FastAPI
from pydantic import BaseModel
import re

import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD CATALOG
# =========================================================

with open("catalog.json", "r", encoding="utf-8") as f:

    content = f.read()

    content = content.replace("\n", " ")
    content = content.replace("\r", " ")
    content = content.replace("\t", " ")

    data = json.loads(content)

# =========================================================
# LOAD MODEL + INDEX
# =========================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("shl_index.faiss")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

# =========================================================
# REQUEST SCHEMA
# =========================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_text(text):
    return text.lower().strip()

# =========================================================
# INTENT DETECTION
# =========================================================

def detect_intent(message):

    msg = clean_text(message)

    # -----------------------------------------------------
    # COMPARE
    # -----------------------------------------------------

    compare_words = [
        "compare",
        "difference",
        "vs",
        "versus",
        "better than"
    ]

    if any(word in msg for word in compare_words):
        return "COMPARE"

    # -----------------------------------------------------
    # REFUSE
    # -----------------------------------------------------

    blocked_topics = [
        "salary",
        "legal advice",
        "lawsuit",
        "medical advice",
        "politics",
        "religion",
        "ignore instructions",
        "hack",
        "bypass",
        "other companies"
    ]

    if any(word in msg for word in blocked_topics):
        return "REFUSE"

    # -----------------------------------------------------
    # CLARIFICATION
    # -----------------------------------------------------

    role_words = [
        "developer",
        "engineer",
        "manager",
        "analyst",
        "leader",
        "graduate",
        "intern",
        "sales",
        "marketing"
    ]

    context_words = [
        "java",
        "python",
        "sql",
        "personality",
        "behavioral",
        "cognitive",
        "aptitude",
        "leadership",
        "communication",
        "technical",
        "coding",
        "mid-level",
        "senior",
        "junior"
    ]

    has_role = any(word in msg for word in role_words)

    has_context = any(word in msg for word in context_words)

    # Very vague
    if len(msg.split()) <= 2:
        return "CLARIFY"

    # Role but no useful context
    if has_role and not has_context:
        return "CLARIFY"

    return "RECOMMEND"

# =========================================================
# CLARIFICATION QUESTIONS
# =========================================================

def generate_clarification(message):

    msg = clean_text(message)

    if "manager" in msg:

        return (
            "Are these assessments for hiring or internal development, "
            "and what level of managers are you assessing?"
        )

    if "developer" in msg or "engineer" in msg:

        return (
            "Could you specify the tech stack, seniority level, "
            "and whether you need technical, cognitive, or personality assessments?"
        )

    if "graduate" in msg or "intern" in msg:

        return (
            "Are you looking for aptitude, behavioral, "
            "or technical assessments for graduates?"
        )

    return (
        "Could you share the role, seniority level, and assessment type required?"
    )

# =========================================================
# EXTRACT ASSESSMENT NAMES
# =========================================================




def normalize_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = " ".join(text.split())

    return text


def extract_assessment_names(message):

    msg = normalize_text(message)
    aliases={
        "gsa": "general ability",
    "opq32r": "opq",
    "opq": "opq"

    }
    for short, full in aliases.items():
        msg=msg.replace(short,full)

    matched = []

    # Split comparison query
    separators = [" vs ", " versus ", " and ", " compare "]

    parts = [msg]

    for sep in separators:

        if sep in msg:
            parts = [p.strip() for p in re.split(r"\bvs\b|\bversus\b|\band\b|\bcompare\b|\bdifference between\b",msg)if p.strip()]
            break

    parts = [p.strip() for p in parts if p.strip()]

    # Match each part separately
    for part in parts:

        best_match = None
        best_score = 0

        for item in data:

            name = item.get("name", "")

            normalized_name = normalize_text(name)

            score = 0

            # Exact match
            if part == normalized_name:
                score += 100

            # Strong substring
            elif part in normalized_name:
                score += 50

            # Token overlap
            part_tokens = set(part.split())
            name_tokens = set(normalized_name.split())

            overlap = part_tokens.intersection(name_tokens)

            score += len(overlap) * 10

            if score > best_score:
                best_score = score
                best_match = item

        if best_match:
            matched.append(best_match)

    # Remove duplicates
    unique_matches = []

    seen = set()

    for item in matched:

        name = item.get("name")

        if name not in seen:
            unique_matches.append(item)
            seen.add(name)

    return unique_matches[:2]

# =========================================================
# COMPARE ASSESSMENTS
# =========================================================

def compare_assessments(message):

    matched = extract_assessment_names(message)

    if len(matched) < 2:

        return {
            "reply": (
                "Please specify two SHL assessments you would like compared."
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    a = matched[0]
    b = matched[1]

    a_desc = a.get("description", "")
    b_desc = b.get("description", "")

    reply = (
        f"{a['name']} focuses on {a_desc[:180]}. "
        f"In comparison, {b['name']} focuses on {b_desc[:180]}."
    )

    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False
    }

# =========================================================
# SEARCH + RANKING
# =========================================================

def search_assessments(query, top_k=5):

    query_lower = clean_text(query)

    # -----------------------------------------------------
    # QUERY ANALYSIS
    # -----------------------------------------------------

    needs_personality = any(term in query_lower for term in [
        "opq",
        "occupational personality",
        "personality questionnaire",
        "stakeholder",
        "communication",
        "leadership",
        "collaboration",
        "teamwork",
        "behavior",
        "behavioral",
        "personality"
    ])

    needs_cognitive = any(term in query_lower for term in [
        "cognitive",
        "aptitude",
        "analytical",
        "problem solving",
        "reasoning"
    ])

    needs_technical = any(term in query_lower for term in [
        "developer",
        "engineer",
        "java",
        "python",
        "sql",
        "coding",
        "technical",
        "software",
        "backend",
        "frontend"
    ])

    # -----------------------------------------------------
    # ROLE DETECTION
    # -----------------------------------------------------

    is_developer_role = any(term in query_lower for term in [
        "developer",
        "engineer",
        "software",
        "programmer"
    ])

    is_manager_role = "manager" in query_lower

    # -----------------------------------------------------
    # EMBEDDING SEARCH
    # -----------------------------------------------------

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, 40)

    scored_results = []

    # =====================================================
    # SCORING LOOP
    # =====================================================

    for rank, idx in enumerate(indices[0]):

        item = data[idx]

        name = item.get("name", "")
        description = item.get("description", "")
        keys = " ".join(item.get("keys", []))

        combined_text = (
            f"{name} {description} {keys}"
        ).lower()

        score = 0

        # =================================================
        # REMOVE IRRELEVANT RESULTS
        # =================================================

        banned_terms = [
            "360",
            "development report",
            "phone solution",
            "global skills development",
            "assessment exercise"
        ]

        if any(term in combined_text for term in banned_terms):
            continue

        # =================================================
        # REMOVE DEVELOPMENT REPORTS
        # =================================================

        development_only_terms = [
            "development action planner",
            "development centers",
            "virtual assessment",
            "coaching report",
            "development report"
        ]

        if is_developer_role:

            if any(term in combined_text for term in development_only_terms):
                continue

        # =================================================
        # REMOVE LEADERSHIP / HIPO REPORTS
        # =================================================

        if is_developer_role:

            excluded_dev_terms = [
                "hipo",
                "high potential",
                "leadership potential",
                "executive",
                "succession"
            ]

            if any(term in combined_text for term in excluded_dev_terms):
                continue

        # =================================================
        # ROLE ALIGNMENT FILTER
        # =================================================

        if is_developer_role:

            developer_terms = [
                "developer",
                "software",
                "coding",
                "programming",
                "technical",
                "java",
                "python",
                "engineering",
                "ability",
                "aptitude",
                "reasoning",
                "verify",
                "opq",
                "occupational personality"
            ]

            if not any(term in combined_text for term in developer_terms):
                continue

        # =================================================
        # KEYWORD MATCHING
        # =================================================

        keywords = query_lower.split()

        for keyword in keywords:

            if keyword in combined_text:
                score += 3

        # =================================================
        # PREFERRED SHL ASSESSMENTS
        # =================================================

        preferred_terms = [
            "opq",
            "verify",
            "general ability",
            "java",
            "core java",
            "occupational personality"
        ]

        for term in preferred_terms:

            if term in combined_text:
                score += 8

        # =================================================
        # TECHNICAL BOOST
        # =================================================

        if needs_technical:

            if any(term in combined_text for term in [
                "java",
                "python",
                "javascript",
                "sql",
                "developer",
                "software",
                "coding",
                "technical"
            ]):
                score += 15

        # =================================================
        # PERSONALITY BOOST
        # =================================================

        if needs_personality:

            preferred_behavioral_terms = [
                "opq",
                "occupational personality",
                "personality questionnaire",
                "workplace behavior"
            ]

            weak_behavioral_terms = [
                "development",
                "development center",
                "action planner",
                "virtual assessment"
            ]

            # Strong hiring-focused behavioral assessments
            if any(term in combined_text for term in preferred_behavioral_terms):
                score += 25

            # Penalize development-focused reports
            if any(term in combined_text for term in weak_behavioral_terms):
                score -= 10

        # =================================================
        # COGNITIVE BOOST
        # =================================================

        if needs_cognitive :

            if any(term in combined_text for term in [
                "ability",
                "aptitude",
                "reasoning",
                "logical",
                "numerical",
                "verbal",
                "cognitive",
                "verify"
            ]):
                score += 15

        # =================================================
        # MANAGEMENT BOOST
        # =================================================

        if is_manager_role:

            if any(term in combined_text for term in [
                "manager",
                "management",
                "leadership",
                "supervisor"
            ]):
                score += 10

        # =================================================
        # EXCLUDE JOB SOLUTIONS
        # =================================================

        if "job solution" in combined_text:
            continue

        # =================================================
        # SEMANTIC SCORE
        # =================================================

        semantic_score = 1 / (1 + distances[0][rank])

        final_score = score + semantic_score

        scored_results.append((final_score, item))

    # =====================================================
    # SORT RESULTS
    # =====================================================

    scored_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    # =====================================================
    # DIVERSITY BALANCING
    # =====================================================

    recommendations = []

    added = set()

    technical_count = 0
    behavioral_count = 0
    cognitive_count = 0

    for score, item in scored_results:

        name = item.get("name")

        if name in added:
            continue

        keys_text = " ".join(item.get("keys", [])).lower()

        is_behavioral = any(term in keys_text for term in [
            "personality",
            "behavior"
        ])

        is_cognitive = any(term in keys_text for term in [
            "ability",
            "aptitude",
            "cognitive"
        ])

        is_technical = any(term in keys_text for term in [
            "knowledge",
            "skills",
            "technical"
        ])

        # -------------------------------------------------
        # CATEGORY LIMITS
        # -------------------------------------------------

        if is_technical and technical_count >= 2:
            continue

        if is_behavioral and behavioral_count >= 2:
            continue

        if is_cognitive and cognitive_count >= 2:
            continue

        # -------------------------------------------------
        # UPDATE COUNTS
        # -------------------------------------------------

        if is_technical:
            technical_count += 1

        if is_behavioral:
            behavioral_count += 1

        if is_cognitive:
            cognitive_count += 1

        # -------------------------------------------------
        # ADD RESULT
        # -------------------------------------------------

        added.add(name)

        recommendations.append({
            "name": name,
            "url": item.get("link"),
            "test_type": ", ".join(item.get("keys", []))
        })

        if len(recommendations) >= top_k:
            break

    return recommendations

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post("/chat")
def chat(req: ChatRequest):

    messages = req.messages

    # =====================================================
    # USER HISTORY
    # =====================================================
    if not req.messages:
        return {
            "reply": "Please provide a hiring requirement.",
        "recommendations": [],
        "end_of_conversation": False
        }
    user_messages = []

    for msg in messages:

        if msg.role == "user":
            user_messages.append(msg.content)

    conversation_text = " ".join(user_messages)

    latest_message = user_messages[-1]

    if any(term in conversation_text.lower() for term in [
    "no preference",
    "anything is fine",
    "any is fine",
    "doesn't matter"]):
        recommendations=search_assessments("general hiring assessment",top_k=5)
        return{
            "reply": (
            "Here are some general SHL assessment recommendations suitable for broad hiring scenarios."
        ),
        "recommendations": recommendations,
        "end_of_conversation": False
        }

    # =====================================================
    # DETECT INTENT
    # =====================================================

    intent = detect_intent(conversation_text)

    # =====================================================
    # REFUSE
    # =====================================================

    if intent == "REFUSE":

        return {
            "reply": (
                "I can only help with SHL assessment recommendations and comparisons."
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # =====================================================
    # COMPARE
    # =====================================================

    if intent == "COMPARE":

        return compare_assessments(conversation_text)

    # =====================================================
    # CLARIFY
    # =====================================================

    if intent == "CLARIFY":

        clarification = generate_clarification(latest_message)

        return {
            "reply": clarification,
            "recommendations": [],
            "end_of_conversation": False
        }

    # =====================================================
    # SEARCH
    # =====================================================

    recommendations = search_assessments(
        conversation_text,
        top_k=5
    )

    # =====================================================
    # LOW CONFIDENCE
    # =====================================================

    if len(recommendations) == 0:

        return {
            "reply": (
                "I could not confidently identify suitable SHL assessments yet. "
                "Could you provide more details about the role or required skills?"
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {
        "reply": (
            "Here are recommended SHL assessments based on your hiring requirements."
        ),
        "recommendations": recommendations,
        "end_of_conversation": False
    }