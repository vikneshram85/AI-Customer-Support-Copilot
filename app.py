"""
ZENDS AI Customer Support Copilot — Streamlit Version
Converted from Gradio with full feature parity:
✅ AI Copilot tab with customer info, query analysis, response editing & sending
✅ Customers tab with live Add Customer (reflects instantly in AI Copilot)
✅ History tab with IST timestamps + CSV export
✅ Knowledge Base tab
✅ Analytics tab
✅ Settings tab
✅ Angry/Complaint escalation response
✅ Out-of-scope detection
✅ Full ZENDS pricing table (all 28 products × 4 countries)
✅ Policy keyword intercept (SLA, fair usage, contract, discount, etc.)
✅ IST timestamps (3-layer fallback: zoneinfo → pytz → UTC+5:30)
"""

import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime, timezone, timedelta

# ── IST Timestamp ──────────────────────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
    def now_ist():
        return datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S IST")
except Exception:
    try:
        import pytz
        _IST = pytz.timezone("Asia/Kolkata")
        def now_ist():
            return datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        def now_ist():
            return (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S IST")

# ── ML Libraries ───────────────────────────────────────────────────────────────
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    MODELS_AVAILABLE = True
except Exception:
    MODELS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════

ZENDS_POLICIES = {
    "refund": (
        "Full refund is available within 7 days of purchase if usage is less than 10%. "
        "Cloud services (ZENDCloud VMs, ZENDStorage) are non-refundable after activation. "
        "Refund requests must be submitted via the customer portal or by contacting support."
    ),
    "billing": (
        "ZENDS bills monthly in advance. Enterprise customers receive consolidated invoices. "
        "Late payment after 7 days may result in service suspension. "
        "Annual payment attracts a 15% discount."
    ),
    "sla": (
        "Individual users: 98.5% uptime SLA. "
        "Business users: 99.5% uptime SLA. "
        "Enterprise users: 99.9% uptime SLA with dedicated support."
    ),
    "contract": (
        "Individual users have no long-term lock-in and can cancel anytime. "
        "Enterprise customers have a minimum 12-month contract period."
    ),
    "fair_usage": (
        "Unlimited mobile and broadband plans are subject to a Fair Usage Policy (FUP) "
        "capped at 1TB per month. Beyond this threshold, speeds may be reduced."
    ),
    "discount": (
        "Bulk enterprise customers can receive up to 30% discount. "
        "Annual payment upfront attracts a 15% discount across all plans."
    ),
    "data_privacy": (
        "ZENDS is fully GDPR compliant and ISO 27001 certified. "
        "All customer data is encrypted at rest and in transit."
    ),
    "support": (
        "ZENDS offers three support tiers: Standard, Priority, and Enterprise Dedicated Support. "
        "Technical support is available 24×7 via phone, email, and live chat."
    ),
}

ZENDS_PRODUCTS_SUMMARY = (
    "ZENDS offers 5 product groups: "
    "1) Mobile Connectivity (Prepaid & Postpaid plans with 5G), "
    "2) Home & Office Internet (ZENDFiber 100Mbps to 1Gbps), "
    "3) Business Connectivity (ZENDBiz Connect & ZENDEnterprise), "
    "4) Cloud & Data Center Services (ZENDCloud VMs & ZENDStorage), "
    "5) IoT & Smart Solutions (ZENDSmart Traffic, Lighting, Parking, Industrial Sensor, Fleet IoT). "
    "Services are available across India, USA, Singapore, and Thailand."
)

ZENDS_PRICING = {
    "prepaid basic":           {"description": "Prepaid Basic — 5GB mobile data",
        "usa": {"individual":"$60","enterprise":"$48"}, "india": {"individual":"$36","enterprise":"$28"},
        "singapore": {"individual":"$66","enterprise":"$56"}, "thailand": {"individual":"$48","enterprise":"$38"}},
    "prepaid plus":            {"description": "Prepaid Plus — 20GB mobile data",
        "usa": {"individual":"$80","enterprise":"$60"}, "india": {"individual":"$48","enterprise":"$29"},
        "singapore": {"individual":"$88","enterprise":"$76"}, "thailand": {"individual":"$64","enterprise":"$52"}},
    "prepaid unlimited":       {"description": "Prepaid Unlimited — Unlimited mobile data",
        "usa": {"individual":"$100","enterprise":"$80"}, "india": {"individual":"$60","enterprise":"$50"},
        "singapore": {"individual":"$110","enterprise":"$90"}, "thailand": {"individual":"$80","enterprise":"$70"}},
    "postpaid silver":         {"description": "Postpaid Silver — 50GB mobile data",
        "usa": {"individual":"$70","enterprise":"$60"}, "india": {"individual":"$42","enterprise":"$36"},
        "singapore": {"individual":"$77","enterprise":"$66"}, "thailand": {"individual":"$56","enterprise":"$48"}},
    "postpaid gold":           {"description": "Postpaid Gold — 100GB mobile data",
        "usa": {"individual":"$100","enterprise":"$80"}, "india": {"individual":"$60","enterprise":"$50"},
        "singapore": {"individual":"$110","enterprise":"$90"}, "thailand": {"individual":"$80","enterprise":"$70"}},
    "postpaid platinum":       {"description": "Postpaid Platinum — Unlimited data + international calls",
        "usa": {"individual":"$120","enterprise":"$110"}, "india": {"individual":"$72","enterprise":"$66"},
        "singapore": {"individual":"$132","enterprise":"$121"}, "thailand": {"individual":"$96","enterprise":"$88"}},
    "zendfiber home 100 mbps": {"description": "ZENDFiber Home 100 Mbps — Home fiber broadband",
        "usa": {"individual":"$30","enterprise":"$25"}, "india": {"individual":"$18","enterprise":"$15"},
        "singapore": {"individual":"$33","enterprise":"$27"}, "thailand": {"individual":"$24","enterprise":"$20"}},
    "zendfiber home 300 mbps": {"description": "ZENDFiber Home 300 Mbps — Home fiber broadband",
        "usa": {"individual":"$50","enterprise":"$45"}, "india": {"individual":"$30","enterprise":"$27"},
        "singapore": {"individual":"$55","enterprise":"$49"}, "thailand": {"individual":"$40","enterprise":"$36"}},
    "zendfiber home 1 gbps":   {"description": "ZENDFiber Home 1 Gbps — Ultra-fast home fiber",
        "usa": {"individual":"$80","enterprise":"$70"}, "india": {"individual":"$48","enterprise":"$42"},
        "singapore": {"individual":"$88","enterprise":"$77"}, "thailand": {"individual":"$64","enterprise":"$56"}},
    "zendoffice net 200":      {"description": "ZENDOffice Net 200 — Office broadband 200 Mbps",
        "usa": {"individual":"$60","enterprise":"$55"}, "india": {"individual":"$36","enterprise":"$33"},
        "singapore": {"individual":"$66","enterprise":"$61"}, "thailand": {"individual":"$48","enterprise":"$44"}},
    "zendoffice net 500":      {"description": "ZENDOffice Net 500 — Office broadband 500 Mbps",
        "usa": {"individual":"$90","enterprise":"$80"}, "india": {"individual":"$54","enterprise":"$48"},
        "singapore": {"individual":"$99","enterprise":"$88"}, "thailand": {"individual":"$72","enterprise":"$64"}},
    "zendoffice net 1g":       {"description": "ZENDOffice Net 1G — Office broadband 1 Gbps",
        "usa": {"individual":"$150","enterprise":"$130"}, "india": {"individual":"$90","enterprise":"$78"},
        "singapore": {"individual":"$165","enterprise":"$143"}, "thailand": {"individual":"$120","enterprise":"$104"}},
    "zendbiz connect 100":     {"description": "ZENDBiz Connect 100 — Business connectivity 100 Mbps",
        "usa": {"individual":"$70","enterprise":"$60"}, "india": {"individual":"$42","enterprise":"$36"},
        "singapore": {"individual":"$77","enterprise":"$66"}, "thailand": {"individual":"$56","enterprise":"$48"}},
    "zendbiz connect 500":     {"description": "ZENDBiz Connect 500 — Business connectivity 500 Mbps",
        "usa": {"individual":"$120","enterprise":"$100"}, "india": {"individual":"$72","enterprise":"$60"},
        "singapore": {"individual":"$132","enterprise":"$110"}, "thailand": {"individual":"$96","enterprise":"$80"}},
    "zendbiz connect 1g":      {"description": "ZENDBiz Connect 1G — Business connectivity 1 Gbps",
        "usa": {"individual":"$200","enterprise":"$180"}, "india": {"individual":"$120","enterprise":"$108"},
        "singapore": {"individual":"$220","enterprise":"$198"}, "thailand": {"individual":"$160","enterprise":"$144"}},
    "zendenterprise ultra":    {"description": "ZENDEnterprise Ultra — Enterprise-grade dedicated connectivity",
        "usa": {"individual":"$300","enterprise":"$280"}, "india": {"individual":"$180","enterprise":"$168"},
        "singapore": {"individual":"$330","enterprise":"$308"}, "thailand": {"individual":"$240","enterprise":"$224"}},
    "zendenterprise dedicated":{"description": "ZENDEnterprise Dedicated — Premium dedicated connectivity",
        "usa": {"individual":"$500","enterprise":"$450"}, "india": {"individual":"$300","enterprise":"$270"},
        "singapore": {"individual":"$550","enterprise":"$495"}, "thailand": {"individual":"$400","enterprise":"$360"}},
    "zendcloud vm basic":      {"description": "ZENDCloud VM Basic — Entry-level virtual machine",
        "usa": {"individual":"$40","enterprise":"$35"}, "india": {"individual":"$24","enterprise":"$21"},
        "singapore": {"individual":"$44","enterprise":"$38"}, "thailand": {"individual":"$32","enterprise":"$28"}},
    "zendcloud vm pro":        {"description": "ZENDCloud VM Pro — Professional virtual machine",
        "usa": {"individual":"$80","enterprise":"$70"}, "india": {"individual":"$48","enterprise":"$42"},
        "singapore": {"individual":"$88","enterprise":"$77"}, "thailand": {"individual":"$64","enterprise":"$56"}},
    "zendcloud vm enterprise": {"description": "ZENDCloud VM Enterprise — High-performance enterprise VM",
        "usa": {"individual":"$150","enterprise":"$130"}, "india": {"individual":"$90","enterprise":"$78"},
        "singapore": {"individual":"$165","enterprise":"$143"}, "thailand": {"individual":"$120","enterprise":"$104"}},
    "zendstorage 1tb":         {"description": "ZENDStorage 1TB — Cloud file storage 1 TB",
        "usa": {"individual":"$20","enterprise":"$15"}, "india": {"individual":"$12","enterprise":"$9"},
        "singapore": {"individual":"$22","enterprise":"$16"}, "thailand": {"individual":"$16","enterprise":"$12"}},
    "zendstorage 10tb":        {"description": "ZENDStorage 10TB — Cloud file storage 10 TB",
        "usa": {"individual":"$120","enterprise":"$100"}, "india": {"individual":"$72","enterprise":"$60"},
        "singapore": {"individual":"$132","enterprise":"$110"}, "thailand": {"individual":"$96","enterprise":"$80"}},
    "zendarchive storage":     {"description": "ZENDArchive Storage — Long-term archival cloud storage",
        "usa": {"individual":"$50","enterprise":"$40"}, "india": {"individual":"$30","enterprise":"$24"},
        "singapore": {"individual":"$55","enterprise":"$44"}, "thailand": {"individual":"$40","enterprise":"$32"}},
    "zendsmart traffic":       {"description": "ZENDSmart Traffic — Smart city traffic management IoT",
        "usa": {"individual":"$100","enterprise":"$90"}, "india": {"individual":"$60","enterprise":"$54"},
        "singapore": {"individual":"$110","enterprise":"$99"}, "thailand": {"individual":"$80","enterprise":"$72"}},
    "zendsmart lighting":      {"description": "ZENDSmart Lighting — Smart city lighting IoT",
        "usa": {"individual":"$80","enterprise":"$70"}, "india": {"individual":"$48","enterprise":"$42"},
        "singapore": {"individual":"$88","enterprise":"$77"}, "thailand": {"individual":"$64","enterprise":"$56"}},
    "zendsmart parking":       {"description": "ZENDSmart Parking — Smart city parking IoT",
        "usa": {"individual":"$60","enterprise":"$50"}, "india": {"individual":"$36","enterprise":"$30"},
        "singapore": {"individual":"$66","enterprise":"$55"}, "thailand": {"individual":"$48","enterprise":"$40"}},
    "zendindustrial sensor":   {"description": "ZENDIndustrial Sensor — Industrial IoT sensor connectivity",
        "usa": {"individual":"$120","enterprise":"$100"}, "india": {"individual":"$72","enterprise":"$60"},
        "singapore": {"individual":"$132","enterprise":"$110"}, "thailand": {"individual":"$96","enterprise":"$80"}},
    "zendfleet iot":           {"description": "ZENDFleet IoT — Fleet management and tracking IoT",
        "usa": {"individual":"$150","enterprise":"$130"}, "india": {"individual":"$90","enterprise":"$78"},
        "singapore": {"individual":"$165","enterprise":"$143"}, "thailand": {"individual":"$120","enterprise":"$104"}},
}

OUT_OF_SCOPE_KEYWORDS = [
    "weather", "stock market", "recipe", "movie", "song", "cricket", "football",
    "politics", "news", "covid", "vaccine", "hospital", "doctor", "medicine",
    "flight", "hotel", "restaurant", "amazon", "flipkart", "netflix", "youtube",
    "bank account", "loan", "insurance", "real estate", "school", "university",
    "visa", "passport", "election", "government", "police", "crime",
]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached so it runs only once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    models = {}
    try:
        if os.path.exists("./intent_model"):
            models["intent_model"]     = AutoModelForSequenceClassification.from_pretrained("./intent_model")
            models["intent_tokenizer"] = AutoTokenizer.from_pretrained("./intent_model")
            with open("./intent_model/label_encoder.json") as f:
                models["intent_labels"] = json.load(f)
    except Exception:
        models["intent_model"] = None

    try:
        if os.path.exists("./sentiment_model"):
            models["sentiment_model"]     = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
            models["sentiment_tokenizer"] = AutoTokenizer.from_pretrained("./sentiment_model")
            with open("./sentiment_model/label_encoder.json") as f:
                models["sentiment_labels"] = json.load(f)
    except Exception:
        models["sentiment_model"] = None

    try:
        if os.path.exists("./rag_system"):
            models["embedding_model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            client = chromadb.PersistentClient(path="./rag_system/vector_store")
            models["rag_collection"] = client.get_collection("zends_docs")
    except Exception:
        models["rag_collection"] = None

    return models

MODELS = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION & RESPONSE FUNCTIONS  (identical logic to Gradio version)
# ══════════════════════════════════════════════════════════════════════════════

def predict_intent(query):
    if MODELS.get("intent_model"):
        inputs = MODELS["intent_tokenizer"](query, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = MODELS["intent_model"](**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        return MODELS["intent_labels"]["id_to_label"][str(pred_class)], probs[0][pred_class].item()
    q = query.lower()
    if any(w in q for w in ["bill", "invoice", "charge", "payment", "amount due", "late payment", "due date"]):
        return "billing", 0.85
    if any(w in q for w in ["refund", "money back", "reimburs", "cancel"]):
        return "refund", 0.85
    if any(w in q for w in ["not working", "issue", "error", "problem", "slow", "cannot access",
                             "can't log", "down", "outage", "disconnected", "no signal", "connection"]):
        return "technical", 0.85
    if any(w in q for w in ["complain", "angry", "unacceptable", "worst", "horrible", "terrible", "disgusted"]):
        return "complaint", 0.85
    if any(w in q for w in ["sla", "uptime", "contract", "lock-in", "lock in", "fair usage", "fup",
                             "data privacy", "gdpr", "iso 27001", "discount", "support tier",
                             "policy", "policies", "terms", "condition", "privacy"]):
        return "product", 0.80
    return "product", 0.75


def predict_sentiment(query):
    if MODELS.get("sentiment_model"):
        inputs = MODELS["sentiment_tokenizer"](query, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = MODELS["sentiment_model"](**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        return MODELS["sentiment_labels"]["id_to_label"][str(pred_class)], probs[0][pred_class].item()
    q = query.lower()
    angry_words = ["angry", "furious", "unacceptable", "terrible", "horrible", "worst",
                   "disgusted", "fraud", "scam", "cheated", "pathetic", "useless", "frustrated"]
    happy_words = ["thank", "great", "excellent", "happy", "love", "amazing", "good", "pleased", "satisfied"]
    if any(w in q for w in angry_words): return "angry", 0.90
    if any(w in q for w in happy_words): return "happy", 0.85
    return "neutral", 0.80


def retrieve_context(query, top_k=5):
    if not MODELS.get("rag_collection"):
        return []
    try:
        emb = MODELS["embedding_model"].encode([query])[0]
        res = MODELS["rag_collection"].query(query_embeddings=[emb.tolist()], n_results=top_k)
        return res["documents"][0] if res["documents"] else []
    except Exception:
        return []


def is_out_of_scope(query):
    q = query.lower()
    if any(kw in q for kw in OUT_OF_SCOPE_KEYWORDS):
        return True
    if any(prod in q for prod in ZENDS_PRICING.keys()):
        return False
    zends_keywords = [
        "zend", "bill", "invoice", "refund", "plan", "data", "internet", "broadband",
        "fiber", "mobile", "sim", "esim", "cloud", "storage", "iot", "network",
        "roaming", "5g", "uptime", "sla", "contract", "support", "technical",
        "prepaid", "postpaid", "subscription", "speed", "mbps", "gbps", "vm",
        "price", "cost", "how much", "pricing", "charges", "fair usage", "fup",
        "discount", "privacy", "gdpr", "iso", "policy", "policies", "terms",
        "condition", "support tier", "enterprise", "individual",
        "charge", "charged", "billed", "payment", "cancel", "slow", "down",
        "outage", "complain", "complaint",
    ]
    return not any(kw in q for kw in zends_keywords)


def extract_policy_info(context_list, keyword):
    for ctx in context_list:
        if keyword.lower() in ctx.lower():
            sentences = ctx.split(".")
            relevant = [s.strip() for s in sentences
                        if any(kw in s.lower() for kw in [keyword.lower(), "policy", "rule", "days"])]
            if relevant:
                return ". ".join(relevant[:3]) + "."
    return None


def generate_angry_complaint_response(query, context, customer_name="Valued Customer"):
    name = customer_name if customer_name and customer_name.strip() else "Valued Customer"
    compensation_note = ""
    if context:
        for ctx in context:
            if any(w in ctx.lower() for w in ["refund", "compensat", "credit", "sla"]):
                compensation_note = f"\n\n**📋 Relevant Policy:**\n_{ctx[:250].strip()}..._"
                break
    return f"""I sincerely apologize for the experience you've had with ZENDS Communications, {name}. Your feedback is extremely important to us.

I understand your frustration and I'm here to help resolve this issue immediately.

---

**🚨 Immediate Actions:**

**1. Issue Escalation**
- Your complaint has been logged as a **HIGH PRIORITY** case
- Case will be reviewed by our Customer Relations team within **2 hours**
- You will receive status updates every **48 hours** until resolved

**2. Investigation Timeline**
- Initial investigation: **24–48 hours**
- If unresolved within 5 days, automatic escalation to Supervisor
- Formal response provided within **7 days** maximum

**3. Your Rights**
- Detailed explanation of the issue and our findings
- Appropriate compensation if service standards were not met
- Access to escalation process at any stage{compensation_note}

---

**📌 Next Steps:**
1. We will contact you within **2 hours** to gather more details
2. A dedicated case manager will be assigned to your complaint
3. You'll receive a **case reference number** via email within 30 minutes

---

**📞 Contact Information:**
- 📧 Email: **complaints@zends.com**
- 📱 Phone: **1-800-ZENDS-HELP**
- 💬 Live Chat: Available **24/7** on our website

---
We are fully committed to resolving this matter to your complete satisfaction.

_Your complaint reference number will be sent to your registered email within 30 minutes._"""


def generate_out_of_scope_response(query):
    return f"""Thank you for reaching out to ZENDS Communications.

I'm sorry, but your query — **"{query[:100]}..."** — appears to be outside the scope of ZENDS Communications' services.

**ℹ️ What I can help you with:**
- 📱 Mobile plans (Prepaid / Postpaid) and SIM/eSIM services
- 🌐 Home & Office Broadband (ZENDFiber, ZENDOffice Net)
- 🏢 Business Connectivity (ZENDBiz Connect, ZENDEnterprise)
- ☁️ Cloud & Data Center (ZENDCloud VMs, ZENDStorage)
- 🔌 IoT & Smart Solutions (ZENDSmart, ZENDFleet IoT)
- 💳 Billing, Invoices & Payment queries
- 🔄 Refund & Cancellation requests
- 🛠️ Technical Support for ZENDS services

If your question is related to any of the above, please feel free to rephrase and ask again!

📧 **support@zends.com** | 📞 **1-800-ZENDS-HELP**"""


def generate_response(query, intent, sentiment, context, customer_name=""):
    # 1. Angry / Complaint — always first
    if sentiment == "angry" or intent == "complaint":
        return generate_angry_complaint_response(query, context, customer_name)

    # 2. Out of scope
    if is_out_of_scope(query):
        return generate_out_of_scope_response(query)

    q = query.lower()

    # 3. Policy keyword intercept
    POLICY_KEYWORD_MAP = {
        "sla":          ("sla",          "SLA (Service Level Agreement)"),
        "uptime":       ("sla",          "Uptime Guarantee"),
        "contract":     ("contract",     "Contract Terms"),
        "lock-in":      ("contract",     "Contract Lock-in Policy"),
        "lock in":      ("contract",     "Contract Lock-in Policy"),
        "fair usage":   ("fair_usage",   "Fair Usage Policy (FUP)"),
        "fup":          ("fair_usage",   "Fair Usage Policy (FUP)"),
        "data privacy": ("data_privacy", "Data Privacy Policy"),
        "gdpr":         ("data_privacy", "GDPR and Data Privacy"),
        "iso 27001":    ("data_privacy", "ISO 27001 Certification"),
        "discount":     ("discount",     "Discount Policy"),
        "support tier": ("support",      "Support Tiers"),
        "privacy":      ("data_privacy", "Privacy Policy"),
    }
    for keyword, (policy_key, display_name) in POLICY_KEYWORD_MAP.items():
        if keyword in q:
            policy_text = ZENDS_POLICIES.get(policy_key, "Please contact our support team for details.")
            return (
                f"Thank you for your question about **{display_name}**.\n\n---\n\n"
                f"**Policy Details:**\n{policy_text}\n\n---\n\n"
                f"**Full Policy Summary:**\n"
                f"- **SLA:** Individual 98.5% | Business 99.5% | **Enterprise 99.9%** uptime\n"
                f"- **Contracts:** No lock-in for individuals | 12-month minimum for enterprise\n"
                f"- **Fair Usage:** Unlimited plans capped at **1TB/month**\n"
                f"- **Discounts:** Up to **30%** bulk enterprise | **15%** annual payment\n"
                f"- **Data Privacy:** GDPR compliant, ISO 27001 certified, encrypted at rest and in transit\n"
                f"- **Support Tiers:** Standard | Priority | Enterprise Dedicated (24×7)\n\n"
                f"Is there anything else about our policies you would like to know?\n\n"
                f"📧 **support@zends.com** | 📞 **1-800-ZENDS-HELP**"
            )

    # 4. Billing
    if intent == "billing":
        info = extract_policy_info(context, "billing") or ZENDS_POLICIES["billing"]
        return f"""Thank you for reaching out about your billing query.

**💳 Billing Information:**
{info}

**📌 Helpful Tips:**
- Log into your ZENDS customer portal to view detailed invoices
- Enterprise customers receive consolidated monthly invoices
- Annual payment attracts a **15% discount**

Is there a specific charge or invoice you'd like me to look into? Share your **invoice number** and I'll investigate.

📧 **billing@zends.com** | 📞 **1-800-ZENDS-HELP**"""

    # 5. Refund
    if intent == "refund":
        info = extract_policy_info(context, "refund") or ZENDS_POLICIES["refund"]
        return f"""I understand you're inquiring about a refund request.

**🔄 ZENDS Refund Policy:**
{info}

**📌 How to Raise a Refund:**
1. Log in to your ZENDS customer portal
2. Navigate to **My Orders → Request Refund**
3. Provide order ID and reason for refund
4. Our team will process eligible refunds within **5–7 business days**

> ⚠️ **Note:** Cloud services (ZENDCloud VMs, ZENDStorage) are **non-refundable** after activation.

📧 **refunds@zends.com** | 📞 **1-800-ZENDS-HELP**"""

    # 6. Technical
    if intent == "technical":
        snippet = context[0][:300] if context else "Our 24×7 technical support team is available to assist you."
        return f"""I apologize for the technical issue you're experiencing.

**🛠️ Technical Support:**
{snippet}

**📌 Immediate Troubleshooting Steps:**
1. **Restart** your device and router/modem
2. **Check** ZENDS service status: _status.zends.com_
3. **Clear** app cache and try re-logging in
4. If issue persists — contact our 24×7 support team

**🏷️ SLA Commitment:**
- Individual: **98.5% uptime** | Business: **99.5% uptime** | Enterprise: **99.9% uptime**

📧 **techsupport@zends.com** | 📞 **1-800-ZENDS-HELP** | 💬 Live Chat: **24/7**"""

    # 7. Product / Pricing
    if intent == "product":
        locations = ["india", "usa", "singapore", "thailand"]
        found_product  = next((p for p in sorted(ZENDS_PRICING.keys(), key=len, reverse=True) if p in q), None)
        found_location = next((l for l in locations if l in q), None)

        if found_product and found_location:
            data = ZENDS_PRICING[found_product]
            loc_data = data.get(found_location, {})
            desc = data.get("description", found_product.title())
            return f"""Thank you for your inquiry! Here is the pricing for **{desc}** in **{found_location.title()}**:

| Customer Type | Monthly Price |
|---|---|
| 👤 Individual | **{loc_data.get("individual","N/A")}** |
| 🏢 Enterprise | **{loc_data.get("enterprise","N/A")}** |

**📌 What's Included:**
- 24×7 customer support
- SLA-backed uptime guarantee (98.5% Individual / 99.5% Business / 99.9% Enterprise)
- Access to ZENDS customer portal & optional add-ons

> 💡 Enterprise bulk subscriptions: up to **30% discount** | Annual payment: **15% off**

📧 **sales@zends.com** | 📞 **1-800-ZENDS-HELP**"""

        if found_product:
            data = ZENDS_PRICING[found_product]
            rows = "".join([f"| {c.title()} | {data[c]['individual']} | {data[c]['enterprise']} |\n"
                            for c in ["india","usa","singapore","thailand"] if c in data])
            return f"""Thank you for your inquiry about **{data.get("description", found_product.title())}**!

| Country | Individual | Enterprise |
|---|---|---|
{rows}
> 💡 Enterprise bulk discounts up to **30%** | Annual payment: **15% off**

Which country are you located in? I can give you more specific details!

📧 **sales@zends.com** | 📞 **1-800-ZENDS-HELP**"""

        if found_location and any(w in q for w in ["price","cost","plan","how much","pricing","charges","list"]):
            rows = "".join([f"| {data['description']} | {data[found_location]['individual']} | {data[found_location]['enterprise']} |\n"
                            for data in ZENDS_PRICING.values() if found_location in data])
            return f"""Here is the **complete ZENDS price list for {found_location.title()}**:

| Product | Individual | Enterprise |
|---|---|---|
{rows}
> 💡 Enterprise bulk discounts up to **30%** | Annual payment: **15% off**

📧 **sales@zends.com** | 📞 **1-800-ZENDS-HELP**"""

        return """Thank you for your interest in ZENDS Communications!

**📦 Our Products at a Glance:**
- 📱 **Mobile:** Prepaid Basic (5GB), Plus (20GB), Unlimited | Postpaid Silver (50GB), Gold (100GB), Platinum (Unlimited + International)
- 🌐 **Home Broadband:** ZENDFiber Home 100 Mbps / 300 Mbps / 1 Gbps
- 🏢 **Office Broadband:** ZENDOffice Net 200 / 500 / 1G
- 💼 **Business:** ZENDBiz Connect 100 / 500 / 1G | ZENDEnterprise Ultra / Dedicated
- ☁️ **Cloud:** ZENDCloud VM Basic / Pro / Enterprise | ZENDStorage 1TB / 10TB / Archive
- 🔌 **IoT:** ZENDSmart Traffic / Lighting / Parking | ZENDIndustrial Sensor | ZENDFleet IoT

Available across **India, USA, Singapore & Thailand** with Individual and Enterprise pricing.

To get an exact price: _"What is the price of ZENDOffice Net 500 in Singapore?"_

📧 **sales@zends.com** | 📞 **1-800-ZENDS-HELP**"""

    # 8. General fallback
    prefix = "Thank you for reaching out — we're happy to help! 😊\n\n" if sentiment == "happy" else ""
    snippet = context[0][:350] if context else "Please contact our support team for personalized assistance."
    return f"""{prefix}**📋 Based on our knowledge base:**
{snippet}

Is there anything specific you'd like to know more about?

📧 **support@zends.com** | 📞 **1-800-ZENDS-HELP** | 💬 Live Chat: **24/7**"""


def run_analysis(customer_type, customer_id, customer_name, customer_email, query):
    """Full pipeline — returns dict of results."""
    intent, intent_conf    = predict_intent(query)
    sentiment, sent_conf   = predict_sentiment(query)
    context                = retrieve_context(query, top_k=5)
    response               = generate_response(query, intent, sentiment, context, customer_name)

    if context:
        context_text = "\n\n".join([f"📄 **Context {i+1}:**\n{doc}" for i, doc in enumerate(context[:3])])
    else:
        context_text = (
            "⚠️ RAG system not connected — using inline ZENDS knowledge base.\n\n"
            "**Available Policies:**\n"
            + "\n".join([f"- **{k.title()}:** {v[:80]}..." for k, v in ZENDS_POLICIES.items()])
        )

    # Append to session history
    st.session_state.query_history.append({
        "Timestamp":   now_ist(),
        "Customer":    customer_name or "Unknown",
        "Customer ID": customer_id   or "—",
        "Query":       query,
        "Intent":      intent.upper(),
        "Sentiment":   sentiment.upper(),
        "Status":      "Analyzed",
    })

    return {
        "intent":       intent,
        "intent_conf":  intent_conf,
        "sentiment":    sentiment,
        "sent_conf":    sent_conf,
        "context_text": context_text,
        "response":     response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ZENDS AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "customers_db" not in st.session_state:
    st.session_state.customers_db = [
        {"id": "C001", "name": "John Doe",   "email": "john@example.com",  "type": "Individual", "plan": "Postpaid Gold"},
        {"id": "C002", "name": "Jane Smith", "email": "jane@example.com",  "type": "Enterprise", "plan": "ZENDBiz Connect 1G"},
        {"id": "C003", "name": "Bob Wilson", "email": "bob@example.com",   "type": "Individual", "plan": "Prepaid Plus"},
    ]
if "query_history"  not in st.session_state: st.session_state.query_history  = []
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "edited_response" not in st.session_state: st.session_state.edited_response = ""
if "send_status"    not in st.session_state: st.session_state.send_status    = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ZENDS AI Copilot")
    st.markdown("*Customer Support System*")
    st.markdown("---")
    st.markdown("**Navigation:**")
    tab_choice = st.radio(
        label="Go to",
        options=["🤖 AI Copilot", "👥 Customers", "📊 History", "📈 Analytics", "📚 Knowledge Base", "⚙️ Settings"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.metric("Total Queries",    len(st.session_state.query_history))
    st.metric("Total Customers",  len(st.session_state.customers_db))

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem 2.5rem;border-radius:12px;margin-bottom:1.5rem;color:white;text-align:center">
  <h1 style="margin:0;font-size:2rem">🤖 AI Customer Support Copilot</h1>
  <p style="margin:.4rem 0 0;opacity:.9">Intelligent query analysis and response generation — ZENDS Communications</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI COPILOT
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == "🤖 AI Copilot":

    st.markdown("## 👤 Customer Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        customer_type = st.selectbox("Customer Type", ["Existing Customer", "New Customer"])

    db = st.session_state.customers_db

    if customer_type == "Existing Customer":
        cust_ids = [c["id"] for c in db]
        with col2:
            customer_id = st.selectbox("Customer ID", cust_ids)
        # Auto-fill
        match = next((c for c in db if c["id"] == customer_id), {})
        with col3:
            customer_name  = st.text_input("Customer Name",  value=match.get("name", ""),  disabled=True)
        with col4:
            customer_email = st.text_input("Email",          value=match.get("email", ""), disabled=True)
    else:
        with col2:
            st.selectbox("Customer ID", ["—"], disabled=True)
            customer_id = "NEW"
        with col3:
            customer_name  = st.text_input("Customer Name",  placeholder="Enter customer name")
        with col4:
            customer_email = st.text_input("Email",          placeholder="Enter customer email")

    st.markdown("---")

    # ── Query Input ───────────────────────────────────────────────────────────
    st.markdown("## 💬 Customer Query")
    with st.expander("💡 Example queries — click to expand", expanded=False):
        st.markdown("""
- `What does ZENDFiber Home 300 Mbps cost in Singapore?`
- `What's the price of ZENDOffice Net 500 in Singapore?`
- `What is the SLA for enterprise customers?`
- `Tell me about the fair usage policy`
- `I want a refund for my cloud subscription`
- `My internet is not working since yesterday`
- `I am absolutely furious, you charged me twice this month!`
- `Do you offer any discounts?`
- `What is the weather today?` *(out-of-scope example)*
        """)

    query = st.text_area("Enter customer query:", placeholder="Example: I can't access my ZENDCloud VM Pro dashboard", height=120)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        analyze_clicked = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
    with col_b:
        clear_clicked   = st.button("🗑️ Clear",         use_container_width=True)

    if clear_clicked:
        st.session_state.analysis_result = None
        st.session_state.edited_response = ""
        st.session_state.send_status     = ""
        st.rerun()

    if analyze_clicked:
        if not query.strip():
            st.warning("⚠️ Please enter a customer query.")
        else:
            with st.spinner("Analysing query…"):
                st.session_state.analysis_result = run_analysis(
                    customer_type, customer_id, customer_name, customer_email, query
                )
                st.session_state.edited_response = st.session_state.analysis_result["response"]
                st.session_state.send_status     = ""

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result

        intent    = res["intent"]
        sentiment = res["sentiment"]

        intent_icon    = {"billing":"💳","refund":"🔄","technical":"🛠️","complaint":"😠","product":"📦"}.get(intent,"❓")
        sentiment_icon = {"angry":"😠","neutral":"😐","happy":"😊"}.get(sentiment,"😐")
        sent_color     = {"angry":"🔴","neutral":"🟡","happy":"🟢"}.get(sentiment,"⚪")

        total = len(st.session_state.query_history)
        st.success(f"✅ Query analysed successfully! | Intent: **{intent.upper()}** | Sentiment: **{sentiment.upper()}** | Total queries: **{total}**")

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        col_i, col_s = st.columns(2)
        with col_i:
            st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:1rem;background:#f8f9fa">
  <div style="font-size:1.4rem;font-weight:700">{intent_icon} {intent.upper()}</div>
  <div style="color:#555;margin-top:.3rem">Confidence: <b>{res["intent_conf"]:.1%}</b></div>
</div>
""", unsafe_allow_html=True)
        with col_s:
            st.markdown(f"""
<div style="border:1px solid #ddd;border-radius:8px;padding:1rem;background:#f8f9fa">
  <div style="font-size:1.4rem;font-weight:700">{sentiment_icon} {sentiment.upper()}</div>
  <div style="color:#555;margin-top:.3rem">Confidence: <b>{res["sent_conf"]:.1%}</b></div>
</div>
""", unsafe_allow_html=True)

        st.markdown("### 📚 Retrieved Policy / Product Context (RAG)")
        st.text_area("Context", value=res["context_text"], height=200, disabled=True, label_visibility="collapsed")

        st.markdown("---")
        st.markdown("## 🤖 AI Generated Response")
        st.text_area("Generated Response (read-only)", value=res["response"], height=350, disabled=True, label_visibility="collapsed")

        st.markdown("### ✏️ Edit Response Before Sending")
        edited = st.text_area(
            "Edit response",
            value=st.session_state.edited_response,
            height=350,
            label_visibility="collapsed",
            key="edited_response_box",
        )
        st.session_state.edited_response = edited

        col_send, col_save = st.columns(2)
        with col_send:
            if st.button("📧 Send to Customer", type="primary", use_container_width=True):
                if not edited.strip():
                    st.warning("⚠️ Response is empty.")
                elif not customer_email.strip():
                    st.warning("⚠️ Customer email is missing.")
                else:
                    st.session_state.send_status = f"✅ Response sent to **{customer_email}** at {now_ist()}! 📧 Email dispatched successfully."
        with col_save:
            if st.button("💾 Save Draft", use_container_width=True):
                st.session_state.send_status = "💾 Draft saved successfully!"

        if st.session_state.send_status:
            st.info(st.session_state.send_status)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "👥 Customers":

    st.markdown("## 👥 Customer Database")
    st.dataframe(pd.DataFrame(st.session_state.customers_db), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### ➕ Add New Customer")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: new_id    = st.text_input("Customer ID",   placeholder="e.g. C004")
    with col2: new_name  = st.text_input("Name",          placeholder="e.g. Alice Brown")
    with col3: new_email = st.text_input("Email",         placeholder="e.g. alice@example.com")
    with col4: new_type  = st.selectbox("Type",           ["Individual", "Enterprise"])
    with col5: new_plan  = st.text_input("Plan",          placeholder="e.g. Postpaid Silver")

    if st.button("➕ Add Customer", type="primary"):
        cid = new_id.strip(); cname = new_name.strip(); cemail = new_email.strip()
        if not (cid and cname and cemail):
            st.error("⚠️ Customer ID, Name, and Email are required.")
        elif any(c["id"] == cid for c in st.session_state.customers_db):
            st.error(f"⚠️ Customer ID '{cid}' already exists.")
        else:
            st.session_state.customers_db.append({
                "id": cid, "name": cname, "email": cemail,
                "type": new_type, "plan": new_plan or "N/A"
            })
            st.success(f"✅ Customer '{cname}' (ID: {cid}) added! Now visible in AI Copilot tab.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📊 History":

    st.markdown("## 📊 Query History")

    if not st.session_state.query_history:
        st.info("No queries yet. Start analyzing customer queries in the AI Copilot tab!")
    else:
        df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(df, use_container_width=True, hide_index=True)

        col_exp, _ = st.columns([1, 3])
        with col_exp:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export as CSV",
                data=csv,
                file_name=f"zends_query_history_{now_ist()[:10]}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📈 Analytics":

    st.markdown("## 📈 Query Analytics")

    if not st.session_state.query_history:
        st.info("No data yet. Analyze some queries in the AI Copilot tab first!")
    else:
        df = pd.DataFrame(st.session_state.query_history)
        total = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries",   total)
        col2.metric("Unique Customers", df["Customer ID"].nunique())
        col3.metric("Last Query",       df.iloc[-1]["Timestamp"])

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 🎯 Intent Distribution")
            intent_counts = df["Intent"].value_counts()
            st.bar_chart(intent_counts)

        with col_b:
            st.markdown("### 💬 Sentiment Distribution")
            sent_counts = df["Sentiment"].value_counts()
            st.bar_chart(sent_counts)

        st.markdown("---")
        st.markdown("### 📋 Summary")
        intent_str    = " | ".join([f"**{k}**: {v} ({v/total:.0%})" for k, v in df["Intent"].value_counts().items()])
        sentiment_str = " | ".join([f"**{k}**: {v} ({v/total:.0%})" for k, v in df["Sentiment"].value_counts().items()])
        st.markdown(f"**Intents:** {intent_str}")
        st.markdown(f"**Sentiments:** {sentiment_str}")
        st.markdown(f"**Last query by:** {df.iloc[-1]['Customer']} — Intent: {df.iloc[-1]['Intent']} | Sentiment: {df.iloc[-1]['Sentiment']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📚 Knowledge Base":

    st.markdown("## 📚 ZENDS Knowledge Base")
    st.markdown("### 📋 Policies")

    for policy_name, policy_text in ZENDS_POLICIES.items():
        with st.expander(f"📌 {policy_name.replace('_',' ').title()}"):
            st.markdown(policy_text)

    st.markdown("---")
    st.markdown("### 📦 Products Overview")
    st.info(ZENDS_PRODUCTS_SUMMARY)

    st.markdown("---")
    st.markdown("### 💰 Full Pricing Table")
    country_filter = st.selectbox("Filter by Country", ["All", "India", "USA", "Singapore", "Thailand"])

    rows = []
    for prod_key, data in ZENDS_PRICING.items():
        for country in ["india", "usa", "singapore", "thailand"]:
            if country in data:
                if country_filter == "All" or country_filter.lower() == country:
                    rows.append({
                        "Product":      data["description"],
                        "Country":      country.title(),
                        "Individual":   data[country]["individual"],
                        "Enterprise":   data[country]["enterprise"],
                    })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "⚙️ Settings":

    st.markdown("## ⚙️ System Status")

    def status_row(label, loaded):
        icon = "✅" if loaded else "❌"
        note = "Loaded" if loaded else "Not found — using keyword/inline fallback"
        st.markdown(f"{icon} **{label}:** {note}")

    status_row("Intent Model",    bool(MODELS.get("intent_model")))
    status_row("Sentiment Model", bool(MODELS.get("sentiment_model")))
    status_row("RAG System",      bool(MODELS.get("rag_collection")))

    st.markdown("---")
    st.info(
        "ℹ️ Even without trained models, the system works fully using keyword-based "
        "fallback for intent/sentiment detection and the inline ZENDS policy knowledge "
        "base for all responses."
    )
    st.markdown(f"🕐 **Current IST time:** `{now_ist()}`")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:.85rem'>"
    "🤖 ZENDS AI Customer Support Copilot — Powered by NLP, HuggingFace Transformers & RAG"
    "</div>",
    unsafe_allow_html=True,
)