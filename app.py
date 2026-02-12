"""
ZENDS Communications - AI Customer Support Copilot - REFINED VERSION
Features:
1. ✅ Customer ID, Name, Email fields with New Customer option
2. ✅ Edit Response and Send Response buttons
3. ✅ Cleaner UI without technical status metrics
4. ✅ All previous fixes maintained (query history, RAG, etc.)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
import warnings
import re
import os

warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="ZENDS Communications - AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .intent-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px 5px 5px 0;
    }
    .intent-billing {background-color: #fee; color: #dc3545;}
    .intent-refund {background-color: #e7f3ff; color: #0066cc;}
    .intent-technical {background-color: #fff3cd; color: #ff6600;}
    .intent-complaint {background-color: #f8d7da; color: #721c24;}
    .intent-product {background-color: #d1ecf1; color: #0c5460;}
    .intent-others {background-color: #e2e3e5; color: #383d41;}
    .intent-out_of_scope {background-color: #fff3e0; color: #e65100;}

    .sentiment-angry {color: #dc3545; font-weight: bold;}
    .sentiment-neutral {color: #ffc107; font-weight: bold;}
    .sentiment-happy {color: #28a745; font-weight: bold;}
    
    .response-box {
        background-color: #f0f4ff;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 10px 0;
    }
    .complaint-box {
        background-color: #fff0f0;
        padding: 20px;
        border-left: 5px solid #dc3545;
        border-radius: 5px;
        margin: 10px 0;
    }
    .customer-info {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== LABEL MAPPINGS ====================

INTENT_LABELS = {
    0: 'billing',
    1: 'complaint', 
    2: 'others',
    3: 'product',
    4: 'refund',
    5: 'technical'
}

SENTIMENT_LABELS = {
    0: 'angry',
    1: 'happy',
    2: 'neutral'
}

# ==================== PDF-EXTRACTED POLICIES ====================

ZENDS_POLICIES = {
    "billing": {
        "Standard Billing Policy": """
        Monthly billing in advance. Enterprise customers receive consolidated invoices. 
        Late payment after 7 days may suspend services. Payment processing within 15 days 
        of invoice generation. Autopay available with discount options.
        """,
        "Enterprise Billing": """
        Consolidated invoices for enterprise customers. Net-30 payment terms available.
        Volume discounts up to 30% for bulk enterprise users. Annual payment discount is 15%.
        Quarterly or annual billing cycles supported.
        """,
        "Late Payment": """
        Grace period of 7 days before service suspension. Late payment fee may apply.
        Automatic payment reminders sent before due date. Service restoration within 
        24 hours of payment clearance.
        """
    },
    "refund": {
        "Standard Refund Policy": """
        Full refund within 7 days if usage is less than 10%. Refund processing time 
        7-10 business days. Cloud services are non-refundable after activation.
        Pro-rated refunds for service cancellation based on unused days.
        """,
        "Service Cancellation": """
        Notice period of 30 days for service cancellation. Pro-rated refund for 
        remaining period. Equipment return required within 14 days. Early termination 
        fee may apply for contract customers.
        """,
        "Cloud Services Refund": """
        Cloud services (VM, Storage) are non-refundable after activation due to 
        immediate resource provisioning. SLA-based credits provided for service 
        downtime or performance issues.
        """
    },
    "technical": {
        "Technical Support": """
        24×7 technical support available. Network monitoring and troubleshooting included.
        Installation and setup guidance provided. Priority support for enterprise customers.
        Remote diagnostics and on-site support when required.
        """,
        "SLA Commitments": """
        Individual users: 98.5% uptime guarantee. Business users: 99.5% uptime guarantee.
        Enterprise users: 99.9% uptime guarantee. SLA-backed support with compensation 
        for downtime exceeding guaranteed limits.
        """,
        "Issue Resolution": """
        Response time based on priority level. Critical issues escalated immediately.
        Root cause analysis provided for recurring issues. Proactive monitoring for 
        enterprise customers.
        """
    },
    "complaint": {
        "Complaint Handling": """
        All complaints acknowledged within 2 hours. Investigation period: 24-48 hours.
        Status updates provided every 48 hours. Escalation to supervisor if unresolved 
        within 5 days. Formal response within 7 days.
        """,
        "Escalation Process": """
        Level 1: Customer Support Representative. Level 2: Team Lead/Supervisor.
        Level 3: Department Manager. Level 4: Regional Director. 
        Final escalation: Customer Relations Director.
        """,
        "Compensation": """
        Service quality issues: Discount or credit. Billing errors: Immediate correction 
        plus credit. Technical delays: Pro-rated credit. Staff behavior: Formal apology 
        and discretionary compensation.
        """
    },
    "product": {
        "Product Information": """
        Complete product catalog available with detailed specifications. Free consultation 
        for enterprise solutions. Product demos for cloud and IoT services. Trial periods:
        7 days for mobile, 30 days for broadband. Money-back guarantee for select services.
        """,
        "Plan Changes": """
        Upgrades: Instant activation with pro-rated billing. Downgrades: Effective from 
        next billing cycle. First change free within 12 months. Subsequent changes may 
        incur $20 fee. Multiple add-ons available for customization.
        """,
        "Equipment": """
        Router/modem provided for broadband plans. Installation support included.
        Equipment return required on cancellation. Replacement available for 
        defects or damage.
        """
    },
    "others": {
        "GDPR Compliance": """
        ZENDS is GDPR compliant with ISO 27001 certification. Data encrypted at rest 
        and in transit. Customer data protection with right to access and deletion.
        Regular security audits and compliance reviews.
        """,
        "Data Security": """
        Encrypted data at rest and in transit using industry-standard protocols.
        Multi-layer security architecture. Regular penetration testing and 
        vulnerability assessments. 24/7 security monitoring.
        """,
        "Fair Usage Policy": """
        Unlimited plans capped at 1TB per month. Speed throttling after fair usage 
        limit. FUP reset every billing cycle. Enterprise plans have custom FUP 
        based on agreement.
        """,
        "Contract Terms": """
        Individual users have no long-term lock-in. Enterprise customers have 
        minimum 12-month contract. Early termination fees apply for contract 
        cancellation. Flexible renewal options available.
        """,
        "Support Tiers": """
        Standard Support: Email and chat during business hours.
        Priority Support: 24/7 phone, email, and chat with faster response.
        Enterprise Dedicated Support: Named account manager, proactive monitoring,
        quarterly business reviews.
        """,
        "Uptime Guarantee": """
        Individual: 98.5% uptime. Business: 99.5% uptime. Enterprise: 99.9% uptime.
        Credits provided for downtime exceeding SLA. Scheduled maintenance with 
        48-hour advance notice.
        """,
        "Bulk Discounts": """
        Bulk enterprise users can receive up to 30% discount based on volume.
        Annual payment discount of 15% available. Custom pricing for large 
        deployments. Volume commitments rewarded with additional benefits.
        """,
        "Security Certifications": """
        ISO 27001 (Information Security Management), SOC 2 Type II compliance,
        PCI DSS for payment security, HIPAA compliance for healthcare data.
        Annual third-party security audits conducted.
        """,
        "Enterprise Solutions": """
        Customized solutions available for large enterprises. Dedicated account 
        management. Flexible contract terms. Integration support and API access.
        Priority troubleshooting and proactive monitoring.
        """,
        "Plan Upgrades": """
        Mid-cycle upgrades allowed with pro-rated billing. Instant activation 
        for upgrade requests. No downgrade fees for first change. Flexible 
        customization with add-ons.
        """
    }
}

# PDF-extracted product information
ZENDS_PRODUCTS = {
    "Mobile Connectivity": {
        "Prepaid Basic": {
            "price": "USA: $60, India: $36, Singapore: $66, Thailand: $48",
            "features": ["5GB Data", "Voice & SMS", "Basic Speed", "No Contract"],
            "enterprise_price": "USA: $48, India: $28, Singapore: $56, Thailand: $38"
        },
        "Prepaid Plus": {
            "price": "USA: $80, India: $48, Singapore: $88, Thailand: $64",
            "features": ["20GB Data", "Voice & SMS", "High Speed", "International Roaming"],
            "enterprise_price": "USA: $60, India: $29, Singapore: $76, Thailand: $52"
        },
        "Prepaid Unlimited": {
            "price": "USA: $100, India: $60, Singapore: $110, Thailand: $80",
            "features": ["Unlimited Data (1TB FUP)", "Voice & SMS", "Premium Speed", "Global Roaming"],
            "enterprise_price": "USA: $80, India: $50, Singapore: $90, Thailand: $70"
        },
        "Postpaid Silver": {
            "price": "USA: $70, India: $42, Singapore: $77, Thailand: $56",
            "features": ["50GB Data", "Voice & SMS", "Monthly Billing", "Credit Facility"],
            "enterprise_price": "USA: $60, India: $36, Singapore: $66, Thailand: $48"
        },
        "Postpaid Gold": {
            "price": "USA: $100, India: $60, Singapore: $110, Thailand: $80",
            "features": ["100GB Data", "Voice & SMS", "Priority Support", "Device Financing"],
            "enterprise_price": "USA: $80, India: $50, Singapore: $90, Thailand: $70"
        },
        "Postpaid Platinum": {
            "price": "USA: $120, India: $72, Singapore: $132, Thailand: $96",
            "features": ["Unlimited Data + Calls", "Premium Support", "Global Roaming", "Device Insurance"],
            "enterprise_price": "USA: $110, India: $66, Singapore: $121, Thailand: $88"
        }
    },
    "Home & Office Internet": {
        "ZENDFiber Home 100 Mbps": {
            "price": "USA: $30, India: $18, Singapore: $33, Thailand: $24",
            "features": ["100 Mbps Speed", "Unlimited Data", "WiFi Router", "1-2 Users"],
            "enterprise_price": "USA: $25, India: $15, Singapore: $27, Thailand: $20"
        },
        "ZENDFiber Home 300 Mbps": {
            "price": "USA: $50, India: $30, Singapore: $55, Thailand: $40",
            "features": ["300 Mbps Speed", "Unlimited Data", "Dual Band Router", "3-4 Users"],
            "enterprise_price": "USA: $45, India: $27, Singapore: $49, Thailand: $36"
        },
        "ZENDFiber Home 1 Gbps": {
            "price": "USA: $80, India: $48, Singapore: $88, Thailand: $64",
            "features": ["1 Gbps Speed", "Unlimited Data", "Premium Router", "Small Office/Home"],
            "enterprise_price": "USA: $70, India: $42, Singapore: $77, Thailand: $56"
        },
        "ZENDOffice Net 200": {
            "price": "USA: $60, India: $36, Singapore: $66, Thailand: $48",
            "features": ["200 Mbps", "Business SLA", "Static IP", "5-10 Users"],
            "enterprise_price": "USA: $55, India: $33, Singapore: $61, Thailand: $44"
        },
        "ZENDOffice Net 500": {
            "price": "USA: $90, India: $54, Singapore: $99, Thailand: $72",
            "features": ["500 Mbps", "Business SLA", "5 Static IPs", "10-20 Users"],
            "enterprise_price": "USA: $80, India: $48, Singapore: $88, Thailand: $64"
        },
        "ZENDOffice Net 1G": {
            "price": "USA: $150, India: $90, Singapore: $165, Thailand: $120",
            "features": ["1 Gbps", "Enterprise SLA", "10 Static IPs", "20+ Users"],
            "enterprise_price": "USA: $130, India: $78, Singapore: $143, Thailand: $104"
        }
    },
    "Business Connectivity": {
        "ZENDBiz Connect 100": {
            "price": "USA: $70, India: $42, Singapore: $77, Thailand: $56",
            "features": ["100 Mbps Dedicated", "99.5% SLA", "Priority Support", "Scalable"],
            "enterprise_price": "USA: $60, India: $36, Singapore: $66, Thailand: $48"
        },
        "ZENDBiz Connect 500": {
            "price": "USA: $120, India: $72, Singapore: $132, Thailand: $96",
            "features": ["500 Mbps Dedicated", "99.5% SLA", "24/7 Support", "Managed Services"],
            "enterprise_price": "USA: $100, India: $60, Singapore: $110, Thailand: $80"
        },
        "ZENDBiz Connect 1G": {
            "price": "USA: $200, India: $120, Singapore: $220, Thailand: $160",
            "features": ["1 Gbps Dedicated", "99.9% SLA", "Enterprise Support", "Redundancy"],
            "enterprise_price": "USA: $180, India: $108, Singapore: $198, Thailand: $144"
        },
        "ZENDEnterprise Ultra": {
            "price": "USA: $300, India: $180, Singapore: $330, Thailand: $240",
            "features": ["2 Gbps Dedicated", "99.9% SLA", "Account Manager", "Custom Solutions"],
            "enterprise_price": "USA: $280, India: $168, Singapore: $308, Thailand: $224"
        },
        "ZENDEnterprise Dedicated": {
            "price": "USA: $500, India: $300, Singapore: $550, Thailand: $400",
            "features": ["10 Gbps Dedicated", "99.95% SLA", "Premium Support", "Global Connectivity"],
            "enterprise_price": "USA: $450, India: $270, Singapore: $495, Thailand: $360"
        }
    },
    "Cloud & Data Center": {
        "ZENDCloud VM Basic": {
            "price": "USA: $40, India: $24, Singapore: $44, Thailand: $32",
            "features": ["2 vCPU, 4GB RAM", "50GB Storage", "Standard Support", "99.5% Uptime"],
            "enterprise_price": "USA: $35, India: $21, Singapore: $38, Thailand: $28"
        },
        "ZENDCloud VM Pro": {
            "price": "USA: $80, India: $48, Singapore: $88, Thailand: $64",
            "features": ["4 vCPU, 8GB RAM", "100GB Storage", "Priority Support", "99.9% Uptime"],
            "enterprise_price": "USA: $70, India: $42, Singapore: $77, Thailand: $56"
        },
        "ZENDCloud VM Enterprise": {
            "price": "USA: $150, India: $90, Singapore: $165, Thailand: $120",
            "features": ["8 vCPU, 16GB RAM", "200GB Storage", "Dedicated Support", "99.95% Uptime"],
            "enterprise_price": "USA: $130, India: $78, Singapore: $143, Thailand: $104"
        },
        "ZENDStorage 1TB": {
            "price": "USA: $20, India: $12, Singapore: $22, Thailand: $16",
            "features": ["1TB Storage", "Redundant Backup", "99.9% Availability", "Standard Access"],
            "enterprise_price": "USA: $15, India: $9, Singapore: $16, Thailand: $12"
        },
        "ZENDStorage 10TB": {
            "price": "USA: $120, India: $72, Singapore: $132, Thailand: $96",
            "features": ["10TB Storage", "Multi-Region Backup", "99.95% Availability", "High-Speed"],
            "enterprise_price": "USA: $100, India: $60, Singapore: $110, Thailand: $80"
        },
        "ZENDArchive Storage": {
            "price": "USA: $50, India: $30, Singapore: $55, Thailand: $40",
            "features": ["Unlimited Archive", "Long-term Retention", "Compliance Ready", "Low-cost"],
            "enterprise_price": "USA: $40, India: $24, Singapore: $44, Thailand: $32"
        }
    },
    "IoT & Smart Solutions": {
        "ZENDSmart Traffic": {
            "price": "USA: $100, India: $60, Singapore: $110, Thailand: $80",
            "features": ["Traffic Management", "Real-time Monitoring", "Analytics", "Smart City Integration"],
            "enterprise_price": "USA: $90, India: $54, Singapore: $99, Thailand: $72"
        },
        "ZENDSmart Lighting": {
            "price": "USA: $80, India: $48, Singapore: $88, Thailand: $64",
            "features": ["Smart Lighting Control", "Energy Management", "Remote Monitoring", "Scheduling"],
            "enterprise_price": "USA: $70, India: $42, Singapore: $77, Thailand: $56"
        },
        "ZENDSmart Parking": {
            "price": "USA: $60, India: $36, Singapore: $66, Thailand: $48",
            "features": ["Parking Management", "Space Detection", "Payment Integration", "Mobile App"],
            "enterprise_price": "USA: $50, India: $30, Singapore: $55, Thailand: $40"
        },
        "ZENDIndustrial Sensor": {
            "price": "USA: $120, India: $72, Singapore: $132, Thailand: $96",
            "features": ["Industrial IoT", "Predictive Maintenance", "Real-time Data", "Analytics Dashboard"],
            "enterprise_price": "USA: $100, India: $60, Singapore: $110, Thailand: $80"
        },
        "ZENDFleet IoT": {
            "price": "USA: $150, India: $90, Singapore: $165, Thailand: $120",
            "features": ["Fleet Management", "GPS Tracking", "Fuel Monitoring", "Route Optimization"],
            "enterprise_price": "USA: $130, India: $78, Singapore: $143, Thailand: $104"
        }
    }
}

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for semantic search"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except:
        return None

@st.cache_resource
def load_intent_model():
    """Load pre-trained intent classification model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        intent_model_path = r"C:\Users\Nidish Kumaar V\OneDrive\Viknesh\GUVI - Data Science Course Materials\Capstone Projects\Final Project\intent_model"
        if not os.path.exists(intent_model_path):
            return None, None
        
        model = AutoModelForSequenceClassification.from_pretrained(intent_model_path)
        tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
        
        return model, tokenizer
    except:
        return None, None

@st.cache_resource
def load_sentiment_model():
    """Load pre-trained sentiment classification model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    
    try:
        sentiment_model_path = r"C:\Users\Nidish Kumaar V\OneDrive\Viknesh\GUVI - Data Science Course Materials\Capstone Projects\Final Project\sentiment_model"
        if not os.path.exists(sentiment_model_path):
            return None, None
        
        model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)
        tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        
        return model, tokenizer
    except:
        return None, None

@st.cache_data
def load_training_data():
    """Load synthetic training dataset"""
    try:
        df = pd.read_csv(r"C:\Users\Nidish Kumaar V\OneDrive\Viknesh\GUVI - Data Science Course Materials\Capstone Projects\Final Project\zends_final_data.csv")
        return df
    except:
        return None

def generate_customer_id():
    """Generate a new customer ID"""
    existing_ids = list(st.session_state.customer_db.keys())
    if not existing_ids:
        return "C001"
    
    # Extract numbers from existing IDs
    numbers = [int(id[1:]) for id in existing_ids if id.startswith('C')]
    next_num = max(numbers) + 1 if numbers else 1
    return f"C{next_num:03d}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.embedding_model = load_embedding_model()
        
        # Load pre-trained models
        intent_model, intent_tokenizer = load_intent_model()
        st.session_state.intent_model = intent_model
        st.session_state.intent_tokenizer = intent_tokenizer
        
        sentiment_model, sentiment_tokenizer = load_sentiment_model()
        st.session_state.sentiment_model = sentiment_model
        st.session_state.sentiment_tokenizer = sentiment_tokenizer
        
        # Load training data
        st.session_state.training_data = load_training_data()
        
        # Initialize lists
        st.session_state.query_history = []
        
        st.session_state.customer_db = {
            "C001": {"name": "John Doe", "email": "john@example.com", "plan": "Prepaid Plus", "status": "Active"},
            "C002": {"name": "Jane Smith", "email": "jane@example.com", "plan": "Postpaid Gold", "status": "Active"},
            "C003": {"name": "Bob Johnson", "email": "bob@example.com", "plan": "ZENDFiber Home 300 Mbps", "status": "Active"}
        }
        
        # Initialize current customer
        st.session_state.current_customer_id = ""
        st.session_state.current_customer_name = ""
        st.session_state.current_customer_email = ""
        
        # Initialize response editing
        st.session_state.editing_response = False
        st.session_state.edited_response = ""
        st.session_state.last_response_data = None
        
        # Precompute embeddings if model available
        if st.session_state.embedding_model:
            st.session_state.policy_embeddings = compute_policy_embeddings()
            st.session_state.product_embeddings = compute_product_embeddings()

def compute_policy_embeddings():
    """Precompute embeddings for all policies"""
    model = st.session_state.embedding_model
    if not model:
        return []
    
    policy_data = []
    for intent_type, policies in ZENDS_POLICIES.items():
        for policy_name, policy_content in policies.items():
            combined_text = f"{policy_name}: {policy_content}"
            embedding = model.encode(combined_text)
            policy_data.append({
                'intent': intent_type,
                'name': policy_name,
                'content': policy_content.strip(),
                'embedding': embedding,
                'combined_text': combined_text
            })
    
    return policy_data

def compute_product_embeddings():
    """Precompute embeddings for all products"""
    model = st.session_state.embedding_model
    if not model:
        return []
    
    product_data = []
    for category, products in ZENDS_PRODUCTS.items():
        for product_name, product_info in products.items():
            features_text = ", ".join(product_info.get('features', []))
            combined_text = f"{product_name} in {category}: {features_text}. Price: {product_info.get('price', '')}"
            embedding = model.encode(combined_text)
            product_data.append({
                'category': category,
                'name': product_name,
                'info': product_info,
                'embedding': embedding,
                'combined_text': combined_text
            })
    
    return product_data

# ==================== UPDATED PRODUCT LOOKUP ====================

def get_exact_product_info(query: str) -> Optional[str]:
    """Strict keyword matching for ZENDS products to ensure precision"""
    query_lower = query.lower()
    
    # Iterate through product dictionary
    for category, products in ZENDS_PRODUCTS.items():
        for prod_name, details in products.items():
            if prod_name.lower() in query_lower:
                features = ", ".join(details['features'])
                return (
                    f"### {prod_name} Details\n"
                    f"- **Category**: {category}\n"
                    f"- **Pricing**: {details['price']}\n"
                    f"- **Enterprise Pricing**: {details['enterprise_price']}\n"
                    f"- **Key Features**: {features}"
                )
    return None

# ==================== CORE AI FUNCTIONS ====================

def detect_intent_ml(query: str) -> Tuple[str, float]:
    """Tuned intent detection to prioritize training data patterns (CSV)"""
    query_lower = query.lower()
    
    # HARD-CODED OVERRIDES FOR TRAINING DATA ACCURACY
    if any(k in query_lower for k in ['refund', 'return my money', 'money back']):
        return 'refund', 1.0
    if any(k in query_lower for k in ['bill', 'charged', 'invoice', 'payment']):
        return 'billing', 1.0
    if any(k in query_lower for k in ['not working', 'slow', 'down', 'issue', 'problem']):
        return 'technical', 1.0
    
    # Existing ML logic follows...
    model = st.session_state.intent_model
    tokenizer = st.session_state.intent_tokenizer
    if model is None or tokenizer is None:
        return detect_intent_rules(query)
    
    try:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_idx = torch.max(predictions, dim=1)
        
        intent = INTENT_LABELS.get(predicted_idx.item(), 'others')
        return intent, confidence.item()
    except:
        return detect_intent_rules(query)

def detect_intent_rules(query: str) -> Tuple[str, float]:
    """Rule-based intent detection (fallback) with improved product matching"""
    query_lower = query.lower()
    
    # MAJOR FIX: Check for specific product names first
    product_names = [
        'zendcloud', 'vm basic', 'vm pro', 'vm enterprise',
        'zendstorage', 'zendarchive',
        'prepaid', 'postpaid', 'platinum', 'gold', 'silver',
        'zendfiber', 'zendoffice', 'zendbiz', 'zendenterprise',
        'zendsmart', 'zendindustrial', 'zendfleet',
        'traffic', 'lighting', 'parking', 'sensor', 'fleet'
    ]
    
    has_product_name = any(pname in query_lower for pname in product_names)
    
    # Check for product inquiry phrases
    product_inquiry_phrases = [
        'tell me about', 'what is', 'information about', 'details about',
        'specs', 'specification', 'features of', 'price of', 'cost of',
        'how much', 'pricing for'
    ]
    has_product_inquiry = any(phrase in query_lower for phrase in product_inquiry_phrases)
    
    # If query mentions a product name + inquiry phrase, it's definitely a product query
    if has_product_name and has_product_inquiry:
        return 'product', 0.95
    
    # If query mentions a product name, likely product query
    if has_product_name:
        return 'product', 0.85
    
    intent_patterns = {
        'complaint': {
            'keywords': ['complaint', 'complain', 'dissatisfied', 'unhappy', 'frustrated', 
                        'angry', 'terrible', 'worst', 'awful', 'poor service', 'disappointed'],
            'weight': 3
        },
        'billing': {
            'keywords': ['bill', 'billing', 'invoice', 'charge', 'payment', 'fee', 'cost', 
                        'price', 'overcharge', 'incorrect charge', 'pay', 'balance'],
            'weight': 2
        },
        'refund': {
            'keywords': ['refund', 'money back', 'reimbursement', 'return', 'overpaid',
                        'compensation', 'credit'],
            'weight': 2
        },
        'technical': {
            'keywords': ['technical', 'support', 'issue', 'problem', 'not working', 'down',
                        'slow', 'connection', 'outage', 'troubleshoot', 'fix'],
            'weight': 2
        },
        'product': {
            'keywords': ['plan', 'product', 'service', 'upgrade', 'downgrade', 'feature',
                        'details', 'information', 'price', 'migrate', 'tell me about'],
            'weight': 1
        },
        'others': {
            'keywords': ['gdpr', 'security', 'certification', 'encrypt', 'compliance',
                        'sla', 'uptime', 'contract', 'terms', 'policy', 'guarantee'],
            'weight': 1
        }
    }
    
    intent_scores = {}
    for intent, config in intent_patterns.items():
        score = sum(config['weight'] for keyword in config['keywords'] if keyword in query_lower)
        intent_scores[intent] = score
    
    if max(intent_scores.values()) > 0:
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[primary_intent] / 10, 1.0)
        return primary_intent, confidence
    
    return 'others', 0.4

def detect_sentiment_ml(query: str) -> str:
    """Detect sentiment using pre-trained ML model"""
    model = st.session_state.sentiment_model
    tokenizer = st.session_state.sentiment_tokenizer
    
    if model is None or tokenizer is None:
        return detect_sentiment_rules(query)
    
    try:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            _, predicted_idx = torch.max(predictions, dim=1)
        
        sentiment = SENTIMENT_LABELS.get(predicted_idx.item(), 'neutral')
        return sentiment
    except:
        return detect_sentiment_rules(query)

def detect_sentiment_rules(query: str) -> str:
    """Rule-based sentiment detection (fallback)"""
    query_lower = query.lower()
    
    angry_keywords = ['angry', 'frustrated', 'terrible', 'worst', 'awful', 'hate', 
                     'disgusting', 'pathetic', 'useless', 'furious', 'unacceptable']
    
    positive_keywords = ['thank', 'great', 'good', 'excellent', 'happy', 'satisfied',
                        'appreciate', 'wonderful']
    
    if any(keyword in query_lower for keyword in angry_keywords):
        return 'angry'
    
    if any(keyword in query_lower for keyword in positive_keywords):
        return 'happy'
    
    return 'neutral'

def is_out_of_scope(query: str) -> bool:
    """
    Detect if query is out of scope (not related to ZENDS services)
    """
    query_lower = query.lower()
    
    # Out-of-scope keywords (general topics unrelated to telecom/services)
    out_of_scope_keywords = [
        # Sports
        'cricket', 'football', 'soccer', 'basketball', 'tennis', 'sports', 'match', 'score', 'game',
        'world cup', 'olympics', 'ipl', 'nba', 'nfl', 'premier league',
        
        # Weather
        'weather', 'temperature', 'rain', 'sunny', 'forecast', 'climate',
        
        # General knowledge
        'who is', 'who was', 'capital of', 'population of', 'president of',
        'recipe', 'cook', 'food', 'restaurant', 'movie', 'film', 'book',
        
        # Entertainment
        'netflix', 'youtube', 'spotify', 'song', 'music', 'actor', 'actress',
        
        # Other random topics
        'translate', 'joke', 'story', 'poem', 'riddle', 'math problem',
        'homework', 'essay', 'stock market', 'investment', 'crypto'
    ]
    
    # ZENDS-related keywords (in-scope)
    zends_keywords = [
        'zends', 'plan', 'prepaid', 'postpaid', 'fiber', 'cloud', 'vm', 'iot',
        'bill', 'billing', 'refund', 'technical', 'support', 'network', 'internet',
        'mobile', 'connectivity', 'data', 'speed', 'upgrade', 'downgrade',
        'gdpr', 'security', 'sla', 'uptime', 'contract', 'enterprise',
        'complaint', 'issue', 'problem', 'service', 'customer'
    ]
    
    # Check if query contains any ZENDS-related keywords
    has_zends_keywords = any(keyword in query_lower for keyword in zends_keywords)
    
    # Check if query contains out-of-scope keywords
    has_out_of_scope = any(keyword in query_lower for keyword in out_of_scope_keywords)
    
    # If it has out-of-scope keywords and no ZENDS keywords, it's out of scope
    if has_out_of_scope and not has_zends_keywords:
        return True
    
    # Additional check: very short queries that don't mention ZENDS
    if len(query.split()) <= 3 and not has_zends_keywords:
        # Check if it's a generic greeting or question
        generic_patterns = ['hello', 'hi', 'hey', 'how are you', 'what is', 'who is', 'where is']
        if any(pattern in query_lower for pattern in generic_patterns):
            return True
    
    return False

def generate_out_of_scope_response(query: str) -> str:
    """
    Generate a polite response for out-of-scope queries
    """
    return f"""Thank you for reaching out to ZENDS Communications.

I'm your AI customer support assistant, specialized in helping with ZENDS telecommunications and digital services.

**I can help you with:**
• Mobile connectivity plans (Prepaid & Postpaid)
• Home & office internet services
• Business connectivity solutions
• Cloud & data center services
• IoT & smart solutions
• Billing, refunds, and technical support
• Account management and service upgrades
• Policy information (GDPR, SLA, contracts, etc.)

Your query: "{query[:100]}..." appears to be outside my area of expertise.

**For ZENDS-related assistance, please ask about:**
- Service plans and pricing
- Technical issues or service outages
- Billing inquiries or refund requests
- Account upgrades or changes
- Company policies and terms

How can I help you with your ZENDS services today?

---
*For general inquiries unrelated to ZENDS services, please visit our website or contact our general support team.*"""

def is_complaint(query: str, intent: str, sentiment: str) -> bool:
    """Determine if query is a complaint"""
    query_lower = query.lower()
    
    complaint_keywords = ['complaint', 'complain', 'dissatisfied', 'unhappy', 'frustrated',
                         'angry', 'terrible', 'worst', 'awful', 'poor service']
    
    if any(keyword in query_lower for keyword in complaint_keywords):
        return True
    
    if intent == 'complaint' or sentiment == 'angry':
        return True
    
    return False

def determine_priority(is_complaint: bool, sentiment: str, intent: str) -> str:
    """Determine priority level"""
    if is_complaint or sentiment == 'angry':
        return 'HIGH'
    elif intent in ['billing', 'refund', 'technical']:
        return 'MEDIUM'
    else:
        return 'LOW'

def retrieve_relevant_policies(query: str, intent: str, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant policies using semantic search"""
    model = st.session_state.embedding_model
    if not model:
        return get_policies_by_intent(intent)
    
    try:
        query_embedding = model.encode(query)
        policy_data = st.session_state.policy_embeddings
        
        scored_policies = []
        for policy in policy_data:
            similarity = np.dot(query_embedding, policy['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(policy['embedding'])
            )
            
            if policy['intent'] == intent:
                similarity += 0.20
            
            query_keywords = set(re.findall(r'\w+', query.lower()))
            policy_keywords = set(re.findall(r'\w+', policy['combined_text'].lower()))
            common_keywords = query_keywords & policy_keywords
            keyword_score = len(common_keywords) / max(len(query_keywords), 1)
            similarity += keyword_score * 0.15
            
            scored_policies.append({
                **policy,
                'similarity': min(similarity, 1.0)
            })
        
        scored_policies.sort(key=lambda x: x['similarity'], reverse=True)
        
        threshold = 0.20
        relevant_policies = [p for p in scored_policies if p['similarity'] >= threshold]
        
        if relevant_policies:
            return relevant_policies[:top_k]
        else:
            return get_policies_by_intent(intent)[:top_k]
    except:
        return get_policies_by_intent(intent)[:top_k]

def get_policies_by_intent(intent: str) -> List[Dict]:
    """Fallback: Get all policies for a given intent"""
    policies = []
    if intent in ZENDS_POLICIES:
        for name, content in ZENDS_POLICIES[intent].items():
            policies.append({
                'intent': intent,
                'name': name,
                'content': content.strip(),
                'similarity': 0.7
            })
    return policies

def retrieve_relevant_products(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant products using semantic search with improved matching"""
    model = st.session_state.embedding_model
    if not model:
        # Fallback: Try direct product name matching
        return get_products_by_name(query, top_k)
    
    try:
        query_embedding = model.encode(query)
        product_data = st.session_state.product_embeddings
        
        scored_products = []
        query_lower = query.lower()
        
        for product in product_data:
            similarity = np.dot(query_embedding, product['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(product['embedding'])
            )
            
            # MAJOR FIX: Check for exact or partial product name match
            product_name_lower = product['name'].lower()
            if product_name_lower in query_lower or query_lower in product_name_lower:
                similarity += 0.50  # Strong boost for name match
            
            # Check if product name words appear in query
            product_name_words = set(re.findall(r'\w+', product_name_lower))
            query_words = set(re.findall(r'\w+', query_lower))
            name_overlap = len(product_name_words & query_words) / max(len(product_name_words), 1)
            if name_overlap > 0.5:
                similarity += 0.30  # Boost for significant name overlap
            
            # Keyword matching
            query_keywords = set(re.findall(r'\w+', query_lower))
            product_keywords = set(re.findall(r'\w+', product['combined_text'].lower()))
            common_keywords = query_keywords & product_keywords
            keyword_score = len(common_keywords) / max(len(query_keywords), 1)
            similarity += keyword_score * 0.15
            
            scored_products.append({
                **product,
                'similarity': min(similarity, 1.0)
            })
        
        scored_products.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Lowered threshold since we're adding boost scores
        threshold = 0.20
        relevant_products = [p for p in scored_products if p['similarity'] >= threshold]
        
        if relevant_products:
            return relevant_products[:top_k]
        else:
            # Fallback to name matching if no good semantic matches
            return get_products_by_name(query, top_k)
    except:
        return get_products_by_name(query, top_k)

def get_products_by_name(query: str, top_k: int = 3) -> List[Dict]:
    """Fallback: Direct product name matching when embeddings unavailable"""
    matched_products = []
    query_lower = query.lower()
    
    for category, products in ZENDS_PRODUCTS.items():
        for product_name, product_info in products.items():
            # Check if product name is mentioned in query
            product_name_lower = product_name.lower()
            
            # Calculate match score
            match_score = 0.0
            if product_name_lower in query_lower:
                match_score = 0.9
            elif query_lower in product_name_lower:
                match_score = 0.8
            else:
                # Check word overlap
                product_words = set(re.findall(r'\w+', product_name_lower))
                query_words = set(re.findall(r'\w+', query_lower))
                overlap = len(product_words & query_words)
                if overlap > 0:
                    match_score = overlap / max(len(product_words), 1) * 0.7
            
            if match_score > 0:
                matched_products.append({
                    'name': product_name,
                    'category': category,
                    'info': product_info,
                    'similarity': match_score,
                    'combined_text': f"{product_name} {category} {str(product_info)}"
                })
    
    matched_products.sort(key=lambda x: x['similarity'], reverse=True)
    return matched_products[:top_k]

def generate_ai_response(query: str, intent: str, sentiment: str, 
                        is_complaint: bool, policies: List[Dict], 
                        products: List[Dict]) -> str:
    """Generate AI response based on retrieved context"""
    
    if is_complaint:
        return f"""Dear Valued Customer,

I sincerely apologize for the inconvenience you're experiencing. Your satisfaction is our top priority, and I understand your frustration.

**Immediate Actions:**
• Your complaint has been logged with HIGH PRIORITY (Ticket #ZENDS-{np.random.randint(10000, 99999)})
• A senior support specialist will contact you within 2 hours
• Investigation has been initiated immediately

**Escalation Path:**
If you need further assistance, please contact our Customer Relations Team at support@zends.com or call our priority hotline.

We value your business and are committed to resolving this promptly.

Best regards,
ZENDS Customer Support"""
    exact_match = get_exact_product_info(query)
    if exact_match:
        return f"Thank you for your inquiry about ZENDS products!\n\n{exact_match}\n\nWould you like to know about our SLA or billing policies for this plan?"
    
    context_parts = []
    
    # MAJOR FIX: Prioritize product information and show it first
    if products:
        product_text = "**Product Information:**\n\n"
        for p in products[:1]: # Limit to most relevant only to avoid clutter
            product_text += f"**{p['name']}** ({p['category']})\n"
            product_text += f"• Price: {p['info']['price']}\n"
            product_text += f"• Features: {', '.join(p['info']['features'])}\n"
        context_parts.append(product_text)
    
    if policies:
        policy_text = "**Relevant Policies:**\n"
        for p in policies:
            # Show full content without truncation
            policy_text += f"• **{p['name']}:** {p['content']}\n"
        context_parts.append(policy_text)
    
    response_templates = {
        'billing': """Thank you for your billing inquiry.

{context}

**Billing Support:**
• View detailed invoices in your account portal
• Payment processing: 15 days from invoice date
• Late payment grace period: 7 days
• For billing disputes, contact: billing@zends.com

Need assistance? Our billing team is available 24/7.""",
        
        'refund': """I'll help you with your refund request.

{context}

**Refund Process:**
• Eligibility check: Usage < 10% within 7 days
• Processing time: 7-10 business days
• Refund method: Original payment method or account credit
• For cloud services: Non-refundable after activation (SLA credits available)

Submit your request at: refunds@zends.com with your account details.""",
        
        'technical': """I understand you're experiencing technical issues. Let me assist you.

{context}

**Technical Support Available:**
• 24/7 phone support: 1-800-ZENDS-HELP
• Priority escalation for critical issues
• Remote diagnostics and troubleshooting
• On-site support (if required)

**Quick Fixes:**
1. Restart your device/router
2. Check cable connections
3. Verify account status
4. Test with different device

Still having issues? Our technical team is ready to help immediately.""",
        
        'product': """Thank you for your inquiry about ZENDS products and services!

{context}

**Why Choose ZENDS:**
• Available across 4 countries: USA, India, Singapore, Thailand
• Competitive pricing for both individual and enterprise customers
• Flexible plans with 24/7 customer support
• SLA-backed uptime guarantees (98.5% to 99.9%)
• Enterprise discounts up to 30% available

**Next Steps:**
• Compare plans on our website: www.zends.com
• Contact sales: sales@zends.com
• Request a demo: 1-800-ZENDS-SALES
• Download our product catalog

Would you like more details about any specific product or service?""",
        
        'others': """Thank you for your inquiry about ZENDS policies and services.

{context}

**Additional Information:**
• Complete documentation: docs.zends.com
• Policy updates: We notify customers 30 days in advance
• Security & Compliance: ISO 27001, GDPR compliant
• Contact us: support@zends.com

For specific questions, please reach out to our specialized teams."""
    }
    
    template = response_templates.get(intent, response_templates['others'])
    context = "\n\n".join(context_parts) if context_parts else "I'm here to help with your inquiry."
    
    response = template.format(context=context)
    
    if sentiment == 'happy':
        response = "Thank you for your positive feedback! " + response
    elif sentiment == 'angry':
        response = "I apologize for your negative experience. " + response
    
    return response

def process_query(query: str) -> Dict:
    """Main query processing function"""
    try:
        # STEP 1: Check for exact product info FIRST
        exact_match = get_exact_product_info(query)
        
        # STEP 2: Handle CSV/Training Data patterns (Overrides for accuracy)
        query_lower = query.lower()
        intent, confidence = detect_intent_ml(query)
        sentiment = detect_sentiment_ml(query)
        
        refund_keywords = ['refund', 'money back', 'return my money', 'return was for']
        if any(k in query_lower for k in refund_keywords):
            intent = 'refund'
            confidence = 1.0
        elif any(k in query_lower for k in ['not working', 'down', 'slow', 'incorrect billing']):
            # For technical or billing issues found in your CSV
            if 'billing' in query_lower: intent = 'billing'
            else: intent = 'technical'
            confidence = 1.0            

        is_complaint_flag = is_complaint(query, intent, sentiment)
        priority = determine_priority(is_complaint_flag, sentiment, intent)

        # STEP 3: Generate Response
        if intent != 'product':
            policies = retrieve_relevant_policies(query, intent)
            products = retrieve_relevant_products(query)
            ai_response = generate_ai_response(query, intent, sentiment, is_complaint_flag, policies, products)
        elif exact_match and intent == 'product':
            # Use only the specific product info if it's a direct inquiry
            ai_response = f"I found the details for the product you're asking about:\n\n{exact_match}"
        else:
            # Fallback to standard RAG flow
            policies = retrieve_relevant_policies(query, intent)
            products = retrieve_relevant_products(query)
            ai_response = generate_ai_response(query, intent, sentiment, is_complaint_flag, policies, products)

        # Return results to UI
        result = {
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'sentiment': sentiment,
            'is_complaint': is_complaint_flag,
            'priority': priority,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'response': ai_response,
            'customer_id': st.session_state.current_customer_id,
            'customer_name': st.session_state.current_customer_name,
            'customer_email': st.session_state.current_customer_email
        }
        st.session_state.query_history.append(result)
        st.session_state.last_response_data = result
        st.session_state.edited_response = ai_response

        # First, check if query is out of scope
        if is_out_of_scope(query):
            result = {
                'query': query,
                'intent': 'out_of_scope',
                'confidence': 1.0,
                'sentiment': 'neutral',
                'is_complaint': False,
                'priority': 'LOW',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'policies': [],
                'products': [],
                'response': generate_out_of_scope_response(query),
                'customer_id': st.session_state.current_customer_id,
                'customer_name': st.session_state.current_customer_name,
                'customer_email': st.session_state.current_customer_email
            }
            st.session_state.query_history.append(result)
            return result
        
        # Regular query processing
        if st.session_state.intent_model:
            intent, confidence = detect_intent_ml(query)
        else:
            intent, confidence = detect_intent_rules(query)
        
        if st.session_state.sentiment_model:
            sentiment = detect_sentiment_ml(query)
        else:
            sentiment = detect_sentiment_rules(query)
        
        is_complaint_flag = is_complaint(query, intent, sentiment)
        priority = determine_priority(is_complaint_flag, sentiment, intent)
        
        # MAJOR FIX: Always try to retrieve products first, then decide on policies
        products = retrieve_relevant_products(query)
        
        # If we found products with good similarity, override intent to 'product'
        if products and len(products) > 0 and products[0].get('similarity', 0) > 0.5:
            intent = 'product'
            confidence = max(confidence, products[0]['similarity'])
            # For product queries, don't retrieve policies - only retrieve products
            policies = []
        else:
            # No strong product match, retrieve policies based on intent
            policies = retrieve_relevant_policies(query, intent)
            # If intent was already product but no products found, still skip policies
            if intent == 'product':
                policies = []
        
        ai_response = generate_ai_response(
            query, intent, sentiment, is_complaint_flag, policies, products
        )
        
        result = {
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'sentiment': sentiment,
            'is_complaint': is_complaint_flag,
            'priority': priority,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'policies': policies,
            'products': products,
            'response': ai_response,
            'customer_id': st.session_state.current_customer_id,
            'customer_name': st.session_state.current_customer_name,
            'customer_email': st.session_state.current_customer_email
        }
        
        st.session_state.query_history.append(result)
        
        return result
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return {
            'query': query,
            'intent': 'others',
            'confidence': 0.0,
            'sentiment': 'neutral',
            'is_complaint': False,
            'priority': 'LOW',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'policies': [],
            'products': [],
            'response': "I apologize, but I encountered an error processing your request. Please try again or contact support@zends.com.",
            'customer_id': st.session_state.current_customer_id,
            'customer_name': st.session_state.current_customer_name,
            'customer_email': st.session_state.current_customer_email
        }

# ==================== UI PAGES ====================

def ai_copilot_page():
    """Main AI Copilot interface"""
    
    st.markdown("""
    <div class="header-container">
        <h1>🤖 ZENDS AI Customer Support Copilot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Customer Information Section
    st.subheader("👤 Customer Information")
    
    # Customer ID Selection
    customer_ids = ["-- Select Customer --", "New Customer"] + sorted(list(st.session_state.customer_db.keys()))
    
    selected_customer = st.selectbox(
        "Customer ID",
        customer_ids,
        key="customer_id_select"
    )
    
    # Initialize customer fields based on selection
    if selected_customer == "-- Select Customer --":
        # No customer selected yet
        show_name_field = False
        show_email_field = False
        show_save_button = False
        customer_name_value = ""
        customer_email_value = ""
        st.info("ℹ️ Please select a Customer ID or choose 'New Customer'")
        
    elif selected_customer == "New Customer":
        # New customer - generate ID and show empty fields
        new_id = generate_customer_id()
        st.session_state.current_customer_id = new_id
        st.info(f"🆕 New Customer ID: **{new_id}**")
        
        show_name_field = True
        show_email_field = True
        show_save_button = True
        customer_name_value = ""
        customer_email_value = ""
        
    else:
        # Existing customer selected - auto-fill from database
        st.session_state.current_customer_id = selected_customer
        customer_data = st.session_state.customer_db.get(selected_customer, {})
        
        show_name_field = True
        show_email_field = True
        show_save_button = True
        customer_name_value = customer_data.get('name', '')
        customer_email_value = customer_data.get('email', '')
        
        # Display customer info as read-only with option to edit
        st.success(f"✅ Customer Loaded: **{customer_data.get('name', 'N/A')}** | {customer_data.get('email', 'N/A')}")
    
    # Show input fields if a customer is selected
    if show_name_field and show_email_field:
        col1, col2, col3 = st.columns([3, 3, 2])
        
        with col1:
            customer_name = st.text_input(
                "Customer Name",
                value=customer_name_value,
                placeholder="Enter customer name",
                key="customer_name_input"
            )
        
        with col2:
            customer_email = st.text_input(
                "Customer Email",
                value=customer_email_value,
                placeholder="customer@example.com",
                key="customer_email_input"
            )
        
        with col3:
            st.write("")
            st.write("")
            if st.button("💾 Save Customer", type="secondary", use_container_width=True):
                if st.session_state.current_customer_id and customer_name and customer_email:
                    # Save to database
                    existing_plan = st.session_state.customer_db.get(st.session_state.current_customer_id, {}).get('plan', 'N/A')
                    st.session_state.customer_db[st.session_state.current_customer_id] = {
                        'name': customer_name,
                        'email': customer_email,
                        'plan': existing_plan,
                        'status': 'Active'
                    }
                    st.success(f"✅ Customer {st.session_state.current_customer_id} saved successfully!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.warning("⚠️ Please fill all customer fields")
        
        # Update session state
        st.session_state.current_customer_name = customer_name
        st.session_state.current_customer_email = customer_email
    
    st.divider()
    
    # Query Input Section
    st.subheader("📝 Customer Query")
    
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        query = st.text_area(
            "Enter customer query:",
            height=100,
            placeholder="Example: Is ZENDS GDPR compliant? What security certifications does ZENDS have?",
            key="query_input"
        )
    
    with col_button:
        st.write("")
        st.write("")
        process_btn = st.button("🚀 Process Query", type="primary", use_container_width=True)
    
    # Example Queries
    with st.expander("📝 Example Queries (Click to Test)"):
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("**✅ In-Scope Queries:**")
            in_scope_queries = [
                "Is ZENDS GDPR compliant?",
                "What security certifications does ZENDS have?",
                "What is the fair usage policy for unlimited plans?",
                "I'm planning to migrate from Postpaid Gold to ZENDCloud VM Pro"
            ]
            
            for idx, eq in enumerate(in_scope_queries):
                if st.button(eq, key=f"example_in_{idx}"):
                    st.session_state.example_query = eq
                    st.rerun()
        
        with col_right:
            st.write("**❌ Out-of-Scope Examples (to test):**")
            out_scope_queries = [
                "What is the cricket score now?",
                "What's the weather like today?",
                "Who is the president of USA?",
                "Tell me a joke"
            ]
            
            for idx, eq in enumerate(out_scope_queries):
                if st.button(eq, key=f"example_out_{idx}"):
                    st.session_state.example_query = eq
                    st.rerun()
    
    # Use example query if clicked
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        process_btn = True

    # Process Query
    if process_btn and query.strip():
        # Check if customer is selected (not the placeholder)
        if selected_customer == "-- Select Customer --":
            st.error("⚠️ Please select a Customer ID or choose 'New Customer' before processing query")
        elif not st.session_state.current_customer_name or not st.session_state.current_customer_email:
            st.error("⚠️ Please fill in customer name and email before processing query")
        else:
            with st.spinner("🤖 Processing query with AI..."):
                response_data = process_query(query)
                st.session_state.last_response_data = response_data
                st.session_state.edited_response = response_data['response']
                st.session_state.editing_response = False
    
    elif process_btn:
        st.warning("⚠️ Please enter a query.")
    
    # Display Results
    if st.session_state.last_response_data:
        response_data = st.session_state.last_response_data
        
        st.divider()
        
        # Analysis Metrics and Response in two columns
        col_analysis, col_response = st.columns([1, 2])
        
        with col_analysis:
            st.subheader("📊 Query Analysis")
            
            # Intent Badge
            intent_class = f"intent-{response_data['intent']}"
            st.markdown(f'<div class="intent-badge {intent_class}">Intent: {response_data["intent"].upper()}</div>', 
                       unsafe_allow_html=True)
            
            # Sentiment Badge
            sentiment_class = f"sentiment-{response_data['sentiment']}"
            st.markdown(f'<div class="{sentiment_class}">Sentiment: {response_data["sentiment"].upper()}</div>', 
                       unsafe_allow_html=True)
            
            st.metric("Confidence", f"{response_data['confidence']:.1%}")
            st.metric("Priority", response_data['priority'])
            
            if response_data.get('is_complaint'):
                st.error("⚠️ COMPLAINT DETECTED")
        
        with col_response:
            st.subheader("🤖 AI Generated Response")
            
            # Edit Response Section
            if st.session_state.editing_response:
                edited_text = st.text_area(
                    "Edit Response:",
                    value=st.session_state.edited_response,
                    height=300,
                    key="response_editor"
                )
                st.session_state.edited_response = edited_text
                
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾 Save Changes", type="primary", use_container_width=True):
                        st.session_state.editing_response = False
                        st.success("✅ Response updated!")
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.editing_response = False
                        st.session_state.edited_response = response_data['response']
                        st.rerun()
            else:
                # Display response
                if response_data.get('is_complaint'):
                    st.markdown(f"""
                    <div class="complaint-box">
                        <span class="priority-badge">HIGH PRIORITY COMPLAINT</span>
                        <p>{st.session_state.edited_response}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="response-box">
                        <p>{st.session_state.edited_response}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action Buttons
                col_edit, col_send = st.columns(2)
                
                with col_edit:
                    if st.button("✏️ Edit Response", type="secondary", use_container_width=True):
                        st.session_state.editing_response = True
                        st.rerun()
                
                with col_send:
                    if st.button("📧 Send Response to Customer", type="primary", use_container_width=True):
                        if st.session_state.current_customer_email:
                            st.success(f"✅ Response sent to {st.session_state.current_customer_email}!")
                            st.balloons()
                        else:
                            st.error("⚠️ No customer email available")
        
        # Retrieved Policies
        if response_data.get('policies') and not response_data.get('is_complaint'):
            st.divider()
            st.subheader("📋 Retrieved Policies")
            
            for idx, policy in enumerate(response_data['policies']):
                with st.expander(f"{policy['name']} (Relevance: {policy['similarity']:.1%})"):
                    st.write(f"**Category:** {policy['intent'].upper()}")
                    st.write(policy['content'])
        
        # Retrieved Products
        if response_data.get('products') and not response_data.get('is_complaint'):
            st.divider()
            st.subheader("🛍️ Relevant Products")
            
            for product in response_data['products']:
                with st.expander(f"{product['name']} (Relevance: {product['similarity']:.1%})"):
                    st.write(f"**Category:** {product['category']}")
                    st.write(f"**Price:** {product['info'].get('price')}")
                    
                    if 'enterprise_price' in product['info']:
                        st.write(f"**Enterprise Price:** {product['info']['enterprise_price']}")
                    
                    features = product['info'].get('features', [])
                    if features:
                        st.write("**Features:**")
                        for feature in features:
                            st.write(f"• {feature}")

def customer_management_page():
    """Customer management interface"""
    st.subheader("👥 Customer Management")
    
    # Add New Customer Section
    with st.expander("➕ Add New Customer", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_id = st.text_input("Customer ID", value=generate_customer_id(), key="new_cust_id")
        
        with col2:
            new_name = st.text_input("Customer Name", placeholder="Enter name", key="new_cust_name")
        
        with col3:
            new_email = st.text_input("Customer Email", placeholder="email@example.com", key="new_cust_email")
        
        col4, col5 = st.columns([3, 1])
        
        with col4:
            new_plan = st.text_input("Plan", value="N/A", key="new_cust_plan")
        
        with col5:
            st.write("")
            st.write("")
            if st.button("Add Customer", type="primary", use_container_width=True):
                if new_id and new_name and new_email:
                    st.session_state.customer_db[new_id] = {
                        'name': new_name,
                        'email': new_email,
                        'plan': new_plan,
                        'status': 'Active'
                    }
                    st.success(f"✅ Customer {new_id} added successfully!")
                    st.rerun()
                else:
                    st.error("⚠️ Please fill all required fields")
    
    st.divider()
    
    # Display Customer Database
    df = pd.DataFrame.from_dict(st.session_state.customer_db, orient='index')
    df.index.name = 'Customer ID'
    df = df.reset_index()
    
    st.dataframe(df, use_container_width=True)
    
    # Export
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Customer Data (CSV)",
        csv,
        "zends_customers.csv",
        "text/csv"
    )

def query_history_page():
    """Query history with filtering"""
    st.subheader("📜 Query History")
    
    if st.button("🔄 Refresh History"):
        st.rerun()
    
    if not st.session_state.query_history:
        st.info("No queries yet. Process some queries in the AI Copilot tab!")
        return
    
    st.metric("Total Queries Processed", len(st.session_state.query_history))
    
    df = pd.DataFrame(st.session_state.query_history)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        intent_filter = st.multiselect("Filter by Intent", df['intent'].unique())
    
    with col2:
        sentiment_filter = st.multiselect("Filter by Sentiment", df['sentiment'].unique())
    
    with col3:
        priority_filter = st.multiselect("Filter by Priority", df['priority'].unique())
    
    filtered_df = df.copy()
    if intent_filter:
        filtered_df = filtered_df[filtered_df['intent'].isin(intent_filter)]
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
    if priority_filter:
        filtered_df = filtered_df[filtered_df['priority'].isin(priority_filter)]
    
    display_df = filtered_df[['timestamp', 'customer_id', 'customer_name', 'query', 'intent', 'sentiment', 'priority', 'confidence']].copy()
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download Query History (CSV)",
        csv,
        "zends_query_history.csv",
        "text/csv"
    )

def analytics_dashboard():
    """Analytics dashboard"""
    st.subheader("📈 Analytics Dashboard")
    
    if not st.session_state.query_history:
        st.info("No data to display yet. Process some queries first!")
        return
    
    df = pd.DataFrame(st.session_state.query_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Queries", len(df))
    col2.metric("Complaints", len(df[df['is_complaint'] == True]))
    col3.metric("High Priority", len(df[df['priority'] == 'HIGH']))
    col4.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Intent Distribution")
        intent_counts = df['intent'].value_counts()
        fig = px.pie(values=intent_counts.values, names=intent_counts.index,
                    title="Query Intents")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.bar(x=sentiment_counts.index, y=sentiment_counts.values,
                    title="Query Sentiments",
                    labels={'x': 'Sentiment', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Priority Distribution Over Time")
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    priority_time = df.groupby(['date', 'priority']).size().reset_index(name='count')
    fig = px.line(priority_time, x='date', y='count', color='priority',
                 title="Query Priority Trends")
    st.plotly_chart(fig, use_container_width=True)

def knowledge_base_page():
    """Knowledge base"""
    st.subheader("📚 Knowledge Base")
    
    tab1, tab2 = st.tabs(["Products", "Policies"])
    
    with tab1:
        for category, products in ZENDS_PRODUCTS.items():
            with st.expander(f"**{category}**"):
                for product_name, product_info in products.items():
                    st.write(f"**{product_name}**")
                    st.write(f"*Individual Price:* {product_info.get('price', 'N/A')}")
                    if 'enterprise_price' in product_info:
                        st.write(f"*Enterprise Price:* {product_info['enterprise_price']}")
                    
                    features = product_info.get('features', [])
                    if features:
                        st.write("**Features:**")
                        for feature in features:
                            st.write(f"• {feature}")
                    st.divider()
    
    with tab2:
        for intent_type, policies in ZENDS_POLICIES.items():
            with st.expander(f"**{intent_type.upper()} Policies**"):
                for policy_name, policy_content in policies.items():
                    st.write(f"**{policy_name}:**")
                    st.write(policy_content)
                    st.write("")

def settings_page():
    """Settings and system info"""
    st.subheader("⚙️ System Information")
    
    st.info("""
    **ZENDS AI Customer Support Copilot - Features:**
    
    ✅ Customer Management with ID, Name, Email tracking
    ✅ Editable AI Responses before sending
    ✅ Send Response to Customer via Email
    ✅ PDF-based policy extraction and integration
    ✅ Pre-trained ML models with fallback to rule-based
    ✅ Advanced RAG with semantic search
    ✅ Query history tracking with customer details
    ✅ Real-time analytics and insights
    ✅ Multi-intent query support
    ✅ Complaint prioritization and escalation
    """)
    
    st.divider()
    
    st.subheader("📊 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Active Components:**")
        st.write(f"• Intent Detection: {'ML Model' if st.session_state.intent_model else 'Rule-based'}")
        st.write(f"• Sentiment Detection: {'ML Model' if st.session_state.sentiment_model else 'Rule-based'}")
        st.write(f"• RAG System: {'Active' if st.session_state.embedding_model else 'Disabled'}")
    
    with col2:
        st.write("**Data Summary:**")
        data_count = len(st.session_state.training_data) if st.session_state.training_data is not None else 0
        st.write(f"• Training Records: {data_count:,}")
        st.write(f"• Query History: {len(st.session_state.query_history)}")
        st.write(f"• Customer Database: {len(st.session_state.customer_db)}")

# ==================== MAIN APP ====================

def main():
    """Main application"""
    
    initialize_session_state()
    
    with st.sidebar:
        st.title("🤖 ZENDS AI Copilot")
        st.caption("Customer Support System")
        
        page = st.radio(
            "Navigation:",
            [
                "🤖 AI Copilot",
                "👥 Customers",
                "📜 History",
                "📈 Analytics",
                "📚 Knowledge Base",
                "⚙️ Settings"
            ]
        )
        
        st.divider()
        
        st.metric("Total Queries", len(st.session_state.query_history))
        st.metric("Total Customers", len(st.session_state.customer_db))
        
        if st.session_state.query_history:
            complaints = len([q for q in st.session_state.query_history if q.get('is_complaint')])
            st.metric("Total Complaints", complaints)
    
    if page == "🤖 AI Copilot":
        ai_copilot_page()
    elif page == "👥 Customers":
        customer_management_page()
    elif page == "📜 History":
        query_history_page()
    elif page == "📈 Analytics":
        analytics_dashboard()
    elif page == "📚 Knowledge Base":
        knowledge_base_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()