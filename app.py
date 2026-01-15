import streamlit as st
import requests
import json
import os
from groq import Groq
from tavily import TavilyClient
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Budget Window Predictor",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .score-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-green {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .score-yellow {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
    }
    .score-red {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .evidence-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: #e3f2fd;
        color: #1976d2;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎯 Budget Window Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Sales Intelligence with 100% Free Stack (Groq + Tavily + FullEnrich)</p>', unsafe_allow_html=True)

# Get API keys from environment variables or sidebar
groq_key = os.getenv('GROQ_API_KEY', '')
tavily_key = os.getenv('TAVILY_API_KEY', '')
fullenrich_key = os.getenv('FULLENRICH_API_KEY', '')

# Sidebar for API Keys (optional override)
with st.sidebar:
    st.header("🔑 API Configuration")
    
    if groq_key and tavily_key and fullenrich_key:
        st.success("✅ API keys loaded from environment")
        st.caption("Keys are securely loaded from .env file")
    else:
        st.info("💡 Add keys to .env file or enter below")
        st.markdown("**Get your FREE keys:**")
        st.markdown("- [Groq Console](https://console.groq.com/keys)")
        st.markdown("- [Tavily](https://tavily.com/)")
        st.markdown("- [FullEnrich](https://fullenrich.com/)")
    
    st.divider()
    
    # Allow manual override if needed
    groq_key_input = st.text_input("Groq API Key (optional override)", type="password", value=groq_key, help="Leave empty to use .env")
    tavily_key_input = st.text_input("Tavily API Key (optional override)", type="password", value=tavily_key, help="Leave empty to use .env")
    fullenrich_key_input = st.text_input("FullEnrich API Key (optional override)", type="password", value=fullenrich_key, help="Leave empty to use .env")
    
    # Use manual input if provided, otherwise use env vars
    groq_key = groq_key_input if groq_key_input else groq_key
    tavily_key = tavily_key_input if tavily_key_input else tavily_key
    fullenrich_key = fullenrich_key_input if fullenrich_key_input else fullenrich_key
    
    st.divider()
    st.caption("💰 Total Cost: **$0.00** per analysis")

# Main content
domain = st.text_input("🌐 Enter Target Domain", placeholder="e.g., stripe.com", help="Company domain to analyze")

def get_fullenrich_data(domain, api_key):
    """Fetch company and contact data from FullEnrich API"""
    try:
        url = "https://app.fullenrich.com/api/v1/company/enrich"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {"domain": domain}
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"FullEnrich API returned status {response.status_code}")
            return None
    except Exception as e:
        st.error(f"FullEnrich Error: {str(e)}")
        return None

def get_market_signals(domain, api_key):
    """Use Tavily to search for market signals"""
    try:
        client = TavilyClient(api_key=api_key)
        
        signals = {}
        
        # Search for funding information
        funding_query = f"When was {domain} last funding round?"
        funding_results = client.search(funding_query, max_results=3)
        signals['funding'] = funding_results
        
        # Search for hiring activity
        hiring_query = f"Is {domain} hiring for sales roles?"
        hiring_results = client.search(hiring_query, max_results=3)
        signals['hiring'] = hiring_results
        
        # Search for tech stack
        tech_query = f"What tech stack does {domain} use?"
        tech_results = client.search(tech_query, max_results=3)
        signals['tech_stack'] = tech_results
        
        return signals
    except Exception as e:
        st.error(f"Tavily Error: {str(e)}")
        return None

def analyze_with_groq_advanced(company_data, market_signals, api_key, domain):
    """
    ADVANCED: Multi-step reasoning with Groq (Llama 3.3)
    
    This approach uses:
    1. Structured data extraction
    2. Multi-dimensional scoring
    3. Chain-of-thought reasoning
    4. Confidence intervals
    """
    try:
        client = Groq(api_key=api_key)
        
        # Step 1: Extract structured insights from market signals
        extraction_prompt = f"""You are a data extraction specialist. Analyze these market signals and extract ONLY factual data points in JSON format.

MARKET SIGNALS:
{json.dumps(market_signals, indent=2)}

Return ONLY this JSON structure:
{{
    "funding": {{
        "last_round_date": "<date or null>",
        "amount": "<amount or null>",
        "recency_months": <number or null>
    }},
    "hiring": {{
        "active_roles": <number>,
        "sales_focused": <true/false>,
        "growth_signal": "<expanding|stable|contracting>"
    }},
    "tech_stack": {{
        "modern_tools": ["tool1", "tool2"],
        "legacy_systems": ["system1"],
        "recent_changes": <true/false>
    }}
}}"""

        extraction_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000
        )
        
        extracted_data = json.loads(extraction_response.choices[0].message.content.strip())
        
        # Step 2: Multi-dimensional scoring
        scoring_prompt = f"""You are a sales intelligence scoring engine. Calculate scores across 5 dimensions:

COMPANY DATA:
{json.dumps(company_data, indent=2)}

EXTRACTED INSIGHTS:
{json.dumps(extracted_data, indent=2)}

Calculate scores (0-100) for each dimension using this logic:

1. TIMING_SCORE (0-100):
   - Recent funding (<3mo): 100
   - Funding 3-6mo: 80
   - Funding 6-12mo: 60
   - Funding >12mo: 30
   - No funding data: 50

2. GROWTH_SCORE (0-100):
   - Active sales hiring + expansion: 100
   - Active hiring (non-sales): 70
   - Stable: 50
   - Contracting: 20

3. TECH_MODERNIZATION_SCORE (0-100):
   - Recent stack changes: 90
   - Modern stack: 70
   - Mixed stack: 50
   - Legacy-heavy: 30

4. COMPANY_SIZE_SCORE (0-100):
   - 50-500 employees (sweet spot): 100
   - 500-2000: 80
   - 2000-5000: 60
   - <50 or >5000: 40

5. BUDGET_AVAILABILITY_SCORE (0-100):
   - Recent funding + hiring: 100
   - Recent funding OR hiring: 70
   - Stable revenue: 50
   - No signals: 30

Return ONLY this JSON:
{{
    "scores": {{
        "timing": <0-100>,
        "growth": <0-100>,
        "tech_modernization": <0-100>,
        "company_size": <0-100>,
        "budget_availability": <0-100>
    }},
    "weights": {{
        "timing": 0.30,
        "growth": 0.25,
        "tech_modernization": 0.20,
        "company_size": 0.15,
        "budget_availability": 0.10
    }},
    "weighted_score": <calculated weighted average>,
    "confidence": "<high|medium|low>"
}}"""

        scoring_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a scoring algorithm. Return only valid JSON."},
                {"role": "user", "content": scoring_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=800
        )
        
        scores = json.loads(scoring_response.choices[0].message.content.strip())
        
        # Step 3: Generate strategic insights with chain-of-thought
        insight_prompt = f"""You are a senior sales strategist. Given these scores, provide strategic recommendations.

SCORES:
{json.dumps(scores, indent=2)}

COMPANY: {domain}
DATA: {json.dumps(company_data, indent=2)[:500]}

Use chain-of-thought reasoning:
1. Analyze the strongest signals
2. Identify the primary budget trigger
3. Determine the best approach angle
4. Craft a value proposition

Return ONLY this JSON:
{{
    "status": "<GREEN|YELLOW|RED>",
    "reasoning": "<2-3 sentence chain-of-thought explanation>",
    "primary_trigger": "<funding|hiring|tech_debt|expansion>",
    "approach_angle": "<specific recommendation>",
    "evidence": ["<specific fact 1>", "<specific fact 2>", "<specific fact 3>"],
    "recommendation": "<concrete next step>",
    "email_draft": "<personalized 100-word email using the insights>"
}}

STATUS RULES:
- GREEN: weighted_score >= 70
- YELLOW: weighted_score 40-69
- RED: weighted_score < 40"""

        insight_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strategic sales advisor. Use chain-of-thought reasoning."},
                {"role": "user", "content": insight_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=1200
        )
        
        insights = json.loads(insight_response.choices[0].message.content.strip())
        
        # Combine all results
        final_analysis = {
            "score": round(scores['weighted_score']),
            "status": insights['status'],
            "reasoning": insights['reasoning'],
            "evidence": insights['evidence'],
            "recommendation": insights['recommendation'],
            "email_draft": insights['email_draft'],
            "detailed_scores": scores['scores'],
            "confidence": scores['confidence'],
            "primary_trigger": insights['primary_trigger'],
            "approach_angle": insights['approach_angle']
        }
        
        return final_analysis
        
    except Exception as e:
        st.error(f"Groq Advanced Analysis Error: {str(e)}")
        return None

def analyze_with_groq_simple(company_data, market_signals, api_key, domain):
    """SIMPLE: Single-pass analysis (original approach)"""
    try:
        client = Groq(api_key=api_key)
        
        context = {
            "domain": domain,
            "company_info": company_data,
            "market_signals": market_signals
        }
        
        prompt = f"""You are a sales intelligence AI. Analyze this company data and return ONLY a JSON object with this exact structure:

{{
    "score": <number 0-100>,
    "status": "<GREEN|YELLOW|RED>",
    "reasoning": "<2-3 sentence explanation>",
    "evidence": ["<bullet point 1>", "<bullet point 2>", "<bullet point 3>"],
    "recommendation": "<action to take>",
    "email_draft": "<personalized outreach email>"
}}

SCORING LOGIC:
- GREEN (70-100): Recent funding (<6 months) OR active hiring OR expansion signals
- YELLOW (40-69): Stable company, potential tech renewal, moderate signals
- RED (0-39): No recent activity, risk signals, or stagnant

COMPANY DATA:
{json.dumps(context, indent=2)}

Return ONLY the JSON object, no other text."""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a sales intelligence expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = chat_completion.choices[0].message.content
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        json_str = response_text[start_idx:end_idx]
        
        analysis = json.loads(json_str)
        return analysis
        
    except Exception as e:
        st.error(f"Groq Simple Analysis Error: {str(e)}")
        return None

# Analysis mode selection
analysis_mode = st.radio(
    "🧠 Analysis Mode",
    ["Simple (1 API call)", "Advanced (3-step reasoning)"],
    horizontal=True,
    help="Simple: Fast single-pass analysis. Advanced: Multi-dimensional scoring with chain-of-thought reasoning."
)

# Main Analysis Button
if st.button("🚀 Analyze Budget Window", type="primary", use_container_width=True):
    
    # Validate API keys
    missing_keys = []
    if not groq_key:
        missing_keys.append("Groq")
    if not tavily_key:
        missing_keys.append("Tavily")
    if not fullenrich_key:
        missing_keys.append("FullEnrich")
    
    if missing_keys:
        st.error(f"⚠️ Missing API Keys: {', '.join(missing_keys)}. Please add them to .env file or sidebar.")
    elif not domain:
        st.warning("⚠️ Please enter a domain to analyze.")
    else:
        with st.spinner("🔍 Enriching company data..."):
            company_data = get_fullenrich_data(domain, fullenrich_key)
        
        with st.spinner("📡 Gathering market signals..."):
            market_signals = get_market_signals(domain, tavily_key)
        
        if company_data or market_signals:
            if "Advanced" in analysis_mode:
                with st.spinner("🧠 Running advanced multi-step analysis with Groq AI..."):
                    analysis = analyze_with_groq_advanced(company_data, market_signals, groq_key, domain)
            else:
                with st.spinner("🧠 Analyzing with Groq AI (Simple Mode)..."):
                    analysis = analyze_with_groq_simple(company_data, market_signals, groq_key, domain)
            
            if analysis:
                st.success("✅ Analysis Complete!")
                
                # Display Score Card
                score = analysis.get('score', 0)
                status = analysis.get('status', 'YELLOW')
                
                status_class = {
                    'GREEN': 'score-green',
                    'YELLOW': 'score-yellow',
                    'RED': 'score-red'
                }.get(status, 'score-yellow')
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    confidence_badge = ""
                    if 'confidence' in analysis:
                        confidence_badge = f"<p style='margin-top:0.5rem;'>Confidence: {analysis['confidence'].upper()}</p>"
                    
                    st.markdown(f"""
                    <div class="score-card {status_class}">
                        <h2 style="margin:0;">Budget Window Score</h2>
                        <h1 style="font-size:4rem;margin:0.5rem 0;">{score}</h1>
                        <h3 style="margin:0;">Status: {status}</h3>
                        {confidence_badge}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if company_data:
                        revenue = company_data.get('revenue', 'N/A')
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>💰 Revenue</h4>
                            <h2>{revenue}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    if company_data:
                        employees = company_data.get('employees', 'N/A')
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>👥 Employees</h4>
                            <h2>{employees}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                
                # Advanced mode: Show detailed scores
                if "Advanced" in analysis_mode and 'detailed_scores' in analysis:
                    st.subheader("📊 Multi-Dimensional Scores")
                    score_cols = st.columns(5)
                    score_names = ['Timing', 'Growth', 'Tech Mod', 'Size', 'Budget']
                    score_keys = ['timing', 'growth', 'tech_modernization', 'company_size', 'budget_availability']
                    
                    for idx, (col, name, key) in enumerate(zip(score_cols, score_names, score_keys)):
                        with col:
                            score_val = analysis['detailed_scores'].get(key, 0)
                            st.metric(name, f"{score_val}/100")
                    
                    if 'primary_trigger' in analysis:
                        st.info(f"**Primary Trigger:** {analysis['primary_trigger'].replace('_', ' ').title()}")
                    if 'approach_angle' in analysis:
                        st.success(f"**Recommended Approach:** {analysis['approach_angle']}")
                    
                    st.divider()
                
                # Reasoning
                st.subheader("🎯 AI Reasoning")
                st.info(analysis.get('reasoning', 'No reasoning provided.'))
                
                # Evidence
                st.subheader("📊 Key Evidence")
                evidence_items = analysis.get('evidence', [])
                for item in evidence_items:
                    st.markdown(f"<span class='evidence-tag'>✓ {item}</span>", unsafe_allow_html=True)
                
                st.divider()
                
                # Recommendation
                st.subheader("💡 Recommended Action")
                st.success(analysis.get('recommendation', 'No recommendation provided.'))
                
                # Email Draft
                st.subheader("✉️ AI-Generated Outreach Email")
                email_draft = analysis.get('email_draft', 'No email draft available.')
                st.code(email_draft, language=None)
                
                st.divider()
                
                # Raw Data (Expandable)
                with st.expander("🔍 View Raw Data (FullEnrich)"):
                    if company_data:
                        st.json(company_data)
                    else:
                        st.info("No data available from FullEnrich.")
                
                with st.expander("🔍 View Raw Data (Tavily Market Signals)"):
                    if market_signals:
                        st.json(market_signals)
                    else:
                        st.info("No data available from Tavily.")
        else:
            st.error("❌ Failed to gather data. Please check your API keys and try again.")

# Footer
st.divider()
st.caption("Built with ❤️ using 100% Free AI Stack: Streamlit + Groq (Llama 3.3) + Tavily + FullEnrich")
