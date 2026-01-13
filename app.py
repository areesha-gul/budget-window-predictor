import streamlit as st
import requests
import json
from groq import Groq
from tavily import TavilyClient
import pandas as pd
from datetime import datetime

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

# Sidebar for API Keys
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("**Get your FREE keys:**")
    st.markdown("- [Groq Console](https://console.groq.com/keys)")
    st.markdown("- [Tavily](https://tavily.com/)")
    st.markdown("- [FullEnrich](https://fullenrich.com/)")
    
    st.divider()
    
    groq_key = st.text_input("Groq API Key", type="password", help="Free forever at console.groq.com")
    tavily_key = st.text_input("Tavily API Key", type="password", help="Free tier at tavily.com")
    fullenrich_key = st.text_input("FullEnrich API Key", type="password", help="Sign up at fullenrich.com")
    
    st.divider()
    st.caption("💰 Total Cost: **$0.00** per analysis")

# Main content
domain = st.text_input("🌐 Enter Target Domain", placeholder="e.g., stripe.com", help="Company domain to analyze")

def get_fullenrich_data(domain, api_key):
    """Fetch company and contact data from FullEnrich API"""
    try:
        url = "https://api.fullenrich.com/v1/company/enrich"
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

def analyze_with_groq(company_data, market_signals, api_key, domain):
    """Send data to Groq (Llama 3.3) for intelligent scoring"""
    try:
        client = Groq(api_key=api_key)
        
        # Prepare the context
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
                {
                    "role": "system",
                    "content": "You are a sales intelligence expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Extract JSON from response (in case there's any surrounding text)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        json_str = response_text[start_idx:end_idx]
        
        analysis = json.loads(json_str)
        return analysis
        
    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        return None

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
        st.error(f"⚠️ Missing API Keys: {', '.join(missing_keys)}. Please add them in the sidebar.")
    elif not domain:
        st.warning("⚠️ Please enter a domain to analyze.")
    else:
        with st.spinner("🔍 Enriching company data..."):
            company_data = get_fullenrich_data(domain, fullenrich_key)
        
        with st.spinner("📡 Gathering market signals..."):
            market_signals = get_market_signals(domain, tavily_key)
        
        if company_data or market_signals:
            with st.spinner("🧠 Analyzing with Groq AI (Llama 3.3)..."):
                analysis = analyze_with_groq(company_data, market_signals, groq_key, domain)
            
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
                    st.markdown(f"""
                    <div class="score-card {status_class}">
                        <h2 style="margin:0;">Budget Window Score</h2>
                        <h1 style="font-size:4rem;margin:0.5rem 0;">{score}</h1>
                        <h3 style="margin:0;">Status: {status}</h3>
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
