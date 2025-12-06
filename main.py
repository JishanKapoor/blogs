import sys
import os
import asyncio
import requests
import operator
import json
import random
import argparse
import re
from datetime import datetime
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

# Logic & AI Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# Automation Imports
from playwright.async_api import async_playwright

# Streamlit Import
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    st = None

# --- CONFIGURATION ---
load_dotenv()

LOGIN_URL = "https://www.coderdesign.com/manage-blogs"
TARGET_URL = "https://www.coderdesign.com/upload-blog"
PASSWORD = "jishan1010"

CATEGORIES_WEIGHTS = [
    ("AI & Machine Learning", 0.30),
    ("Full-Stack Development", 0.25),
    ("AI SEO & AEO Services", 0.20),
    ("Mobile App Development", 0.25)
]


# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    topic: str
    custom_instructions: str
    messages: Annotated[List[BaseMessage], operator.add]
    research_data: str
    content_draft: str
    critique_feedback: Optional[str]
    image_path: str
    iteration_count: int
    final_category: str
    final_short_desc: str
    final_title: str


# --- 2. TOOLS ---
def suggest_authoritative_urls(topic: str, max_urls: int = 8) -> List[str]:
    print(f"[Tool] Finding verified URLs for: {topic}")
    gpt4_turbo, gpt5.1 = get_llms()
    prompt = f"""
    Provide {max_urls} EXTREMELY reliable, high-level URLs for this topic.
    Topic: {topic}
    
    CRITICAL RULES FOR LINK SELECTION:
    1. Prefer **ROOT DOMAINS** or **MAIN DOCUMENTATION** pages (e.g., "https://www.nasa.gov", "https://cloud.google.com/dialogflow").
    2. AVOID specific deep article links (they often return 404).
    3. NO GitHub, NO Stack Overflow, NO aggregators.
    4. Must be major official sources (e.g., Microsoft, Google, NASA, OpenAI, Stripe, AWS).
    
    Output ONLY raw URLs, one per line.
    """
    resp = gpt5.1.invoke([HumanMessage(content=prompt)])
    urls = [u.strip() for u in resp.content.splitlines() if u.strip().startswith("http")]
    return urls[:max_urls]

def fetch_url(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        try:
            h = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
            if h.status_code >= 400:
                return None
        except:
            pass 

        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code >= 400:
            return None
            
        ct = r.headers.get('Content-Type', '').lower()
        if 'text/html' not in ct and 'application/json' not in ct:
            return None
            
        return r.text[:8000] 
    except Exception:
        return None


def generate_relevant_image(scene_description: str):
    from openai import OpenAI
    client = OpenAI()

    style_instruction = "Style: Minimalist Tech Editorial. 3D Render or Clean Vector. High quality. NO TEXT IN IMAGE."
    final_prompt = f"{scene_description}. {style_instruction}"

    if len(final_prompt) > 3900:
        final_prompt = final_prompt[:3900]

    print(f"[Tool] Generating Image...")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            size="1024x1024",
            quality="hd",
            n=1,
            style="natural"
        )
        img_data = requests.get(response.data[0].url).content
        filename = f"blog_{datetime.now().strftime('%M%S')}.png"
        with open(filename, 'wb') as f:
            f.write(img_data)
        return filename
    except Exception as e:
        print(f"[Error] Image Generation Failed: {e}")
        return None


# --- 3. AGENT NODES ---
def get_llms():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing.")
    return (
        ChatOpenAI(model="gpt-4-turbo", temperature=0.2),
        ChatOpenAI(model="gpt-4o", temperature=0.5)
    )


def trend_spotter_node(state: AgentState):
    print("[Trend Spotter] Selecting category...")
    _, gpt5.1 = get_llms()

    categories, weights = zip(*CATEGORIES_WEIGHTS)
    target_category = random.choices(categories, weights=weights, k=1)[0]
    
    system_prompt = f"""You are an Editor-in-Chief. 
    Generate a compelling, modern tech topic for the category: '{target_category}'.
    Focus on: Emerging trends, controversy, or "How-To" guides for 2025.
    Output ONLY the topic title."""

    response = gpt5.1.invoke([SystemMessage(content=system_prompt)])
    new_topic = response.content.strip().replace('"', '')
    print(f"[Trend Spotter] Topic: {new_topic}")

    return {"topic": new_topic, "final_category": target_category}


def researcher_node(state: AgentState):
    print(f"[Researcher] Verifying links for: {state['topic']}")
    
    candidate_urls = suggest_authoritative_urls(state['topic'], max_urls=8)
    valid_blobs = []
    working_urls = []
    
    for u in candidate_urls:
        if len(working_urls) >= 3:
            break
        if any(dom in u.lower() for dom in ["github.com", "stackoverflow.com", "stackexchange.com", "reddit.com"]):
            continue
            
        content = fetch_url(u)
        if content:
            working_urls.append(u)
            snippet = content[:500].replace("\n", " ")
            valid_blobs.append(f"URL: {u}\nCONTEXT: {snippet}")
            print(f"   ✓ Valid: {u}")
        else:
            print(f"   ✗ Invalid: {u}")

    combined_data = "\n\n".join(valid_blobs)
    return {"research_data": combined_data, "iteration_count": 0}


def writer_node(state: AgentState):
    print("[Writer] Drafting content...")
    _, gpt5.1 = get_llms()

    topic = state["topic"]
    feedback = state.get("critique_feedback", None)

    system = """You are a professional tech blogger.

    **STYLE GUIDE:**
    1. **Tone:** Friendly, informative, and direct. Use "You".
    2. **Structure:**
       - H1 Title
       - Intro (Short paragraphs)
       - H2 Sections
       - H3 Sub-sections (Break up text!)
       - Lists (<li> items)
       - **FAQ Section (Mandatory)** at the end.
    3. **Links:** EXACTLY 3 valid external links from research data.
    
    **CRITICAL FORMATTING RULE FOR LISTS:**
    - You CAN use bold for emphasis inside a sentence.
    - **BUT**: You MUST NOT start a bullet point with a bold term immediately.
    - **BAD**: `<li><strong>Title</strong>: Description</li>` (Do not do this!)
    - **GOOD**: `<li>Title is important because...</li>`
    - **GOOD**: `<li>Make sure to <strong>emphasize</strong> this part.</li>` (Bold inside is OK)
    
    **FAQ REQUIREMENT:**
    - Include `<h2>Frequently Asked Questions</h2>` section before Conclusion.
    - At least 4 questions.
    """

    prompt = f"""
    TOPIC: {topic}
    RESEARCH DATA: {state['research_data']}
    
    Instructions:
    - Write 1200+ words.
    - Use H3 subheadings frequently.
    - Add FAQ section.
    - Include 3 real links.
    - **Don't start bullets with bold tags.**
    """

    if feedback and feedback != "APPROVED":
        prompt += f"\n\nFEEDBACK FROM EDITOR: {feedback}"

    response = gpt5.1.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    content = response.content

    # --- SMART SANITIZER ---
    # Removes bold ONLY if it is at the START of the list item
    # Matches: <li>   <strong>...
    def clean_leading_bold(match):
        item_content = match.group(1)
        
        # 1. Check HTML <li><strong> or <li><b>
        if re.match(r'^\s*<(strong|b)>', item_content, re.IGNORECASE):
            # Strip the OUTER bold tags only
            item_content = re.sub(r'^\s*<(strong|b)>(.*?)</\1>', r'\2', item_content, count=1, flags=re.IGNORECASE)

        # 2. Check Markdown - **
        if re.match(r'^\s*\*\*', item_content):
            item_content = re.sub(r'^\s*\*\*(.*?)\*\*', r'\1', item_content, count=1)
            
        return f"<li>{item_content}</li>"

    # Apply to all LI items
    content = re.sub(r'<li>(.*?)</li>', clean_leading_bold, content, flags=re.DOTALL)

    # Add Internal Links if missing
    internal_links = [
        "https://www.coderdesign.com/mobile-app-development",
        "https://www.coderdesign.com/full-stack-engineering",
        "https://www.coderdesign.com/ai-workflow",
    ]
    if "coderdesign.com" not in content:
        content += "\n\n<p>Interested in building this? Check out our <a href='https://www.coderdesign.com/ai-workflow'>AI Services</a>.</p>"

    return {"content_draft": content, "iteration_count": state["iteration_count"] + 1}


def seo_analyst_node(state: AgentState):
    print("[SEO Analyst] Auditing...")
    draft = state["content_draft"]
    
    # Check 1: FAQ
    if "Frequently Asked Questions" not in draft and "FAQs" not in draft:
        return {"critique_feedback": "CRITICAL: Missing 'Frequently Asked Questions' section."}

    # Check 2: Leading Bold in Lists (Adjacency check)
    # Finds <li> followed immediately by <strong/b> (ignoring whitespace)
    if re.search(r'<li>\s*<(strong|b)>', draft, re.DOTALL | re.IGNORECASE):
        # Auto-fix: Strip leading bold tags
        def fix_leading(m):
            return re.sub(r'^\s*<(strong|b)>(.*?)</\1>', r'\2', m.group(1), count=1, flags=re.IGNORECASE)
            
        draft = re.sub(r'<li>(.*?)</li>', lambda m: f"<li>{fix_leading(m)}</li>", draft, flags=re.DOTALL)
        return {"content_draft": draft, "critique_feedback": "APPROVED"}

    # Check 3: Links
    link_count = len(re.findall(r'href=["\']http', draft))
    if link_count < 3:
        return {"critique_feedback": f"Found only {link_count} links. Need exactly 3."}

    return {"critique_feedback": "APPROVED"}


def meta_data_node(state: AgentState):
    print("[Meta Data] Generating Metadata...")
    _, gpt5.1 = get_llms()
    draft = state["content_draft"]

    prompt = f"""
    Generate JSON metadata:
    1. "category": [AI & Machine Learning, Full-Stack Development, Mobile App Development, AI SEO & AEO Services]
    2. "short_description": 2 sentences.
    3. "seo_title": Under 50 chars.
    
    Excerpt: {draft[:500]}
    """
    response = gpt5.1.invoke([HumanMessage(content=prompt)])
    
    try:
        clean = response.content.replace("```json", "").replace("```", "")
        data = json.loads(clean)
        return {
            "final_category": state.get('final_category', data['category']),
            "final_short_desc": data['short_description'],
            "final_title": data['seo_title']
        }
    except:
        return {
            "final_category": "AI & Machine Learning",
            "final_short_desc": "Tech insights.",
            "final_title": state['topic'][:50]
        }


def visual_node(state: AgentState):
    print("[Visuals] Generating Image...")
    _, gpt5.1 = get_llms()
    prompt = f"Create a minimalist, high-tech image prompt for: {state['topic']}. No text."
    resp = gpt5.1.invoke([HumanMessage(content=prompt)])
    path = generate_relevant_image(resp.content)
    return {"image_path": path}


def router(state: AgentState):
    if state["critique_feedback"] == "APPROVED": return "meta_data"
    if state["iteration_count"] > 2: return "meta_data" 
    return "writer"


# --- 4. GRAPH BUILD ---
workflow = StateGraph(AgentState)
workflow.add_node("trend_spotter", trend_spotter_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("seo", seo_analyst_node)
workflow.add_node("meta_data", meta_data_node)
workflow.add_node("visuals", visual_node)

def check_topic(state: AgentState):
    if not state.get("topic"): return "trend_spotter"
    return "researcher"

workflow.add_conditional_edges(START, check_topic, {"trend_spotter": "trend_spotter", "researcher": "researcher"})
workflow.add_edge("trend_spotter", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "seo")
workflow.add_conditional_edges("seo", router, {"writer": "writer", "meta_data": "meta_data"})
workflow.add_edge("meta_data", "visuals")
workflow.add_edge("visuals", END)

app_graph = workflow.compile()


# --- 5. UPLOADER ---
async def upload_to_coder_design(data, status_callback=None):
    print("[Upload] Starting upload process...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto(LOGIN_URL)
            await page.get_by_placeholder("Enter admin password").fill(PASSWORD)
            await page.get_by_role("button", name="Access Admin Panel").click()
            await page.wait_for_load_state("networkidle")

            if "upload-blog" not in page.url:
                await page.goto(TARGET_URL)

            if data['image_path'] and os.path.exists(data['image_path']):
                await page.locator('input[type="file"]').set_input_files(data['image_path'])

            await page.get_by_placeholder("Enter blog title").fill(data.get('final_title', data['topic']))
            await page.get_by_placeholder("Enter author name").fill("Sarah Miller") 
            await page.keyboard.press("Enter")
            
            await page.locator("select").select_option(label=data['final_category'])
            await page.get_by_placeholder("Enter a short description...").fill(data['final_short_desc'])
            await page.get_by_placeholder("Enter a short description...").focus()
            await page.keyboard.press("Tab")
            
            await page.keyboard.insert_text(data['content_draft'])

            await page.get_by_role("button", name="Upload Blog Post").click()
            await page.wait_for_timeout(5000)
            print("[Success] Blog Uploaded Successfully!")
            if status_callback: status_callback("Upload Complete!")

        except Exception as e:
            print(f"[Error] Upload failed: {e}")
        finally:
            await browser.close()
            if data['image_path'] and os.path.exists(data['image_path']):
                os.remove(data['image_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="", help="Topic")
    args = parser.parse_args()
    
    initial = {"topic": args.topic, "iteration_count": 0}
    final = app_graph.invoke(initial)
    if final.get('image_path'):
        asyncio.run(upload_to_coder_design(final))
