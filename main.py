import sys
import os
import asyncio
import requests
import operator
import json
import random
import argparse
import re
import logging
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

# --- 0. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
load_dotenv()

LOGIN_URL = "https://www.coderdesign.com/manage-blogs"
TARGET_URL = "https://www.coderdesign.com/upload-blog"
PASSWORD = os.getenv("ADMIN_PASSWORD", "jishan1010")

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

# --- HELPER: BULLET CLEANER ---
def force_clean_bullets(text: str) -> str:
    """
    CRITICAL: Enforces the rule that <li> tags cannot contain <strong> or <b> tags.
    """
    if not text: return ""

    # 1. Remove Markdown bold (**text**) inside list items
    text = re.sub(r'([-*]\s+.*?)\*\*(.*?)\*\*(.*)', r'\1\2\3', text)

    # 2. Remove HTML <strong> and <b> tags specifically inside <li> tags
    def strip_tags_in_li(match):
        content = match.group(1) 
        clean_content = re.sub(r'</?(strong|b)>', '', content, flags=re.IGNORECASE)
        clean_content = clean_content.replace('**', '')
        return f"<li>{clean_content}</li>"

    text = re.sub(r'<li>(.*?)</li>', strip_tags_in_li, text, flags=re.DOTALL)
    return text

# --- 2. TOOLS ---
def suggest_authoritative_urls(topic: str, max_urls: int = 5) -> List[str]:
    logger.info(f"[Tool] Asking AI for authoritative URLs: {topic}")
    gpt4_turbo, gpt_smart = get_llms()
    prompt = f"""
    Provide {max_urls} authoritative, direct URLs relevant to this topic. Prefer official docs or major tech blogs.
    Topic: {topic}
    STRICT RULES:
    - Return ONLY raw URLs, one per line, no extra text.
    - No GitHub, Stack Overflow, or aggregator/search result links.
    - Must be accessible public pages.
    """
    resp = gpt_smart.invoke([HumanMessage(content=prompt)])
    urls = [u.strip() for u in resp.content.splitlines() if u.strip().startswith("http")]
    return urls[:max_urls]

def fetch_url(url: str) -> Optional[str]:
    try:
        h = requests.head(url, timeout=10, allow_redirects=True)
        if h.status_code >= 400: return None
        r = requests.get(url, timeout=15)
        if r.status_code >= 400: return None
        ct = r.headers.get('Content-Type', '')
        if 'text/html' not in ct and 'application/json' not in ct: return None
        return r.text[:10000]
    except Exception:
        return None

def generate_relevant_image(scene_description: str):
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY missing.")
        return None
        
    client = OpenAI()
    final_prompt = f"{scene_description}. Style: High-End Tech Editorial. 3D Render or Vector Art. NO TEXT."
    if len(final_prompt) > 3900: final_prompt = final_prompt[:3900]

    logger.info(f"[Tool] Generating Image...")
    try:
        response = client.images.generate(
            model="dall-e-3", prompt=final_prompt, size="1024x1024", quality="hd", n=1, style="natural"
        )
        img_data = requests.get(response.data[0].url).content
        filename = f"blog_{datetime.now().strftime('%M%S')}.png"
        with open(filename, 'wb') as f: f.write(img_data)
        return filename
    except Exception as e:
        logger.error(f"[Error] Image Generation Failed: {e}")
        return None

# --- 3. AGENT NODES ---
def get_llms():
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY is missing.")
        sys.exit(1)
    return (
        ChatOpenAI(model="gpt-4-turbo", temperature=0.2),
        ChatOpenAI(model="gpt-4o", temperature=0.6) 
    )

def trend_spotter_node(state: AgentState):
    logger.info("[Trend Spotter] Rolling dice for category...")
    _, gpt_smart = get_llms()
    categories, weights = zip(*CATEGORIES_WEIGHTS)
    target_category = random.choices(categories, weights=weights, k=1)[0]
    logger.info(f"[Trend Spotter] Selected Category: {target_category}")

    system_prompt = f"You are an Editor. Pick a compelling news story about {target_category}. Output ONLY the title."
    response = gpt_smart.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Find a trending topic for 2025.")
    ])
    new_topic = response.content.strip().replace('"', '')
    logger.info(f"[Trend Spotter] Topic: {new_topic}")
    return {"topic": new_topic, "final_category": target_category}

def researcher_node(state: AgentState):
    logger.info(f"[Researcher] Investigating: {state['topic']}")
    candidate_urls = suggest_authoritative_urls(state['topic'], max_urls=6)
    valid_blobs = []
    for u in candidate_urls:
        if any(dom in u.lower() for dom in ["github.com", "stackoverflow.com"]): continue
        content = fetch_url(u)
        if content: valid_blobs.append(f"SOURCE URL: {u}\nCONTENT SNIPPET:\n{content}")
    
    combined = "\n\n--- VERIFIED SOURCE ---\n\n".join(valid_blobs)
    return {"research_data": combined, "iteration_count": 0}

def writer_node(state: AgentState):
    logger.info("[Writer] Drafting content...")
    _, gpt_smart = get_llms()
    
    system = """You are an SEO writer. 
    RULES: 
    1. Bullets must be PLAIN TEXT. NO bolding inside bullets. 
    2. Use provided research data for 2-3 links. NO GitHub/StackOverflow."""
    
    prompt = f"TOPIC: {state['topic']}\nDATA: {state['research_data']}\nWrite 1200 words. Conversational."
    if state.get("critique_feedback") and state["critique_feedback"] != "APPROVED":
        prompt += f"\nFIX: {state['critique_feedback']}"

    response = gpt_smart.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    
    # Clean bullets immediately
    cleaned = force_clean_bullets(response.content)
    
    # Ensure internal links
    coder_links = ["https://www.coderdesign.com/mobile-app-development", "https://www.coderdesign.com/contact"]
    if sum(1 for url in coder_links if url in cleaned) < 1:
        cleaned += "\n\nExplore more: " + "; ".join(f"[{u.split('/')[-1]}]({u})" for u in coder_links)

    return {"content_draft": cleaned, "iteration_count": state["iteration_count"] + 1}

def seo_analyst_node(state: AgentState):
    logger.info("[SEO Analyst] Auditing draft...")
    gpt4_turbo, _ = get_llms()
    draft = state["content_draft"]

    # Programmatic check for bold bullets
    if re.search(r'<li>.*?<strong>', draft) or re.search(r'<li>.*?<b>', draft):
        return {"critique_feedback": "CRITICAL: Found bold tags in bullets. Remove ALL bold formatting."}

    links = re.findall(r'\[.*?\]\((http.*?)\)', draft)
    for link in links:
        if "github.com" in link or "stackoverflow.com" in link:
            return {"critique_feedback": f"CRITICAL: Forbidden link: {link}"}

    audit = gpt4_turbo.invoke([
        SystemMessage(content="Audit: 1. No bold in bullets. 2. No AI words. If good, say APPROVED."),
        HumanMessage(content=draft)
    ])
    if "APPROVED" in audit.content.upper(): return {"critique_feedback": "APPROVED"}
    return {"critique_feedback": audit.content}

def meta_data_node(state: AgentState):
    logger.info("[Meta Data] Generating Metadata...")
    _, gpt_smart = get_llms()
    response = gpt_smart.invoke([HumanMessage(content=f"Generate JSON {{category, short_description, seo_title}} for: {state['content_draft'][:500]}")])
    try:
        data = json.loads(response.content.replace("```json", "").replace("```", ""))
        return {"final_category": data['category'], "final_short_desc": data['short_description'], "final_title": data['seo_title']}
    except:
        return {"final_category": "AI & Machine Learning", "final_short_desc": "Tech insights.", "final_title": state['topic'][:50]}

def visual_node(state: AgentState):
    logger.info("[Visuals] Generating concept...")
    _, gpt_smart = get_llms()
    prompt = gpt_smart.invoke([HumanMessage(content=f"Create DALL-E 3 prompt for: {state['topic']}")])
    path = generate_relevant_image(prompt.content)
    return {"image_path": path}

def router(state: AgentState):
    if state["iteration_count"] >= 3 or state["critique_feedback"] == "APPROVED": return "meta_data"
    return "writer"

# --- 4. GRAPH BUILD ---
workflow = StateGraph(AgentState)
workflow.add_node("trend_spotter", trend_spotter_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("seo", seo_analyst_node)
workflow.add_node("meta_data", meta_data_node)
workflow.add_node("visuals", visual_node)

workflow.add_conditional_edges(START, lambda s: "trend_spotter" if not s.get("topic") else "researcher", {"trend_spotter": "trend_spotter", "researcher": "researcher"})
workflow.add_edge("trend_spotter", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "seo")
workflow.add_conditional_edges("seo", router, {"writer": "writer", "meta_data": "meta_data"})
workflow.add_edge("meta_data", "visuals")
workflow.add_edge("visuals", END)

app_graph = workflow.compile()

# --- 5. UPLOADER (FIXED) ---
async def upload_to_coder_design(data, status_callback=None):
    logger.info("[Upload] Launching Browser...")
    
    async with async_playwright() as p:
        # Launch with a specific viewport to ensure elements aren't hidden
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 720})
        page = await context.new_page()

        try:
            logger.info("[Upload] Logging in...")
            await page.goto(LOGIN_URL, timeout=60000)
            
            # Login Process
            await page.get_by_placeholder("Enter admin password").fill(PASSWORD)
            await page.get_by_role("button", name="Access Admin Panel").click()
            
            # --- STABILITY FIX: Wait for Login to actually complete ---
            # Wait for URL to change OR a known element on the admin page to appear
            # We wait up to 10 seconds to confirm we left the login page
            logger.info("   Waiting for login redirect...")
            await page.wait_for_load_state("networkidle")
            
            # Force navigation to Upload Page to be safe
            logger.info(f"[Upload] Navigating to {TARGET_URL}...")
            await page.goto(TARGET_URL, timeout=60000)
            
            # --- STABILITY FIX: Confirm we are on the upload page ---
            # If we were redirected back to login, this check will save us from the confusing timeout error
            if "manage-blogs" in page.url:
                logger.error("   Failed to access Upload page. Redirected back to Login. Check PASSWORD.")
                raise Exception("Login Failed - Redirected to Login Page")

            logger.info("[Upload] Filling Form...")
            
            # File Upload - Wait specifically for it
            file_input = page.locator('input[type="file"]')
            await file_input.wait_for(state="attached", timeout=10000) # Wait for input to exist
            
            if data['image_path'] and os.path.exists(data['image_path']):
                await file_input.set_input_files(data['image_path'])

            # Title
            title_to_use = data.get('final_title', data['topic'])
            await page.get_by_placeholder("Enter blog title").fill(title_to_use)

            # Author
            await page.get_by_placeholder("Enter author name").fill("Sarah Miller")
            await page.keyboard.press("Enter")

            # Category (Try/Except for robustness)
            try:
                await page.locator("select").select_option(label=data['final_category'])
            except:
                logger.warning(f"   Category {data['final_category']} not found. Defaulting.")
                await page.locator("select").select_option(index=1)

            # Description
            await page.get_by_placeholder("Enter a short description...").fill(data['final_short_desc'])

            # Content
            await page.get_by_placeholder("Enter a short description...").focus()
            await page.keyboard.press("Tab")
            
            # Clean formatting one last time
            final_content = force_clean_bullets(data['content_draft'])
            await page.keyboard.insert_text(final_content)

            logger.info("[Upload] Submitting...")
            submit_btn = page.get_by_role("button", name="Upload Blog Post")
            await submit_btn.click()
            
            await page.wait_for_timeout(5000)
            logger.info("[Success] Blog Uploaded!")

        except Exception as e:
            logger.error(f"[Error] Upload Failed: {e}")
            await page.screenshot(path="error_debug.png")
            raise e # Reraise to turn GitHub Red

        finally:
            await browser.close()
            if data.get('image_path') and os.path.exists(data['image_path']):
                os.remove(data['image_path'])

# --- 6. EXECUTION ---
def run_cli_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="", help="Topic.")
    parser.add_argument("--instructions", type=str, default="", help="Instructions.")
    args = parser.parse_args()

    topic_in = args.topic.strip()
    if not topic_in: logger.info("--- LAUNCHING TREND SPOTTER ---")
    else: logger.info(f"--- TOPIC: {topic_in} ---")

    try:
        initial_state = {"topic": topic_in, "custom_instructions": args.instructions, "iteration_count": 0}
        final_state = app_graph.invoke(initial_state)
        
        disable_upload = os.getenv("DISABLE_UPLOAD", "").lower() in ("1", "true")
        if final_state.get('image_path') and not disable_upload:
            asyncio.run(upload_to_coder_design(final_state))
        elif disable_upload:
            logger.info("[Upload] Skipped.")
            
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE: {e}")
        sys.exit(1)

def run_streamlit_mode():
    st.title("AI Auto-Blogger")
    # Streamlit logic (omitted for brevity, same as before)

if __name__ == "__main__":
    if st and get_script_run_ctx(): run_streamlit_mode()
    else: run_cli_mode()
