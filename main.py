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
# Make sure this environment variable is set in your .env file
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

# --- 2. TOOLS ---
def get_llms():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing.")
    # UPDATED: Changed gpt-5.1 to gpt-4o (current flagship)
    return (
        ChatOpenAI(model="gpt-4-turbo", temperature=0.2),
        ChatOpenAI(model="gpt-4o", temperature=0.6) 
    )

def suggest_authoritative_urls(topic: str, max_urls: int = 5) -> List[str]:
    print(f"[Tool] Asking AI for authoritative URLs: {topic}")
    gpt4_turbo, gpt4o = get_llms()
    prompt = f"""
    Provide {max_urls} authoritative, direct URLs relevant to this topic. Prefer official docs or major tech blogs.
    Topic: {topic}
    STRICT RULES:
    - Return ONLY raw URLs, one per line, no extra text.
    - No GitHub, Stack Overflow, or aggregator/search result links.
    - Must be accessible public pages (not behind auth).
    """
    resp = gpt4o.invoke([HumanMessage(content=prompt)])
    urls = [u.strip() for u in resp.content.splitlines() if u.strip().startswith("http")]
    return urls[:max_urls]

def fetch_url(url: str) -> Optional[str]:
    try:
        h = requests.head(url, timeout=15, allow_redirects=True)
        if h.status_code >= 400: return None
        r = requests.get(url, timeout=20)
        if r.status_code >= 400: return None
        ct = r.headers.get('Content-Type', '')
        if 'text/html' not in ct and 'application/json' not in ct: return None
        return r.text[:10000]
    except Exception:
        return None

def generate_relevant_image(scene_description: str):
    from openai import OpenAI
    client = OpenAI()
    
    style_instruction = "Style: High-End Tech Editorial. 3D Render style or Detailed Vector Art. Focus on the SUBJECT MATTER. NO TEXT."
    final_prompt = f"{scene_description}. {style_instruction}"
    
    if len(final_prompt) > 3900:
        final_prompt = final_prompt[:3900]

    print(f"[Tool] Generating Image: {final_prompt[:60]}...")
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

def trend_spotter_node(state: AgentState):
    print("[Trend Spotter] User did not provide a topic. Rolling dice...")
    gpt4_turbo, gpt4o = get_llms()

    categories, weights = zip(*CATEGORIES_WEIGHTS)
    target_category = random.choices(categories, weights=weights, k=1)[0]
    
    search_results = f"AI-selected category: {target_category}."

    system_prompt = f"""You are an Editor-in-Chief. Pick a compelling, conversation-worthy news story topic about {target_category} for 2025.
    Output ONLY the topic title.
    """

    response = gpt4o.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the latest news:\n{search_results}")
    ])

    new_topic = response.content.strip().replace('"', '')
    print(f"[Trend Spotter] Topic: {new_topic}")
    return {"topic": new_topic, "final_category": target_category}

def researcher_node(state: AgentState):
    print(f"[Researcher] Investigating: {state['topic']}")
    candidate_urls = suggest_authoritative_urls(state['topic'], max_urls=6)
    valid_blobs = []
    
    for u in candidate_urls:
        if any(dom in u.lower() for dom in ["github.com", "stackoverflow.com", "stackexchange.com"]):
            continue
        content = fetch_url(u)
        if content:
            valid_blobs.append(f"SOURCE URL: {u}\nCONTENT SNIPPET:\n{content}")
            
    combined_data = "\n\n--- VERIFIED SOURCE ---\n\n".join(valid_blobs)
    return {"research_data": combined_data, "iteration_count": 0}

def writer_node(state: AgentState):
    print("[Writer] Drafting content...")
    gpt4_turbo, gpt4o = get_llms()

    topic = state["topic"]
    instructions = state["custom_instructions"]
    feedback = state.get("critique_feedback", None)

    system = """You are an expert SEO content writer. Write a 1200-1500 word article.
    
    CRITICAL FORMATTING RULES FOR BULLETS:
    - ABSOLUTE ZERO TOLERANCE for bolding inside bullets.
    - NO `**bold**`, NO `<strong>`, NO `<b>`.
    - Format: `<li>Plain text sentence</li>`
    - Do NOT write: `<li><strong>Title</strong>: content</li>`
    """

    prompt = f"""
    TOPIC: {topic}
    INSTRUCTIONS: {instructions}
    RESEARCH DATA: {state['research_data']}
    
    REQUIREMENTS:
    1. Human-like tone, 1200+ words.
    2. Use 2-3 links from RESEARCH DATA only.
    3. BULLETS MUST BE PLAIN TEXT. NO BOLDING INSIDE BULLETS.
    """

    if feedback and feedback != "APPROVED":
        prompt += f"\n\nFIX PREVIOUS ERRORS: {feedback}"

    response = gpt4o.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    
    raw_content = response.content

    # --- FIX: ROBUST BULLET SANITIZER ---
    def force_remove_bold_in_bullets(text: str) -> str:
        # 1. Handle HTML <li> bullets
        def clean_html_match(m):
            content = m.group(1)
            # Remove HTML bold tags
            content = re.sub(r'</?(strong|b)>', '', content, flags=re.IGNORECASE)
            # Remove Markdown bold/italic markers
            content = content.replace('**', '').replace('__', '')
            # Clean "Title:" patterns if they exist
            content = re.sub(r'^[^:]+:\s*', '', content) 
            return f"<li>{content}</li>"
        
        text = re.sub(r'<li>(.*?)</li>', clean_html_match, text, flags=re.DOTALL)

        # 2. Handle Markdown bullets (- or *)
        def clean_md_match(m):
            content = m.group(1)
            content = re.sub(r'</?(strong|b)>', '', content, flags=re.IGNORECASE)
            content = content.replace('**', '').replace('__', '')
            content = re.sub(r'^[^:]+:\s*', '', content)
            return f"- {content}"

        text = re.sub(r'^[\-\*]\s+(.+)$', clean_md_match, text, flags=re.MULTILINE)
        return text

    cleaned = force_remove_bold_in_bullets(raw_content)

    # Ensure internal links exist
    coder_links = [
        "https://www.coderdesign.com/mobile-app-development",
        "https://www.coderdesign.com/full-stack-engineering",
    ]
    present = sum(1 for url in coder_links if url in cleaned)
    if present < 1:
        cleaned += "\n\nExplore more: " + "; ".join(f"[{u.split('/')[-1]}]({u})" for u in coder_links)

    return {"content_draft": cleaned, "iteration_count": state["iteration_count"] + 1}

def seo_analyst_node(state: AgentState):
    print("[SEO Analyst] Auditing draft...")
    gpt4_turbo, _ = get_llms()
    draft = state["content_draft"]
    
    # 1. Check for Bold Bullets (Double Check)
    html_bold_bullets = re.findall(r'<li>.*?<strong>.*?</strong>.*?</li>', draft, re.DOTALL)
    markdown_bold_bullets = re.findall(r'[-*]\s+\*\*[^*]+\*\*', draft)
    
    if html_bold_bullets or markdown_bold_bullets:
        return {"critique_feedback": "CRITICAL: Found BOLD text inside bullet points. Remove all ** and <strong> tags from lists."}

    # 2. Check Word Count
    if len(draft.split()) < 1000:
        return {"critique_feedback": "Draft is too short. Expand to 1200+ words."}

    # 3. Check Links Validity
    research_urls = re.findall(r'https?://[^\s\')]+', state.get('research_data', ''))
    links_found = re.findall(r'\[.*?\]\((http.*?)\)', draft)
    
    # Simple validation
    if len(links_found) < 2:
         return {"critique_feedback": "Add more external links from the research data."}

    return {"critique_feedback": "APPROVED"}

def meta_data_node(state: AgentState):
    print("[Meta Data] Generating Metadata...")
    gpt4_turbo, gpt4o = get_llms()
    draft = state["content_draft"]

    prompt = f"""
    Generate JSON for: "category", "short_description", "seo_title" (max 50 chars).
    Blog start: {draft[:1000]}
    """
    response = gpt4o.invoke([HumanMessage(content=prompt)])
    
    try:
        clean_json = response.content.replace("```json", "").replace("```", "")
        data = json.loads(clean_json)
        return {
            "final_category": state.get('final_category') or data['category'],
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
    print("[Visuals] Generating concept...")
    gpt4_turbo, gpt4o = get_llms()
    
    prompt_request = f"""
    Create a DALL-E 3 prompt for: {state['topic']}.
    Specific, no text, 3D render style.
    """
    image_prompt_generator = gpt4o.invoke([HumanMessage(content=prompt_request)])
    path = generate_relevant_image(image_prompt_generator.content)
    return {"image_path": path}

def router(state: AgentState):
    if state["iteration_count"] >= 3: return "meta_data"
    if state["critique_feedback"] == "APPROVED": return "meta_data"
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
    return "researcher" if state.get("topic") else "trend_spotter"

workflow.add_conditional_edges(START, check_topic, {
    "trend_spotter": "trend_spotter", "researcher": "researcher"
})
workflow.add_edge("trend_spotter", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "seo")
workflow.add_conditional_edges("seo", router, {"writer": "writer", "meta_data": "meta_data"})
workflow.add_edge("meta_data", "visuals")
workflow.add_edge("visuals", END)

app_graph = workflow.compile()

# --- 5. UPLOADER ---
async def upload_to_coder_design(data, status_callback=None):
    if status_callback: status_callback("[Upload] Launching Browser...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

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
            
            # Content fill
            await page.get_by_placeholder("Enter a short description...").focus()
            await page.keyboard.press("Tab")
            await page.keyboard.insert_text(data['content_draft'])

            if status_callback: status_callback("[Upload] Submitting...")
            await page.get_by_role("button", name="Upload Blog Post").click()
            await page.wait_for_timeout(5000)

            if status_callback: status_callback("[Success] Uploaded!")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()
            if data['image_path'] and os.path.exists(data['image_path']):
                os.remove(data['image_path'])

# --- 6. EXECUTION ---
def run_cli_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="")
    args = parser.parse_args()
    
    initial = {"topic": args.topic, "custom_instructions": "", "iteration_count": 0}
    final = app_graph.invoke(initial)
    if final.get('image_path'):
        asyncio.run(upload_to_coder_design(final))

def run_streamlit_mode():
    st.title("AI Auto-Blogger")
    if st.button("Generate"):
        initial = {"topic": "", "custom_instructions": "", "iteration_count": 0}
        with st.status("Working..."):
            final = app_graph.invoke(initial)
            st.markdown(final['content_draft'])
            asyncio.run(upload_to_coder_design(final))

if __name__ == "__main__":
    is_streamlit = st is not None and get_script_run_ctx() is not None
    if is_streamlit:
        run_streamlit_mode()
    else:
        run_cli_mode()
