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
from typing import TypedDict, Annotated, List, Optional, Dict
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
# Use hardcoded password in Playwright to avoid env issues
PASSWORD_VALUE = "jishan1010" 

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
    research_data: str
    
    # New Fields for Long-Form Writing
    outline_sections: List[str]  # Holds the headlines for the blog
    current_section_index: int   # Tracks which section we are writing
    draft_parts: Annotated[List[str], operator.add] # Accumulates the blog parts
    
    final_content: str
    critique_feedback: Optional[str]
    image_path: str
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
    _, gpt_smart = get_llms()
    prompt = f"""
    Provide {max_urls} authoritative, direct URLs relevant to this topic. 
    Topic: {topic}
    RULES:
    - Official documentation (React docs, AWS docs, Google AI paper).
    - High-quality tech blogs (Martin Fowler, Uber Engineering, Netflix Tech).
    - NO GitHub repos, NO StackOverflow, NO Aggregators.
    - Return ONLY the raw URL per line.
    """
    resp = gpt_smart.invoke([HumanMessage(content=prompt)])
    urls = [u.strip() for u in resp.content.splitlines() if u.strip().startswith("http")]
    return urls[:max_urls]

def fetch_url(url: str) -> Optional[str]:
    try:
        h = requests.head(url, timeout=5, allow_redirects=True)
        if h.status_code >= 400: return None
        r = requests.get(url, timeout=10)
        if r.status_code >= 400: return None
        ct = r.headers.get('Content-Type', '')
        if 'text/html' not in ct and 'application/json' not in ct: return None
        return r.text[:8000] # Cap text to avoid token overflow
    except Exception:
        return None

def generate_relevant_image(scene_description: str):
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY missing.")
        return None
        
    client = OpenAI()
    final_prompt = f"{scene_description}. Style: High-End Tech Editorial, Abstract 3D Render, Soft Lighting. NO TEXT."
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
    # Using 'gpt-4o' for main writing (best for creative flow) and 'gpt-4-turbo' for logic
    return (
        ChatOpenAI(model="gpt-4-turbo", temperature=0.3),
        ChatOpenAI(model="gpt-4o", temperature=0.7) 
    )

def trend_spotter_node(state: AgentState):
    logger.info("[Trend Spotter] Rolling dice for category...")
    _, gpt_smart = get_llms()
    categories, weights = zip(*CATEGORIES_WEIGHTS)
    target_category = random.choices(categories, weights=weights, k=1)[0]
    logger.info(f"[Trend Spotter] Selected Category: {target_category}")

    # Generate a specific, spicy angle, not just a generic topic
    system_prompt = f"""You are a Senior Editor for a high-end tech publication.
    Goal: Find a specific, opinionated, and deep technical topic about {target_category}.
    
    BAD TOPICS: "What is AI?", "Benefits of React" (Too boring, too generic)
    GOOD TOPICS: "Why Micro-Frontends Are Killing Your Performance", "The Hidden Costs of Fine-Tuning LLMs in 2025"
    
    Output ONLY the title.
    """
    response = gpt_smart.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Give me a trending, specific, expert-level topic.")
    ])
    new_topic = response.content.strip().replace('"', '')
    logger.info(f"[Trend Spotter] Topic: {new_topic}")
    return {"topic": new_topic, "final_category": target_category}

def researcher_node(state: AgentState):
    logger.info(f"[Researcher] Deep diving into: {state['topic']}")
    candidate_urls = suggest_authoritative_urls(state['topic'], max_urls=5)
    valid_blobs = []
    
    for u in candidate_urls:
        content = fetch_url(u)
        if content: 
            # We add a label to the snippet so the writer knows where it came from
            valid_blobs.append(f"SOURCE_LINK: {u}\nCONTENT: {content[:2000]}...")
    
    combined = "\n\n".join(valid_blobs)
    if not combined:
        combined = "No external sources found. Rely on expert internal knowledge."
        
    return {"research_data": combined}

def outliner_node(state: AgentState):
    """
    Instead of writing immediately, we plan the blog post.
    We generate a list of 5-7 distinct section headers.
    """
    logger.info("[Architect] Planning the Outline...")
    gpt_logic, _ = get_llms()
    
    prompt = f"""
    You are an Expert Technical Writer. Plan a 2000-word deep-dive article on: {state['topic']}
    
    Target Audience: Senior Developers, CTOs, and Technical Leads.
    Tone: Professional, Insightful, "No Fluff".
    
    Create an outline with 6 to 8 headers.
    1. Introduction (Must have a Hook)
    2. Deep Dive Section 1
    3. Deep Dive Section 2
    4. Technical Implementation / Real World Strategy
    5. Challenges & Solutions
    6. Future Outlook
    7. Conclusion (Next Steps)
    
    OUTPUT FORMAT: JSON List of strings ONLY.
    Example: ["Introduction: The Truth About X", "Why Legacy Systems Fail", "Step-by-Step Implementation", "Conclusion"]
    """
    
    response = gpt_logic.invoke([HumanMessage(content=prompt)])
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    try:
        sections = json.loads(clean_json)
    except:
        # Fallback if JSON fails
        sections = [
            f"Introduction: Unpacking {state['topic']}",
            "The Core Problem with Current Solutions",
            "Advanced Strategies for 2025",
            "Technical Implementation Guide",
            "Real-World Case Studies",
            "Conclusion: The Path Forward"
        ]
        
    logger.info(f"[Architect] Outline created with {len(sections)} sections.")
    return {"outline_sections": sections, "current_section_index": 0, "draft_parts": []}

def section_writer_node(state: AgentState):
    """
    Writes ONE section at a time to ensure depth and length.
    """
    _, gpt_writer = get_llms()
    
    current_idx = state["current_section_index"]
    sections = state["outline_sections"]
    current_header = sections[current_idx]
    
    logger.info(f"[Writer] Writing Section {current_idx + 1}/{len(sections)}: {current_header}")
    
    # Check if this is the intro or conclusion to adjust tone
    is_intro = current_idx == 0
    is_conclusion = current_idx == len(sections) - 1
    
    prompt = f"""
    You are writing a SECTION for a long-form technical article.
    TOPIC: {state['topic']}
    CURRENT SECTION HEADER: {current_header}
    
    CONTEXT (Research):
    {state['research_data']}
    
    PREVIOUS CONTENT CONTEXT:
    {''.join(state['draft_parts'][-1:]) if state['draft_parts'] else "Start of article."}
    
    WRITING RULES:
    1. **Length**: Write 300-500 words for this section alone.
    2. **Tone**: Human, expert, slightly opinionated. NOT "AI generic".
       - Use anecdotes: "I once saw a project fail because..."
       - Use rhetorical questions.
       - Use short punchy sentences mixed with long ones.
    3. **Links**: If you use a URL from the Research Context, EXPLAIN IT.
       - WRONG: "Check this link [URL]."
       - RIGHT: "As documented in the official React docs [URL], the concurrent mode..."
    4. **Formatting**:
       - Use <h3> for sub-sub-headings if needed.
       - **BULLETS**: MUST BE PLAIN TEXT. NO BOLDING INSIDE BULLETS.
    5. **Forbidden**: Do not use "In this section", "Delve", "In the ever-evolving world".
    
    {'Start with a strong hook, a story, or a contrarian statement.' if is_intro else 'Dive straight into the details.'}
    """
    
    response = gpt_writer.invoke([HumanMessage(content=prompt)])
    
    # Clean the output immediately
    clean_text = force_clean_bullets(response.content)
    
    # Add header to the text (except for intro where title is usually above)
    if not is_intro:
        clean_text = f"<h2>{current_header}</h2>\n\n{clean_text}"
    else:
        # Intro doesn't need an H2 usually, or use the first header
        clean_text = f"{clean_text}"

    return {
        "draft_parts": [clean_text], # Appends to the list via operator.add
        "current_section_index": current_idx + 1
    }

def assembler_node(state: AgentState):
    """
    Combines all parts into the final draft and adds internal links.
    """
    logger.info("[Assembler] Combining draft parts...")
    full_draft = "\n\n".join(state["draft_parts"])
    
    # Inject Internal Links if missing
    coder_links = [
        ("Mobile App Development", "https://www.coderdesign.com/mobile-app-development"),
        ("Full Stack Engineering", "https://www.coderdesign.com/full-stack-engineering"),
        ("AI Services", "https://www.coderdesign.com/ai-workflow"),
        ("Contact Us", "https://www.coderdesign.com/contact"),
    ]
    
    # Simple injection strategy: finding keywords and replacing 1-2 times
    for text, url in coder_links:
        if text.lower() in full_draft.lower() and url not in full_draft:
            # Replace first occurrence only to avoid spam
            pattern = re.compile(re.escape(text), re.IGNORECASE)
            full_draft = pattern.sub(f"[{text}]({url})", full_draft, count=1)
            
    # Final cleanup of bullets just in case
    full_draft = force_clean_bullets(full_draft)
    
    return {"final_content": full_draft}

def meta_data_node(state: AgentState):
    logger.info("[Meta Data] Generating Metadata...")
    _, gpt_smart = get_llms()
    
    # We use the Intro (first part of draft) to generate metadata
    intro_text = state["draft_parts"][0] if state["draft_parts"] else state["topic"]
    
    prompt = f"""
    Generate JSON {{category, short_description, seo_title}} for this blog.
    Title must be click-worthy, under 60 chars.
    Description: 150-160 chars, optimized for CTR.
    Category: Choose best from {CATEGORIES_WEIGHTS}.
    Context: {intro_text[:800]}
    """
    response = gpt_smart.invoke([HumanMessage(content=prompt)])
    try:
        clean_json = response.content.replace("```json", "").replace("```", "")
        data = json.loads(clean_json)
        return {"final_category": data['category'], "final_short_desc": data['short_description'], "final_title": data['seo_title']}
    except:
        return {"final_category": "AI & Machine Learning", "final_short_desc": "Expert tech insights for 2025.", "final_title": state['topic'][:50]}

def visual_node(state: AgentState):
    logger.info("[Visuals] Generating concept...")
    _, gpt_smart = get_llms()
    prompt = gpt_smart.invoke([HumanMessage(content=f"Create DALL-E 3 prompt for blog titled: {state['topic']}. Abstract, geometric, tech, minimal.")])
    path = generate_relevant_image(prompt.content)
    return {"image_path": path}

# --- 4. GRAPH BUILD ---
workflow = StateGraph(AgentState)

workflow.add_node("trend_spotter", trend_spotter_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("outliner", outliner_node)
workflow.add_node("section_writer", section_writer_node)
workflow.add_node("assembler", assembler_node)
workflow.add_node("meta_data", meta_data_node)
workflow.add_node("visuals", visual_node)

# Conditional Logic to Start
def check_topic(state: AgentState):
    if not state.get("topic"): return "trend_spotter"
    return "researcher"

workflow.add_conditional_edges(START, check_topic, {"trend_spotter": "trend_spotter", "researcher": "researcher"})
workflow.add_edge("trend_spotter", "researcher")
workflow.add_edge("researcher", "outliner")
workflow.add_edge("outliner", "section_writer")

# The Loop: Keep writing sections until we are done
def should_continue_writing(state: AgentState):
    if state["current_section_index"] < len(state["outline_sections"]):
        return "section_writer"
    return "assembler"

workflow.add_conditional_edges("section_writer", should_continue_writing, {
    "section_writer": "section_writer",
    "assembler": "assembler"
})

workflow.add_edge("assembler", "meta_data")
workflow.add_edge("meta_data", "visuals")
workflow.add_edge("visuals", END)

app_graph = workflow.compile()

# --- 5. UPLOADER (STABILITY VERSION) ---
async def upload_to_coder_design(data, status_callback=None):
    logger.info("[Upload] Launching Browser...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()

        try:
            logger.info("[Upload] Logging in...")
            await page.goto(LOGIN_URL, timeout=60000)
            
            # HARDCODED PASSWORD
            await page.get_by_placeholder("Enter admin password").fill(PASSWORD_VALUE)
            await page.get_by_role("button", name="Access Admin Panel").click()
            
            # WAIT FOR REDIRECT
            logger.info("   Waiting for login to complete...")
            try:
                await page.wait_for_url("**/manage-blogs", timeout=15000)
            except:
                logger.warning("   URL didn't change, but continuing to check...")

            # NAVIGATE
            logger.info(f"[Upload] Navigating to {TARGET_URL}...")
            await page.goto(TARGET_URL, timeout=60000)
            
            # VERIFY LOGIN
            if await page.get_by_role("button", name="Access Admin Panel").is_visible():
                logger.error("   LOGIN FAILED: Redirected back to login screen.")
                raise Exception("Login Failed")

            # WAIT FOR FORM
            await page.locator('input[placeholder="Enter blog title"]').wait_for(state="attached", timeout=20000)

            logger.info("[Upload] Filling Form...")
            
            if data['image_path'] and os.path.exists(data['image_path']):
                await page.locator('input[type="file"]').set_input_files(data['image_path'])

            title_to_use = data.get('final_title', data['topic'])
            await page.get_by_placeholder("Enter blog title").fill(title_to_use)

            await page.get_by_placeholder("Enter author name").fill("Sarah Miller")
            await page.keyboard.press("Enter")

            try:
                await page.locator("select").select_option(label=data['final_category'])
            except:
                await page.locator("select").select_option(index=1)

            await page.get_by_placeholder("Enter a short description...").fill(data['final_short_desc'])

            # CONTENT INJECTION
            await page.get_by_placeholder("Enter a short description...").focus()
            await page.keyboard.press("Tab")
            
            # Use the assembled final content
            final_content = data['final_content']
            await page.keyboard.insert_text(final_content)

            logger.info("[Upload] Submitting...")
            submit_btn = page.get_by_role("button", name="Upload Blog Post")
            await submit_btn.scroll_into_view_if_needed()
            await submit_btn.click()
            
            await page.wait_for_timeout(5000)
            logger.info("[Success] Blog Uploaded!")

        except Exception as e:
            logger.error(f"[Error] Upload Failed: {e}")
            await page.screenshot(path="error_debug.png")
            raise e
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
        initial_state = {
            "topic": topic_in, 
            "custom_instructions": args.instructions, 
            "draft_parts": [], # Initialize empty list for loop
            "current_section_index": 0
        }
        
        final_state = app_graph.invoke(initial_state)
        
        # Check if content was actually generated
        if not final_state.get('final_content'):
            logger.error("No content generated!")
            sys.exit(1)
            
        disable_upload = os.getenv("DISABLE_UPLOAD", "").lower() in ("1", "true")
        if final_state.get('image_path') and not disable_upload:
            asyncio.run(upload_to_coder_design(final_state))
        elif disable_upload:
            logger.info("[Upload] Skipped.")
            
    except Exception as e:
        logger.critical(f"CRITICAL FAILURE: {e}")
        sys.exit(1)

def run_streamlit_mode():
    st.title("AI Auto-Blogger (Long-Form)")
    if st.button("Generate"):
        st.write("Running...")
        # (Streamlit implementation omitted for brevity, uses same app_graph)

if __name__ == "__main__":
    if st and get_script_run_ctx(): run_streamlit_mode()
    else: run_cli_mode()
