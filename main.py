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

# Handling the library rename warning
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

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
def perform_web_search(query: str):
    print(f"[Tool] Searching Web: {query}")
    try:
        results = DDGS().text(keywords=query, max_results=5)
        return "\n\n".join([f"SOURCE TITLE: {r['title']}\nLINK: {r['href']}\nINFO: {r['body']}" for r in results])
    except Exception as e:
        print(f"[Error] Search failed: {e}")
        return ""


def generate_relevant_image(scene_description: str):
    from openai import OpenAI
    client = OpenAI()

    style_instruction = "Style: High-End Tech Editorial. 3D Render style or Detailed Vector Art. Focus on the SUBJECT MATTER. NO TEXT."
    final_prompt = f"{scene_description}. {style_instruction}"

    # --- FIX: SAFETY TRUNCATION ---
    # DALL-E 3 limit is 4000 chars. We cut it to 3900 to be safe.
    if len(final_prompt) > 3900:
        print(f"[Tool] Truncating image prompt (Length: {len(final_prompt)} -> 3900)")
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
def get_llms():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing.")
    return (
        ChatOpenAI(model="gpt-4-turbo", temperature=0.2),
        ChatOpenAI(model="gpt-4o", temperature=0.7)
    )


def trend_spotter_node(state: AgentState):
    print("[Trend Spotter] User did not provide a topic. Rolling dice for category...")
    gpt4_turbo, gpt4o = get_llms()

    categories, weights = zip(*CATEGORIES_WEIGHTS)
    target_category = random.choices(categories, weights=weights, k=1)[0]
    print(f"[Trend Spotter] Selected Category: {target_category}")

    search_query = f"trending {target_category} news controversy 2025"
    search_results = perform_web_search(search_query)

    system_prompt = f"""You are an Editor-in-Chief.
    Your goal: Pick the most engaging news story about {target_category}.
    Criteria:
    1. Must be strictly related to {target_category}.
    2. Must be specific.
    3. Output ONLY the topic title.
    """

    response = gpt4o.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the latest news:\n{search_results}")
    ])

    new_topic = response.content.strip().replace('"', '')
    print(f"[Trend Spotter] Decided on Topic: {new_topic}")

    return {"topic": new_topic, "final_category": target_category}


def researcher_node(state: AgentState):
    print(f"[Researcher] Investigating: {state['topic']}")
    
    # Multiple searches for diverse sources
    queries = [
        f"{state['topic']} technical analysis implementation",
        f"{state['topic']} GitHub repository examples",
        f"{state['topic']} official documentation",
        f"{state['topic']} business impact ROI case study",
        f"{state['topic']} best practices industry standards"
    ]
    
    all_data = []
    for query in queries:
        data = perform_web_search(query)
        if data:
            all_data.append(data)
    
    combined_data = "\n\n--- NEW SEARCH ---\n\n".join(all_data)
    return {"research_data": combined_data, "iteration_count": 0}


def writer_node(state: AgentState):
    print("[Writer] Drafting content...")
    gpt4_turbo, gpt4o = get_llms()

    topic = state["topic"]
    instructions = state["custom_instructions"]
    feedback = state.get("critique_feedback", None)

    system = """You are a Senior Tech Columnist for Coder Design.

    CORE OBJECTIVE: Write an engaging deep-dive (1200-1500 words) that combines TECHNICAL depth with BUSINESS insights.

    CONTENT STRATEGY:
    - Balance technical implementation details (40%) with business impact and strategy (30%)
    - Include real-world examples, case studies, and industry applications
    - Discuss ROI, cost implications, market trends, and competitive advantages
    - Cover both developer perspective AND business stakeholder viewpoint
    - Vary your writing style: mix tutorials, analysis, opinion, and storytelling
    - Use different angles: sometimes controversial, sometimes educational, sometimes forward-looking

    CRITICAL FORMATTING RULES (STRICT):
    1. **USE BULLET POINTS** for lists.
    2. **STRICT BULLET RULE**: Do NOT use bold text (`**text**`) inside bullet points. Bullets must be plain text.
    3. **USE H3 HEADERS FREQUENTLY**: Every 150-200 words, create a new `### Subtopic Header`.
    4. **SHORT PARAGRAPHS**: Maximum 3 sentences per paragraph.
    5. **Bold** important concepts in normal paragraphs (not bullets).

    MANDATORY EXTERNAL LINKS (CRITICAL):
    - Include exactly 2-3 high-quality, RELEVANT external links
    - MUST include links from: Official documentation, GitHub repos, authoritative tech blogs (Google, Microsoft, AWS, etc.), or reputable tech publications
    - Links MUST be directly relevant to the topic - no tangential references
    - Format: `[Link Text](URL)`
    - Extract URLs from the RESEARCH DATA provided
    - CRITICAL: Only reference concepts/technologies that are DIRECTLY related to the main topic. Avoid forced or unrelated comparisons.

    MANDATORY CODER DESIGN LINKS:
    Naturally weave in these services (use 2-3 per article):
    - Mobile Development: https://www.coderdesign.com/mobile-app-development
    - Full Stack: https://www.coderdesign.com/full-stack-engineering
    - AI Workflow: https://www.coderdesign.com/ai-workflow
    - SEO Services: https://www.coderdesign.com/seo-management
    - Contact Us: https://www.coderdesign.com/contact
    """

    prompt = f"""
    TOPIC: {topic}
    CUSTOM INSTRUCTIONS: {instructions}

    RESEARCH DATA (CONTAINS DIVERSE URLs - USE THEM ALL):
    {state['research_data']}
    
    IMPORTANT: Make this piece unique by varying:
    - Tone: Choose from authoritative, conversational, analytical, or provocative
    - Structure: Mix how-to guides, opinion pieces, trend analysis, or comparison articles
    - Technical vs Business balance: Adjust based on topic relevance
    - Include practical code examples, architecture diagrams (described), or business metrics where relevant
    """

    if feedback and feedback != "APPROVED":
        prompt += f"\n\nFIX PREVIOUS ERRORS: {feedback}"

    response = gpt4o.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return {"content_draft": response.content, "iteration_count": state["iteration_count"] + 1}


def seo_analyst_node(state: AgentState):
    print("[SEO Analyst] Auditing draft...")
    gpt4_turbo, _ = get_llms()
    draft = state["content_draft"]
    word_count = len(draft.split())

    if word_count < 1100:
        return {"critique_feedback": f"Draft is too short ({word_count} words). EXPAND to 1200-1500 words."}

    links_found = re.findall(r'\[.*?\]\(http.*?\)', draft)
    if len(links_found) < 2:
        return {"critique_feedback": f"CRITICAL: Only {len(links_found)} links found. Add 2-3 highly relevant external hyperlinks from authoritative sources (official docs, GitHub, tech company blogs)."}

    audit = gpt4_turbo.invoke([
        SystemMessage(
            content="Audit for: 1. **Bullet Points must NOT contain bold text.** (Reject if you see '**' inside a bullet). 2. Frequent H3 Headers. 3. Internal Coder Design Links. 4. 2-3 RELEVANT external links from authoritative sources. 5. Balance of technical AND business insights. 6. CRITICAL: All examples, comparisons, and references must be DIRECTLY relevant to the main topic. Reject if you see forced/unrelated comparisons or tangential references that don't make logical sense. If Good, say 'APPROVED'."),
        HumanMessage(content=draft)
    ])

    if "APPROVED" in audit.content.upper():
        print("[SEO Analyst] Approved.")
        return {"critique_feedback": "APPROVED"}

    print(f"[SEO Analyst] Rejection: {audit.content[:100]}...")
    return {"critique_feedback": audit.content}


def meta_data_node(state: AgentState):
    print("[Meta Data] Generating Metadata & Title...")
    gpt4_turbo, gpt4o = get_llms()
    draft = state["content_draft"]

    prompt = f"""
    Based on this blog post, output a JSON object with:
    1. "category": Choose one of [AI & Machine Learning, Full-Stack Development, Mobile App Development, AI SEO & AEO Services]
    2. "short_description": A 2-sentence hook for the homepage.
    3. "seo_title": A punchy, click-worthy title STRICTLY UNDER 50 CHARACTERS.

    Blog start: {draft[:1000]}
    """
    response = gpt4o.invoke([HumanMessage(content=prompt)])

    try:
        clean_json = response.content.replace("```json", "").replace("```", "")
        data = json.loads(clean_json)

        final_cat = state.get('final_category') if state.get('final_category') else data['category']

        return {
            "final_category": final_cat,
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

    image_prompt_system = """You are an expert Art Director creating images for tech articles.
    Your goal is to create a HIGHLY SPECIFIC prompt for DALL-E 3.

    CRITICAL RULES: 
    1. The image MUST visually represent the EXACT subject mentioned in the topic
    2. If it's about a company (Google, Microsoft, etc.), include their recognizable visual identity (colors, logo style, iconic imagery)
    3. If it's about a specific technology, show that exact technology's visual metaphor:
       - Kubernetes -> Shipping containers with helm wheels
       - Python -> Snake imagery with code patterns
       - React -> Atomic structure, components
       - Google -> Colorful G colors (blue, red, yellow, green)
       - AWS -> Orange/cloud imagery
    4. BE SPECIFIC, not generic. "AI automation" is too vague. "Google's AI automation tools" should show Google's signature colors.
    5. Absolutely NO text, letters, numbers, or typography anywhere in the image
    6. High-end 3D render or detailed vector art style
    7. KEEP DESCRIPTION CONCISE (Under 60 words)
    """

    prompt_request = f"""
    ARTICLE TOPIC: {state['topic']}
    FINAL TITLE: {state.get('final_title', state['topic'])}
    CONTENT PREVIEW: {state['content_draft'][:600]}

    Create a SPECIFIC image prompt that directly represents the main subject (company, technology, or concept) mentioned in the title.
    If it's about a specific company or product, reference their visual identity.
    Be concrete and specific, not abstract or generic:
    """

    image_prompt_generator = gpt4o.invoke([
        SystemMessage(content=image_prompt_system),
        HumanMessage(content=prompt_request)
    ])

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
    if not state.get("topic") or state.get("topic") == "":
        return "trend_spotter"
    return "researcher"


workflow.add_conditional_edges(START, check_topic, {
    "trend_spotter": "trend_spotter",
    "researcher": "researcher"
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
    msg = "[Upload] Launching Browser..."
    if status_callback:
        status_callback(msg)
    else:
        print(msg)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            if status_callback: status_callback("[Upload] Logging in...")
            await page.goto(LOGIN_URL)
            await page.get_by_placeholder("Enter admin password").fill(PASSWORD)
            await page.get_by_role("button", name="Access Admin Panel").click()
            await page.wait_for_load_state("networkidle")

            if "upload-blog" not in page.url:
                await page.goto(TARGET_URL)

            if status_callback: status_callback("[Upload] Filling Form...")

            if data['image_path'] and os.path.exists(data['image_path']):
                await page.locator('input[type="file"]').set_input_files(data['image_path'])

            title_to_use = data.get('final_title', data['topic'])
            print(f"   Title Used: {title_to_use} (Length: {len(title_to_use)})")
            await page.get_by_placeholder("Enter blog title").fill(title_to_use)

            authors = [
                "Emily Davis", "James Wilson", "Sarah Miller",
                "Michael Brown", "David Clark", "Jennifer Wu", "Robert Martinez"
            ]
            await page.get_by_placeholder("Enter author name").fill(random.choice(authors))

            await page.keyboard.press("Enter")
            await page.wait_for_timeout(500)

            print(f"   Category: {data['final_category']}")
            await page.locator("select").select_option(label=data['final_category'])

            await page.get_by_placeholder("Enter a short description...").fill(data['final_short_desc'])

            await page.get_by_placeholder("Enter a short description...").focus()
            await page.keyboard.press("Tab")
            await page.keyboard.insert_text(data['content_draft'])

            if status_callback: status_callback("[Upload] Submitting...")
            await page.get_by_role("button", name="Upload Blog Post").click()
            await page.wait_for_timeout(5000)

            success_msg = "[Success] Blog Uploaded!"
            if status_callback:
                status_callback(success_msg)
            else:
                print(success_msg)

        except Exception as e:
            fail_msg = f"[Error] Upload Failed: {e}"
            if status_callback:
                status_callback(fail_msg)
            else:
                print(fail_msg)
            await page.screenshot(path="error_debug.png")

        finally:
            await browser.close()
            if data['image_path'] and os.path.exists(data['image_path']):
                os.remove(data['image_path'])


# --- 6. EXECUTION ---
def run_cli_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="", help="Topic.")
    parser.add_argument("--instructions", type=str, default="", help="Instructions.")
    args = parser.parse_args()

    topic_in = args.topic.strip()

    if not topic_in:
        print("--- LAUNCHING TREND SPOTTER ---")
    else:
        print(f"--- TOPIC: {topic_in} ---")

    initial_state = {
        "topic": topic_in,
        "custom_instructions": args.instructions,
        "iteration_count": 0
    }

    final_state = app_graph.invoke(initial_state)

    if final_state.get('image_path'):
        asyncio.run(upload_to_coder_design(final_state))


def run_streamlit_mode():
    st.title("AI Auto-Blogger")
    st.markdown("Generates engaging blogs (1200+ words).")

    with st.sidebar:
        st.header("Configuration")
        st.info("Weights: AI (30%), Web (25%), SEO (20%), Mobile (25%)")
        topic_input = st.text_input("Enter Blog Topic (Optional)", value="")
        custom_instructions = st.text_area("Custom Instructions", value="Make it professional.", height=150)
        start_btn = st.button("Generate & Upload", type="primary")

    if start_btn:
        topic_clean = topic_input.strip()

        if not topic_clean:
            st.info("🔎 Trend Spotter activated.")
        else:
            st.info(f"🎯 User Topic: **{topic_clean}**")

        with st.status("AI Agent Working...", expanded=True) as status:
            initial_state = {
                "topic": topic_clean,
                "custom_instructions": custom_instructions,
                "iteration_count": 0
            }

            status.write("Initializing Workflow...")
            final_state = app_graph.invoke(initial_state)

            if not topic_clean:
                st.success(f"📈 Found Trending Topic: **{final_state['topic']}** ({final_state.get('final_category')})")

            if final_state.get('content_draft'):
                status.write("Draft Created.")
                with st.expander("Preview Draft"):
                    st.markdown(final_state['content_draft'])

            if final_state.get('image_path'):
                status.write("Image Generated.")
                st.image(final_state['image_path'], caption="Generated Header", width=300)

            def update_status(msg):
                status.write(msg)

            asyncio.run(upload_to_coder_design(final_state, update_status))
            status.update(label="Process Complete!", state="complete")


if __name__ == "__main__":
    is_streamlit = False
    try:
        if st is not None and get_script_run_ctx() is not None:
            is_streamlit = True
    except:
        pass

    if is_streamlit:
        run_streamlit_mode()
    else:
        run_cli_mode()