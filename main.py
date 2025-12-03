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

    search_query = f"trending {target_category} news controversy debate 2025"
    search_results = perform_web_search(search_query)

    system_prompt = f"""You are an Editor-in-Chief looking for ENGAGING story angles.
    Your goal: Pick the most compelling, conversation-worthy news story about {target_category}.
    
    Criteria:
    1. Must be strictly related to {target_category}
    2. Must be INTERESTING - something people will want to read and share
    3. Prefer stories with:
       - Controversy or debate
       - Surprising developments or data
       - Major company announcements
       - Paradigm shifts or industry changes
       - Real-world impact
    4. Avoid generic tutorials or basic explanations
    5. Output ONLY the topic title - make it compelling
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
    
    # Targeted searches for authoritative sources (NO GitHub or Stack Overflow)
    queries = [
        f"{state['topic']} official documentation",
        f"{state['topic']} real-world case studies companies",
        f"{state['topic']} statistics data industry report",
        f"{state['topic']} expert analysis tech blog",
        f"{state['topic']} Google AWS Microsoft blog"
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

    system = """You are a world-class SEO content writer specializing in generating content that is indistinguishable from human authorship. Your expertise lies in capturing emotional nuance, cultural relevance, and contextual authenticity, ensuring content that resonates naturally with any audience focused on tech, full-stack development, SEO, and machine learning.

    CORE OBJECTIVE: Write a 1200-1500 word article that feels genuinely human-written, not AI-generated.

    WRITING STYLE - AUTHENTIC HUMAN VOICE:
    - Conversational and engaging, like talking to a smart colleague
    - Use contractions naturally (don't, won't, here's, that's)
    - Include casual phrases: "You know what?", "Honestly", "Here's the thing"
    - Rhetorical questions to engage: "But why does this matter?"
    - Mix professional jargon with casual explanations
    - Add emotional cues and relatable moments
    - Use idioms and colloquialisms naturally
    - Mild contradictions you later explain ("At first glance X, but actually Y")
    - Natural digressions that connect back to the main point
    - Transitional phrases: "Let me explain", "Here's what I mean"

    FLESCH READING EASE: Target around 80
    - Mix short punchy sentences with longer complex ones
    - Keep it readable but not dumbed down
    - Use dependency grammar for easy comprehension

    VOCABULARY & STYLE:
    - Diverse vocabulary with unexpected word choices
    - Industry-specific metaphors and analogies from everyday life
    - Reference real tools, brands, companies when relevant
    - Include mild repetition for emphasis (humans do this naturally)
    - Mix active and passive voice (lean towards active)
    - Varied punctuation (dashes, semicolons, parentheses)
    - Mix formal and casual language naturally

    FORBIDDEN WORDS/PHRASES (AVOID THESE - THEY SCREAM AI):
    Words: opt, dive, unlock, unleash, intricate, utilization, transformative, alignment, proactive, scalable, benchmark
    Phrases: "In this world", "in today's world", "at the end of the day", "on the same page", "end-to-end", "in order to", "best practices", "dive into"

    STRUCTURAL ELEMENTS:
    - Vary paragraph lengths (1 to 7 sentences) 
    - Short paragraphs for impact, longer for explanation
    - Use bulleted lists SPARINGLY and naturally (only when truly needed)
    - Conversational subheadings that sound human-written
    - Dynamic rhythm across paragraphs
    - High perplexity (varied structures) and burstiness (mix of short/long sentences)

    CRITICAL FORMATTING RULES:
    1. **BULLET POINTS**: ABSOLUTELY NO BOLD TEXT inside bullets. No `**text**` in any bullet point. Plain text only.
    2. **NO EXTRA SPACING** in bullet points - keep them tight
    3. **H3 HEADERS**: Use them every 150-200 words, make them conversational
    4. **SHORT PARAGRAPHS**: Maximum 3-4 sentences usually
    5. **Bold** important concepts in regular paragraphs (NOT in bullets)

    EXTERNAL LINKS (2-3 ONLY):
    - NO GitHub links
    - NO Stack Overflow links  
    - Use: Official documentation, reputable tech publications, major tech company blogs (Google AI Blog, AWS Blog, Microsoft DevBlogs)
    - Links must be directly relevant and add real value
    - Format: `[Link Text](URL)`
    - Extract from RESEARCH DATA provided

    INTERNAL LINKS (Weave in 2-3 naturally):
    - Mobile Development: https://www.coderdesign.com/mobile-app-development
    - Full Stack: https://www.coderdesign.com/full-stack-engineering
    - AI Workflow: https://www.coderdesign.com/ai-workflow
    - SEO Services: https://www.coderdesign.com/seo-management
    - Contact Us: https://www.coderdesign.com/contact

    CONTENT STRATEGY:
    - Start with a HOOK: personal anecdote, surprising stat, or "You know what?"
    - Balance technical depth (40%) with business impact (30%)
    - Real-world examples with specific companies and numbers
    - Discuss both wins AND failures (humans admit mistakes)
    - Include actionable takeaways
    - End with thought-provoking conclusion or relatable call-to-action
    - Seasonal elements or current trends when relevant
    - Cultural references that resonate

    HUMAN AUTHENTICITY CHECKLIST:
    - Does it sound like a person wrote this over coffee?
    - Would you share this with a colleague?
    - Are there natural imperfections and casual moments?
    - Does it avoid robotic, overly-polished language?
    - Is there personality and opinion showing through?
    """

    prompt = f"""
    TOPIC: {topic}
    TARGET AUDIENCE: Tech professionals, developers, business stakeholders interested in full-stack, SEO, AI/ML
    WORD COUNT: 1200-1500 words
    CUSTOM INSTRUCTIONS: {instructions}

    RESEARCH DATA (Use for context and finding authoritative external links - NO GitHub or Stack Overflow):
    {state['research_data']}
    
    CRITICAL REQUIREMENTS:
    1. Write like a human expert, NOT an AI
    2. Start with engaging hook (anecdote, "You know what?", surprising fact)
    3. Use conversational transitions and casual phrases naturally
    4. Include 2-3 external links ONLY from official docs or major tech blogs
    5. NO GitHub, NO Stack Overflow links
    6. Mix sentence lengths dramatically for burstiness
    7. Add subtle emotional cues and rhetorical questions
    8. Include real company names, specific numbers, concrete examples
    9. ABSOLUTELY NO bold text (`**text**`) inside bullet points
    10. Avoid all forbidden AI-sounding words and phrases listed above
    11. End with relatable, actionable conclusion
    
    Write as if you're explaining this to a friend who's genuinely curious. Be opinionated. Share insights. Make it memorable.
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
        return {"critique_feedback": f"CRITICAL: Only {len(links_found)} links found. Add 2-3 external links from official docs or major tech company blogs (NO GitHub, NO Stack Overflow)."}
    
    # Check for forbidden sources
    forbidden_sources = ['github.com', 'stackoverflow.com', 'stackexchange.com']
    for link in links_found:
        link_lower = link.lower()
        if any(forbidden in link_lower for forbidden in forbidden_sources):
            return {"critique_feedback": f"CRITICAL: Found forbidden link source (GitHub/Stack Overflow). Replace with official documentation or tech company blogs."}

    audit = gpt4_turbo.invoke([
        SystemMessage(
            content="""Audit this content with STRICT criteria:
            1. **CRITICAL**: Bullet points must have ZERO bold text. If you see `**text**` inside ANY bullet point, REJECT immediately.
            2. **CRITICAL**: Check for AI-sounding words (opt, dive, unlock, unleash, intricate, utilization, transformative, alignment, proactive, scalable, benchmark). If found, REJECT.
            3. **CRITICAL**: Check for AI phrases ("in this world", "in today's world", "at the end of the day", "best practices", "dive into"). If found, REJECT.
            4. **CRITICAL**: NO GitHub or Stack Overflow links allowed. If found, REJECT.
            5. Conversational H3 headers every 150-200 words
            6. 2-3 external links from official docs or major tech blogs (Google, AWS, Microsoft blogs)
            7. 2-3 internal Coder Design links
            8. Human-like tone: conversational, uses contractions, rhetorical questions, casual phrases
            9. Mix of short and long sentences (burstiness)
            10. Starts with engaging hook, ends with actionable conclusion
            11. Real examples with specific companies/numbers
            12. Avoids overly polished, robotic language
            
            If ALL criteria pass, say 'APPROVED'. Otherwise, list specific issues to fix."""),
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