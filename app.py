import os
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="InterviewAnalyzer - AI-Powered Candidate Assessment",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
# Initialize variables to hold data across app reruns
if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None
if 'audio_url' not in st.session_state:
    st.session_state.audio_url = ""
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'conversation_stages' not in st.session_state:
    st.session_state.conversation_stages = {}
if 'strengths' not in st.session_state:
    st.session_state.strengths = ""
if 'improvements' not in st.session_state:
    st.session_state.improvements = ""
if 'candidate_scores' not in st.session_state:
    st.session_state.candidate_scores = {}
if 'follow_up_questions' not in st.session_state:
    st.session_state.follow_up_questions = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# --- Load API Key ---
DEFAULT_MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; margin-bottom: 1rem; width: 100%; text-align: left; }
    .app-description { margin-bottom: 20px; }
    .sub-header { font-size: 1.5rem; color: #2563EB; margin-bottom: 1rem; width: 100%; }
    .info-box { padding: 1rem; border-radius: 0.5rem; background-color: #EFF6FF; border-left: 5px solid #3B82F6; margin-bottom: 1rem; }
    .success-box { padding: 1rem; border-radius: 0.5rem; background-color: #ECFDF5; border-left: 5px solid #10B981; margin-bottom: 1rem; }
    .warning-box { padding: 1rem; border-radius: 0.5rem; background-color: #FEF3C7; border-left: 5px solid #F59E0B; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F3F4F6; border-radius: 4px 4px 0px 0px; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #DBEAFE; color: #1E3A8A; }
    .block-container { max-width: 100% !important; padding: 1rem 1rem !important; }
</style>
""", unsafe_allow_html=True)

# --- Core API Functions ---

def transcribe_audio(audio_url=None, audio_file=None):
    """Transcribe audio using Mistral API with timestamps."""
    try:
        api_key = st.session_state.api_key if st.session_state.api_key else DEFAULT_MISTRAL_API_KEY
        if not api_key:
            return None, "API key is required. Please provide a Mistral API key in the API Configuration section."

        url = "https://api.mistral.ai/v1/audio/transcriptions"
        headers = {"x-api-key": api_key}
        data = {
            'model': "voxtral-mini-2507",
            'timestamp_granularities': "segment"
        }

        if audio_url:
            data['file_url'] = audio_url
            response = requests.post(url, headers=headers, data=data)
        elif audio_file:
            files = {'file': audio_file, 'model': (None, data['model']), 'timestamp_granularities': (None, data['timestamp_granularities'])}
            response = requests.post(url, headers=headers, files=files)
        else:
            return None, "No audio provided"

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Exception: {str(e)}"

def chat_with_transcript(prompt=""):
    """Analyze a transcript using Mistral's chat API."""
    try:
        api_key = st.session_state.api_key if st.session_state.api_key else DEFAULT_MISTRAL_API_KEY
        if not api_key:
            return None, "API key is required."

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": "voxtral-mini-2507", "messages": [{"role": "user", "content": prompt}]}

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Exception: {str(e)}"

# --- Analysis & Logic Functions ---

def segment_conversation(segments):
    """Segment the interview into logical stages."""
    stages = {
        "introductions": [], "behavioral_questions": [], "technical_questions": [],
        "candidate_questions": [], "wrap_up": []
    }
    total_segments = len(segments)
    if total_segments > 0:
        intro_end = max(1, int(total_segments * 0.15))
        behavioral_end = max(2, int(total_segments * 0.45))
        technical_end = max(3, int(total_segments * 0.70))
        candidate_q_end = max(4, int(total_segments * 0.90))

        stages["introductions"] = segments[:intro_end]
        stages["behavioral_questions"] = segments[intro_end:behavioral_end]
        stages["technical_questions"] = segments[behavioral_end:technical_end]
        stages["candidate_questions"] = segments[technical_end:candidate_q_end]
        stages["wrap_up"] = segments[candidate_q_end:]
    return stages

def calculate_talk_ratio(segments):
    """Calculate interviewer vs. candidate talk ratio."""
    interviewer_duration, candidate_duration = 0, 0
    # Simple assumption: interviewer starts (even index), candidate responds (odd index)
    for i, segment in enumerate(segments):
        duration = segment.get("end", 0) - segment.get("start", 0)
        if i % 2 == 0:
            interviewer_duration += duration
        else:
            candidate_duration += duration
    return interviewer_duration / candidate_duration if candidate_duration > 0 else interviewer_duration

def extract_call_metrics(transcript, segments):
    """Extract basic metrics from the interview."""
    word_count = len(transcript.split())
    duration = segments[-1]["end"] if segments else 0
    talk_ratio = calculate_talk_ratio(segments)
    filler_words = ["um", "uh", "like", "you know", "actually", "basically"]
    filler_count = sum(transcript.lower().count(word) for word in filler_words)
    return {
        "word_count": word_count, "duration": duration, "talk_ratio": talk_ratio,
        "filler_words": filler_count, "filler_frequency": filler_count / (word_count if word_count > 0 else 1)
    }

def analyze_interview(transcript):
    """Generate comprehensive interview analysis using the chat API."""
    # Define interview-specific prompts
    strengths_prompt = f"Analyze this interview transcript and identify the candidate's top 3 strengths, citing evidence from their answers. Format as bullet points.\n\nTranscript:\n{transcript}"
    improvements_prompt = f"Identify 3 areas where the candidate could improve their responses or communication style. Be constructive and specific.\n\nTranscript:\n{transcript}"
    scoring_prompt = f"Evaluate the candidate on Communication, Technical Skill, and STAR Method Usage for behavioral questions. Give a score out of 10 for each and a brief explanation.\n\nTranscript:\n{transcript}"
    follow_up_prompt = f"Based on the transcript, suggest 3 insightful follow-up questions for the next interview stage.\n\nTranscript:\n{transcript}"

    # Get analysis from API
    strengths, _ = chat_with_transcript(prompt=strengths_prompt)
    improvements, _ = chat_with_transcript(prompt=improvements_prompt)
    scoring_text, _ = chat_with_transcript(prompt=scoring_prompt)
    follow_up, _ = chat_with_transcript(prompt=follow_up_prompt)

    # Parse scores from the text response
    scores = {}
    try:
        if "communication:" in scoring_text.lower():
            scores["communication"] = int(scoring_text.lower().split("communication:")[1].split("/")[0].strip())
        if "technical skill:" in scoring_text.lower():
            scores["technical_skill"] = int(scoring_text.lower().split("technical skill:")[1].split("/")[0].strip())
        if "star method usage:" in scoring_text.lower():
            scores["star_method_usage"] = int(scoring_text.lower().split("star method usage:")[1].split("/")[0].strip())
    except Exception:
        # Provide default scores if parsing fails
        scores = {"communication": 0, "technical_skill": 0, "star_method_usage": 0}

    return {
        "strengths": strengths or "Could not analyze strengths.",
        "improvements": improvements or "Could not analyze areas for improvement.",
        "scores": scores,
        "follow_up": follow_up or "Could not generate follow-up questions."
    }

# --- UI Rendering Functions ---

def render_header():
    """Render the app header and API key input."""
    st.markdown("<h1 class='main-header'>InterviewAnalyzer 🧑‍💼</h1>", unsafe_allow_html=True)
    st.markdown("<p><strong>AI-Powered Candidate Assessment</strong></p>", unsafe_allow_html=True)

    with st.expander("About InterviewAnalyzer"):
        st.markdown("""<div class='app-description'>
            <p><strong>InterviewAnalyzer</strong> uses Mistral's Voxtral model to analyze recorded interviews. It helps hiring managers and recruiters by:</p>
            <ul>
                <li>Transcribing and analyzing interview recordings</li>
                <li>Identifying candidate strengths and areas for improvement</li>
                <li>Providing objective scores on key competencies</li>
                <li>Generating insightful follow-up questions for the next stage</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with st.expander("API Configuration"):
        st.markdown("<p>Enter your <a href='https://console.mistral.ai/' target='_blank'>Mistral AI API key</a> to use the app.</p>", unsafe_allow_html=True)
        api_key = st.text_input("Mistral API Key", type="password", value=st.session_state.api_key)

        if api_key:
            st.session_state.api_key = api_key
            st.success("API key saved! You can now analyze interviews.")
        elif DEFAULT_MISTRAL_API_KEY:
            st.info("Using default API key from environment variables.")
        else:
            st.warning("Please enter a Mistral API key to use the app.")

def render_audio_upload():
    """Render the audio upload section and analysis button."""
    st.markdown("<h2 class='sub-header'>Upload Interview Recording</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='info-box'><strong>Upload Audio File (.mp3, .wav)</strong></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"], label_visibility="collapsed")
        if uploaded_file:
            st.session_state.uploaded_audio = uploaded_file
            st.audio(uploaded_file)
    with col2:
        st.markdown("<div class='info-box'><strong>Or Provide Audio URL</strong></div>", unsafe_allow_html=True)
        audio_url = st.text_input("Enter URL to MP3 audio", placeholder="https://example.com/interview.mp3", label_visibility="collapsed")
        if audio_url:
            st.session_state.audio_url = audio_url
            st.audio(audio_url)

    _, mid_col, _ = st.columns([1, 1, 1])
    with mid_col:
        if st.button("Analyze Interview", use_container_width=True, type="primary"):
            if st.session_state.uploaded_audio or st.session_state.audio_url:
                st.session_state.processing = True
                process_interview()
                st.session_state.processing = False
            else:
                st.error("Please upload an audio file or provide a URL.")

def process_interview():
    """Process the audio: transcribe, analyze, and store results."""
    with st.spinner("Step 1/2: Transcribing audio... This may take a moment."):
        audio_input = st.session_state.uploaded_audio or st.session_state.audio_url
        result, error = transcribe_audio(audio_file=st.session_state.uploaded_audio, audio_url=st.session_state.audio_url)

        if error:
            st.error(f"Transcription Error: {error}")
            return

        st.session_state.transcript = result.get("text", "")
        st.session_state.segments = result.get("segments", [])
        st.session_state.conversation_stages = segment_conversation(st.session_state.segments)

    with st.spinner("Step 2/2: Analyzing interview content..."):
        analysis_results = analyze_interview(st.session_state.transcript)
        st.session_state.strengths = analysis_results["strengths"]
        st.session_state.improvements = analysis_results["improvements"]
        st.session_state.candidate_scores = analysis_results["scores"]
        st.session_state.follow_up_questions = analysis_results["follow_up"]

    st.rerun()

def render_results_tabs():
    """Render the main results container with multiple tabs."""
    if not st.session_state.transcript:
        return
    st.markdown("<h2 class='sub-header'>Interview Analysis Results</h2>", unsafe_allow_html=True)
    tab_list = ["Transcript", "Overview", "Candidate Insights", "Performance Scorecard", "Follow-up Questions"]
    tabs = st.tabs(tab_list)

    with tabs[0]: render_transcript_tab()
    with tabs[1]: render_overview_tab()
    with tabs[2]: render_insights_tab()
    with tabs[3]: render_scorecard_tab()
    with tabs[4]: render_follow_up_tab()

def render_transcript_tab():
    """Render the transcript and conversation stages."""
    st.markdown("### Full Interview Transcript")
    if st.session_state.segments:
        df = pd.DataFrame(st.session_state.segments)[['start', 'end', 'text']]
        df['start'] = df['start'].apply(lambda s: f"{s:.1f}s")
        df['end'] = df['end'].apply(lambda s: f"{s:.1f}s")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### Conversation Stages")
        cols = st.columns(len(st.session_state.conversation_stages))
        for i, (stage, segments) in enumerate(st.session_state.conversation_stages.items()):
            with cols[i]:
                st.markdown(f"**{stage.replace('_', ' ').title()}**")
                stage_text = " ".join([s["text"] for s in segments])
                st.markdown(f"<div style='height:150px;overflow-y:auto;font-size:0.9em;padding:5px;background-color:#F3F4F6;border-radius:5px;'>{stage_text}</div>", unsafe_allow_html=True)
    else:
        st.warning("No transcript segments available.")

def render_overview_tab():
    """Render high-level metrics and timeline visualization."""
    st.markdown("### Interview Overview")
    if st.session_state.transcript and st.session_state.segments:
        metrics = extract_call_metrics(st.session_state.transcript, st.session_state.segments)
        cols = st.columns(4)
        cols[0].metric("Interview Duration", f"{metrics['duration']:.1f}s")
        cols[1].metric("Word Count", metrics["word_count"])
        cols[2].metric("Talk Ratio (Interviewer:Candidate)", f"{metrics['talk_ratio']:.1f}:1")
        cols[3].metric("Filler Words Count", metrics["filler_words"])

        st.markdown("### Conversation Timeline")
        timeline_data = []
        for i, segment in enumerate(st.session_state.segments):
            speaker = "Interviewer" if i % 2 == 0 else "Candidate"
            timeline_data.append({"Start": segment["start"], "End": segment["end"], "Speaker": speaker, "Text": segment["text"]})
        fig = px.timeline(timeline_data, x_start="Start", x_end="End", y="Speaker", color="Speaker", hover_data=["Text"], title="Interview Timeline by Speaker")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No call data available.")

def render_insights_tab():
    """Render AI-generated strengths and areas for improvement."""
    st.markdown("### Candidate Strengths")
    st.markdown(f"<div class='success-box'>{st.session_state.strengths}</div>", unsafe_allow_html=True)
    st.markdown("### Areas for Improvement")
    st.markdown(f"<div class='warning-box'>{st.session_state.improvements}</div>", unsafe_allow_html=True)

def render_scorecard_tab():
    """Render the candidate performance scorecard with a radar chart."""
    st.markdown("### Candidate Performance Scorecard")
    if st.session_state.candidate_scores:
        scores = st.session_state.candidate_scores
        categories = [k.replace('_', ' ').title() for k in scores.keys()]
        values = list(scores.values())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=categories + categories[:1],
            fill='toself', name='Candidate Score',
            line_color='#3B82F6', fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(len(scores))
        for i, (metric, score) in enumerate(scores.items()):
            with cols[i]:
                st.metric(metric.replace('_', ' ').title(), f"{score}/10")
    else:
        st.warning("No performance scores available.")

def render_follow_up_tab():
    """Render the AI-generated follow-up questions."""
    st.markdown("### Suggested Follow-up Questions")
    if st.session_state.follow_up_questions:
        st.markdown(f"<div class='info-box'>{st.session_state.follow_up_questions}</div>", unsafe_allow_html=True)
    else:
        st.info("No follow-up questions were generated.")

def render_footer():
    """Render a simple footer."""
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <p>Created by <strong>Toni Ramchandani</strong></p>
        </div>
    """, unsafe_allow_html=True)

# --- Main App Execution ---
def main():
    """Main function to run the Streamlit app."""
    render_header()
    render_audio_upload()
    st.markdown("---")
    render_results_tabs()
    render_footer()

if __name__ == "__main__":
    main()
