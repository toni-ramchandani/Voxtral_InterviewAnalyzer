Here’s your complete `README.md` file for the **InterviewAnalyzer** project, written for a pure Python + Streamlit setup:

---

````markdown
# 🧑‍💼 InterviewAnalyzer – AI-Powered Candidate Assessment Dashboard

> Turn raw audio into a live hiring dashboard powered by AI, in minutes.

InterviewAnalyzer takes a recorded interview (MP3/WAV or a URL), transcribes it using **Mistral's Voxtral model**, segments it into logical conversation stages, analyzes the candidate’s responses, and visualizes the results in a clean, multi-tab Streamlit app.

---

## 🎯 Why use InterviewAnalyzer?

- 🔊 Upload audio or link to a remote file  
- 🧠 AI-generated transcript + segment timing  
- 📊 Dashboard with interview metrics and talk ratio  
- 💡 Candidate strengths, improvements, and follow-up questions  
- 📈 Radar scorecard (Communication, Technical Skill, STAR Method)  
- ⚡ Fully local, fast, and secure — just needs a Mistral API key

---


## 🛠 Setup

1. **Clone the repo**

```bash
git clone https://github.com/toni-ramchandani/Voxtral_InterviewAnalyzer.git
cd interview-analyzer
````

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set your Mistral API Key**

You can either:

* Add a `.env` file with:

```
MISTRAL_API_KEY=your_key_here
```

or enter it via the Streamlit UI at runtime.

4. **Run the app**

```bash
streamlit run app.py
```

---

## 🎧 Audio Input

The app supports:

* `.mp3` or `.wav` file uploads
* Direct URL links to audio files

---

## 📦 Features

| Feature                      | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| 🎙 Transcription             | Uses Mistral’s `voxtral-mini-2507` with segment timestamps |
| 🧩 Conversation Segmentation | Divides the transcript into logical stages                 |
| 📈 Metrics                   | Duration, word count, filler frequency, talk ratio         |
| 💡 AI Insights               | Candidate strengths and improvements                       |
| 🧠 Scoring                   | AI-generated ratings on 3 key dimensions                   |
| 🔁 Follow-ups                | Smart next-round questions                                 |
| 📊 Streamlit UI              | Fully interactive multi-tab dashboard                      |

---

## 🤖 Powered By

* [Mistral AI](https://console.mistral.ai/) – for transcription and chat completions
* [Streamlit](https://streamlit.io/) – for the frontend
* [Plotly](https://plotly.com/) – for the timeline and radar chart
* [Python Dotenv](https://pypi.org/project/python-dotenv/) – for managing API keys

---

## 📍 Roadmap

* Export to PDF / Notion / ATS
* Named entity recognition in transcripts
* Custom rubrics for score generation
* Multi-speaker diarization
* Automatic analysis from calendar events

---

## 🧠 Who is this for?

* Hiring Managers & Recruiters
* Startup Founders conducting tech interviews
* Bootcamp Instructors assessing mock sessions
* Anyone who wants objective, data-rich interview analysis

---

## 🙋‍♂️ Credits

Built with ❤️ by [Toni Ramchandani](https://github.com/toni-ramchandani)

---

## 📄 License

MIT License

```

---

Let me know if you'd like the screenshot placeholder replaced with an actual image or need deployment instructions for Hugging Face Spaces or Docker.
```
