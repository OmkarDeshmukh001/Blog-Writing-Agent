# Blog Writing Agent

An AI-powered blog generation application built with **Python, LangGraph, Streamlit, and Groq**. It automates blog creation through research, content planning, parallel section writing, content merging, and optional AI image generation.

## Features

- **AI-Powered Blog Generation:** Generate structured blog posts from a topic.
- **Web Research:** Retrieve relevant information using Tavily Search.
- **Content Planning:** Organize blogs into logical sections.
- **Parallel Writing:** Generate individual sections through a LangGraph workflow.
- **Content Merging:** Combine generated sections into a coherent blog.
- **AI Image Generation:** Optionally generate and place relevant images using Google's Gemini API.
- **Multiple Writing Options:** Customize the audience, tone, and blog type.
- **Recency-Aware Research:** Filter evidence by publication date when using the relevant research mode.
- **Markdown Output:** Save generated blogs and image assets locally.

## Tech Stack

- **Language:** Python
- **LLM Provider:** Groq
- **LLM Orchestration:** LangGraph
- **Frontend:** Streamlit
- **Web Research:** Tavily Search
- **Image Generation:** Google Gemini
- **Data Validation:** Pydantic

## Project Structure

```text
Blog-Writing-Agent/
├── backend.py
├── frontend.py
├── requirements.txt
├── README.md
├── .gitignore
└── generated_blogs/
    └── <blog_id>/
        ├── blog.md
        └── images/
```

_The `generated_blogs/` directory is created when the application generates output._

## Prerequisites

- Python 3.11 or a compatible Python version supported by your dependencies
- A Groq API key
- A Tavily API key for web research
- A Google API key for optional image generation

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OmkarDeshmukh001/Blog-Writing-Agent.git
cd Blog-Writing-Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

**Command Prompt:**

```cmd
.venv\Scripts\activate
```

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Git Bash:**

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
```

- `GROQ_API_KEY` is required for Groq-based language model calls.
- `TAVILY_API_KEY` enables web research.
- `GOOGLE_API_KEY` enables optional image generation if configured in the backend.

Use only the variable names expected by your code. The backend may also support `GEMINI_API_KEY` for image generation.

**Security:** Never commit your `.env` file or expose API keys in your GitHub repository.

## Run the Application

Start the Streamlit frontend from the project root:

```bash
python -m streamlit run frontend.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. **Input:** Provide the blog topic and desired writing preferences.
2. **Routing:** The AI workflow determines the generation approach and research requirements.
3. **Research:** Search for relevant web sources when research is enabled and configured.
4. **Planning:** Create a structured outline for the blog.
5. **Section Generation:** Write sections using a LangGraph workflow.
6. **Content Integration:** Merge the sections into a complete Markdown blog.
7. **Image Generation:** Optionally generate and insert relevant images.
8. **Output:** Save the final blog and generated assets in the output directory.

## Output

Generated content is stored locally under:

```text
generated_blogs/<blog_id>/blog.md
generated_blogs/<blog_id>/images/
```

The Markdown file contains the generated blog, while the images directory stores generated image assets when image generation is enabled.

## Troubleshooting

### Groq token limit error

If you receive a `413 Request too large` or tokens-per-minute error:

- Reduce the number of research queries.
- Limit the number of search results.
- Shorten search-result snippets.
- Reduce the amount of evidence passed to each LLM call.
- Reduce parallel LLM requests if rate limits are being reached.

### Missing dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

### API key errors

Verify that your `.env` file exists, the variable names match the backend code, and the keys are valid.

### Streamlit does not start

Ensure the virtual environment is activated and run the command from the project root:

```bash
python -m streamlit run frontend.py
```

## Future Improvements

- Add blog export options such as PDF and DOCX.
- Improve citation verification and source quality.
- Add persistent storage for blog history.
- Add automated testing and deployment.
- Improve rate-limit handling and token usage optimization.

## Author

**Omkar Deshmukh**

- GitHub: [OmkarDeshmukh001](https://github.com/OmkarDeshmukh001)

## License

Add a license file if you intend to distribute this project publicly.
