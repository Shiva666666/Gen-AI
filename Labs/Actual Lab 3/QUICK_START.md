# Quick Start Guide

## Run the Agent

```bash
# Install dependencies (optional - code works without them)
pip install -r requirements.txt

# Run the demo
python ticket_analyzer_agent.py
```

## What's Included

### ✅ Complete Implementation

1. **All Prompting Techniques:**
   - ✅ Root Prompt with RGC Framework
   - ✅ Chain of Thought reasoning
   - ✅ Few-shot examples
   - ✅ Tabular format output
   - ✅ Refinement pattern
   - ✅ Zero-shot, One-shot, Few-shot

2. **RAG Pipeline:**
   - ✅ Vector embeddings (sentence-transformers)
   - ✅ Semantic search
   - ✅ Knowledge base management
   - ✅ Similarity threshold filtering

3. **Ticket Analysis:**
   - ✅ Category classification
   - ✅ Urgency assessment
   - ✅ Product extraction
   - ✅ Auto-resolution logic
   - ✅ Routing decisions

4. **Rule Implementation:**
   - ✅ All 8 precision rules from spec
   - ✅ Automatic validation
   - ✅ Refinement logic

## File Structure

```
ticket_analyzer_agent.py  # Main implementation (single file)
requirements.txt          # Dependencies
README.md                 # Full documentation
QUICK_START.md           # This file
```

## Key Classes

- `TicketAnalyzerAgent`: Main agent class
- `PromptTemplates`: All prompt templates
- `VectorStore`: RAG vector store
- `EmbeddingModel`: Text embeddings
- `LLMInterface`: LLM API interface

## Example Usage

```python
from ticket_analyzer_agent import TicketAnalyzerAgent, Ticket
from datetime import datetime

# Initialize
agent = TicketAnalyzerAgent()

# Create ticket
ticket = Ticket(
    ticket_id="TKT-001",
    content="I can't log in. Forgot my password.",
    customer_email="user@example.com"
)

# Analyze
analysis = agent.analyze(ticket)

# View results
print(agent.format_analysis_table(analysis))
```

## Features

- ✅ Works without API keys (mock mode)
- ✅ All prompting techniques implemented
- ✅ Complete RAG pipeline
- ✅ Rule-based validation
- ✅ Structured output
- ✅ Ready for production use

## Next Steps

1. Set `OPENAI_API_KEY` environment variable for real LLM responses
2. Add more knowledge base entries
3. Integrate with your ticket system
4. Customize rules and thresholds
5. Deploy to n8n or other orchestration tools
