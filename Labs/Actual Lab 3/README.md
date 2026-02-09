# Customer Support Ticket Analyzer & Router

A complete AI agent implementation that analyzes customer support tickets using advanced prompting techniques, RAG (Retrieval-Augmented Generation), and intelligent routing.

## Features

✅ **Advanced Prompting Techniques:**
- Root Prompt with RGC Framework (Role, Goal, Context)
- Chain of Thought (CoT) reasoning
- Few-shot prompting with examples
- Tabular format output
- Refinement pattern for validation
- Zero-shot, One-shot, Few-shot variations

✅ **RAG Implementation:**
- Vector embeddings using sentence-transformers
- Semantic search in knowledge base
- Similarity threshold filtering
- Top-K retrieval

✅ **Intelligent Classification:**
- Category classification (Technical, Billing, Account, etc.)
- Urgency assessment (Critical, High, Medium, Low)
- Product extraction
- Auto-resolution capability

✅ **Rule-Based Validation:**
- 8 precision rules for accurate classification
- Automatic refinement and validation
- Confidence scoring

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set up OpenAI API key for real LLM responses:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```python
from ticket_analyzer_agent import TicketAnalyzerAgent, Ticket
from datetime import datetime

# Initialize agent
agent = TicketAnalyzerAgent()

# Create a ticket
ticket = Ticket(
    ticket_id="TKT-001",
    content="I can't log into my account. I've forgotten my password.",
    customer_email="user@example.com",
    created_at=datetime.now().isoformat()
)

# Analyze ticket
analysis = agent.analyze(ticket)

# Print results
print(agent.format_analysis_table(analysis))
```

### Run Demo

```bash
python ticket_analyzer_agent.py
```

This will run 5 demo tickets and show the complete analysis workflow.

## Architecture

### Components

1. **PromptTemplates**: Contains all prompt templates using advanced techniques
2. **EmbeddingModel**: Handles text embeddings (sentence-transformers or fallback)
3. **VectorStore**: In-memory vector store for RAG
4. **LLMInterface**: Interface to OpenAI API (with mock fallback)
5. **TicketAnalyzerAgent**: Main agent class orchestrating the workflow

### Workflow

1. **Ticket Input** → Receives ticket content
2. **Knowledge Base Search** → RAG retrieval of relevant solutions
3. **Prompt Construction** → Builds prompt with all techniques
4. **LLM Analysis** → Gets structured analysis from LLM
5. **Refinement** → Validates and refines against rules
6. **Routing Decision** → Determines team and auto-resolve capability
7. **Response Generation** → Creates customer response if applicable

## Prompting Techniques Used

### 1. Root Prompt (RGC Framework)
- **Role**: Senior Customer Support Analyst
- **Goal**: Accurate classification and routing
- **Context**: Company structure, teams, SLA targets

### 2. Few-Shot Prompting
- 3 comprehensive examples with Chain of Thought
- Shows reasoning process for each example
- Establishes expected output format

### 3. Chain of Thought
- Step-by-step reasoning in `<thinking>` tags
- Transparent decision-making process
- Validates each step

### 4. Refinement Pattern
- Initial analysis
- Self-review against rules
- Validated final output

### 5. Tabular Format
- Structured table output
- Easy to parse programmatically
- Human-readable format

## Knowledge Base

The agent comes with a sample knowledge base. You can add entries:

```python
from ticket_analyzer_agent import KnowledgeBaseEntry

entry = KnowledgeBaseEntry(
    entry_id="KB-006",
    content="Your issue description here",
    solution="Step-by-step solution",
    category="Technical",
    product="Web App",
    tags=["tag1", "tag2"]
)

agent.add_knowledge_base_entry(entry)
```

## Rules Implementation

The agent implements 8 precision rules:

1. Default urgency to 'Medium' if unclear
2. Auto-resolve only with >90% confidence solutions
3. Extract all products mentioned
4. Valid categories only
5. Billing keywords override category
6. Stop after 3 KB matches (>0.85 similarity)
7. Urgency keywords detection
8. Always check KB before routing

## Output Format

The agent provides structured output:

```json
{
  "ticket_id": "TKT-001",
  "category": "Account",
  "urgency": "Medium",
  "products": ["Account System"],
  "key_issues": ["Login failure", "Password forgotten"],
  "kb_matches": [...],
  "routing_team": "Account Management",
  "auto_resolve": true,
  "confidence": 0.90,
  "reasoning": "..."
}
```

## Mock Mode

If OpenAI API is not available, the agent uses a rule-based mock classifier. This allows testing without API costs but with reduced accuracy.

## Customization

### Change Model
Edit `LLMInterface.analyze_ticket()` to use different models:
- GPT-4o (default)
- GPT-4o-mini (faster, cheaper)
- Claude (if using Anthropic API)

### Adjust Thresholds
Modify in `VectorStore.search()`:
- `top_k`: Number of results (default: 3)
- `threshold`: Similarity threshold (default: 0.85)

### Add More Rules
Extend `_refine_analysis()` method to add custom validation rules.

## Testing

Run the demo to see the agent in action:
```bash
python ticket_analyzer_agent.py
```

Test with custom tickets:
```python
ticket = Ticket(
    ticket_id="TEST-001",
    content="Your ticket content here"
)
analysis = agent.analyze(ticket)
```

## Dependencies

- `openai`: For GPT-4o API (optional)
- `sentence-transformers`: For embeddings (optional, has fallback)
- `scikit-learn`: For cosine similarity (optional, has fallback)
- `numpy`: For numerical operations

All dependencies are optional - the code has fallbacks for offline use.

## License

This is a lab/educational project.

## Notes

- The agent works without API keys using mock responses
- Embeddings use sentence-transformers (free, local)
- Vector store is in-memory (can be extended to use Pinecone/Supabase)
- All prompting techniques from the spec are implemented
- Ready for integration with n8n or other orchestration tools
