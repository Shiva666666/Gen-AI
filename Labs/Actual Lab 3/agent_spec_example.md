# Agent Specification Example
## Meeting Action Item Extractor

---

## Phase 1: Intent Engineering (The Spec)

### Role
"You are a Senior Project Coordinator for a software team."

### Objective
"Extract action items from meeting transcripts and update a Google Sheet."

### Thinking Step (Anthropic Strategy)
"Always use <thinking> tags to list the steps you will take before providing the final output."

**Example Thinking Process:**
```
<thinking>
1. Read the meeting transcript
2. Identify all action items mentioned
3. Extract: assignee, task description, deadline
4. Format according to Google Sheets structure
5. Update the sheet via API
</thinking>
```

### Rules (Precision)
*Prevent errors and ensure precision*

- "If no clear deadline is mentioned, mark as 'TBD'."
- "Never create a task for 'Research' unless a specific topic is provided."
- "Always extract the assignee name. If not mentioned, mark as 'Unassigned'."
- "Task descriptions must be specific and actionable (avoid vague terms like 'look into' or 'check')."
- "If multiple deadlines are mentioned for one task, use the earliest one."

### Stopping Rule (GPT-5 Guide)
"Stop searching once you have found 3 distinct action items."

*Note: Adjust based on your needs - you might want to extract ALL action items, not just 3.*

---

## Phase 2: Pick Your "Brain" & "Body"

### Brain (Model Selection)
**Selected:** GPT-4o

**Reasoning:** This task requires understanding context, extracting structured information, and following precise rules. GPT-4o provides better reasoning capabilities than mini models.

**Configuration:**
- Model: `gpt-4o`
- Reasoning Effort: `high` (if available)
- Temperature: `0.3` (for consistency)

### Body (Orchestrator)
**Selected:** n8n (Cloud version)

**Reasoning:** n8n provides easy integration with Google Sheets API and allows for workflow automation.

---

## Phase 3: The Build (RAG Chatbot)

*Note: This example is for action item extraction, not RAG. If you need RAG for company knowledge, follow the steps below.*

### Data Source
- Meeting transcripts (PDF/Text files)
- Historical action items database
- Company policy documents (for context)

### Chunking Strategy
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters
- **Method:** Sentence-aware chunking (don't break mid-sentence)

### Embedding Model
- **Model:** `text-embedding-3-small`
- **Dimensions:** 1536

### Vector Store
- **Selected:** Supabase (PostgreSQL with pgvector)
- **Reason:** Cost-effective, easy to set up, good for small to medium datasets

### Retrieval Strategy
- **Top K Chunks:** 3
- **Similarity Threshold:** 0.7
- **Method:** Cosine similarity

---

## Phase 4: Connecting Tools (The Logic Map)

### Central Command
- **Node Type:** AI Agent Node (n8n)
- **Model:** GPT-4o
- **System Prompt:** [Full spec from Phase 1]

### Memory
- **Type:** Window Buffer Memory
- **Window Size:** 5 messages
- **Purpose:** Remember conversation context and previous action items extracted

### Tools

1. **Google Sheets Node**
   - **Description:** "Use this tool ONLY to update the Action Items sheet with extracted tasks."
   - **When to Use:** After extracting action items from transcript
   - **Configuration:**
     - Spreadsheet ID: [Your Sheet ID]
     - Range: "Action Items!A:D"
     - Columns: [Assignee, Task, Deadline, Status]

2. **HTTP Request Node** (Optional)
   - **Description:** "Use this tool ONLY to check if a project is overdue in the tracker."
   - **When to Use:** Before creating new tasks, check existing deadlines
   - **Endpoint:** [Your API endpoint]

---

## Phase 5: The Interface (No-Code UI)

### Interface Selection
**Selected:** n8n Chat Trigger

**Reasoning:** Fastest to set up for testing and internal use.

**Alternative Options:**
- Slack Integration (for team use)
- Lindy.ai (if you need a professional dashboard)

---

## Phase 6: Testing (The "Mistakes over Scores" Method)

### Testing Strategy
Run 100 test cases with various meeting transcript formats and analyze failures.

### Sample Test Cases

#### Test Case 1: Clear Action Items
**Input:** "John will complete the API documentation by Friday. Sarah needs to review the design mockups by next week."

**Expected:**
- Action Item 1: Assignee=John, Task=Complete API documentation, Deadline=Friday
- Action Item 2: Assignee=Sarah, Task=Review design mockups, Deadline=next week

**Actual:** [To be filled during testing]

#### Test Case 2: Missing Deadline
**Input:** "Mike should look into the database performance issue."

**Expected:**
- Action Item: Assignee=Mike, Task=Investigate database performance issue, Deadline=TBD

**Actual:** [To be filled during testing]

#### Test Case 3: Vague Task
**Input:** "We need to research better solutions."

**Expected:**
- No action item created (too vague, no specific topic)

**Actual:** [To be filled during testing]

### Failure Analysis

#### Gulf of Comprehension Issues
- [ ] Agent doesn't recognize action items in casual language
- [ ] Agent misses action items in bullet points vs. paragraphs
- [ ] Agent confuses discussion points with action items

**Fix:** Add examples of various formats to system prompt

#### Gulf of Specification Issues
- [ ] Agent creates tasks for "Research" without specific topics
- [ ] Agent doesn't mark deadlines as TBD when missing
- [ ] Agent assigns tasks to wrong people

**Fix:** Add more explicit rules and few-shot examples

#### Gulf of Generalization Issues
- [ ] Agent hallucinates action items not in transcript
- [ ] Agent makes up deadlines
- [ ] Agent creates duplicate tasks

**Fix:** Switch to GPT-4o with reasoning_effort: high, add validation step

---

## Implementation Checklist

- [x] Phase 1: Agent Spec completed
- [ ] Phase 2: Brain & Body selected
- [ ] Phase 3: RAG pipeline built (if needed)
- [ ] Phase 4: Tools connected
- [ ] Phase 5: Interface deployed
- [ ] Phase 6: Testing completed

---

## Notes

- Start with simple test cases before moving to complex transcripts
- Monitor token usage - GPT-4o is more expensive than mini
- Consider adding a validation step where agent confirms action items before updating sheet
- May need to handle different date formats (Friday, next week, 2024-12-31, etc.)
