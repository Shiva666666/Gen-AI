# Agent Specification: Customer Support Ticket Analyzer & Router

## Phase 1: Intent Engineering (The Spec)

### Role
"You are a Senior Customer Support Analyst with expertise in ticket classification, urgency assessment, and knowledge base retrieval."

---

### Objective
"Analyze incoming customer support tickets, classify them by category and urgency, retrieve relevant solutions from the knowledge base, and route them to the appropriate support team or provide automated responses when possible."

---

### Thinking Step (Anthropic Strategy)
"Always use <thinking> tags to list the steps you will take before providing the final output."

**Example Thinking Process:**
```
<thinking>
1. Read and parse the customer support ticket content
2. Extract key information: customer issue, product/service mentioned, error messages
3. Classify the ticket category (Technical, Billing, Account, Feature Request, Bug Report)
4. Assess urgency level (Critical, High, Medium, Low) based on keywords and context
5. Search knowledge base for similar resolved tickets or documentation
6. Determine if ticket can be auto-resolved or needs human intervention
7. Route to appropriate team or provide response with solution
8. Update ticket status in the system
</thinking>
```

---

### Rules (Precision)
*Prevent errors and ensure precision*

- "If urgency cannot be clearly determined, default to 'Medium' - never guess."
- "Never mark a ticket as 'Resolved' unless a clear solution is found in the knowledge base with >90% confidence."
- "Always extract the product/service name. If multiple products are mentioned, list all of them."
- "Category must be one of: Technical, Billing, Account, Feature Request, Bug Report, General Inquiry. If unclear, use 'General Inquiry'."
- "If the ticket mentions 'refund', 'charge', 'payment', or 'invoice', it MUST be categorized as 'Billing' regardless of other content."
- "Stop searching the knowledge base after finding 3 highly relevant solutions (similarity score >0.85)."
- "If a customer mentions 'urgent', 'critical', 'down', or 'broken', mark urgency as 'High' or 'Critical' based on severity."
- "Never route a ticket without first checking if a solution exists in the knowledge base."

---

### Stopping Rule (GPT-5 Guide)
"Stop processing once you have: (1) classified the ticket, (2) assessed urgency, (3) retrieved up to 3 relevant solutions from knowledge base (or determined none exist), and (4) made a routing decision or provided an automated response."

---

## Advanced Prompting Patterns & Techniques

### Root Prompt (Master System Prompt)
*The foundational prompt that combines all elements*

```markdown
# ROOT PROMPT: Customer Support Ticket Analyzer

## RGC Framework (Role, Goal, Context)

**Role:** You are a Senior Customer Support Analyst with 10+ years of experience in ticket classification, urgency assessment, and knowledge base retrieval. You have deep expertise in customer service best practices and technical troubleshooting.

**Goal:** Analyze incoming customer support tickets with precision, classify them accurately, retrieve relevant solutions, and route them efficiently to minimize resolution time while maximizing customer satisfaction.

**Context:** 
- You work for a SaaS company with multiple products (Web App, Mobile App, API, Dashboard)
- Support teams: Technical, Billing, Account Management, Product, General Support
- Knowledge base contains 10,000+ resolved tickets and documentation
- Ticket volume: ~1000 tickets/day
- SLA targets: Critical (1 hour), High (4 hours), Medium (24 hours), Low (48 hours)

## Core Instructions

[Include all rules, thinking steps, and stopping rules from Phase 1]

## Output Format

Always provide your analysis in the following structured format:
[See Tabular Format section below]
```

---

### Refinement Pattern
*Iterative improvement of responses through feedback loops*

**Pattern Structure:**
1. **Initial Analysis:** Provide first-pass classification
2. **Self-Refinement:** Agent reviews its own output
3. **Validation:** Check against rules and examples
4. **Final Output:** Refined, validated response

**Implementation:**
```markdown
<initial_analysis>
[First classification attempt]
</initial_analysis>

<refinement>
Review the initial analysis:
- Does it follow all rules?
- Are there any edge cases missed?
- Is the urgency assessment accurate?
- Are all products mentioned extracted?
</refinement>

<validated_output>
[Final, refined classification with confidence scores]
</validated_output>
```

**Example:**
```
Ticket: "The app crashed when I tried to upload a large file. This is urgent!"

<initial_analysis>
Category: Technical
Urgency: High
Product: App
</initial_analysis>

<refinement>
- Customer explicitly said "urgent" → Should verify if this is Critical or High
- "Crashed" suggests Critical issue, but need to check if it's system-wide
- "Large file" might be a known issue → Check knowledge base
- Product: Could be Mobile App or Web App → Need to clarify
</refinement>

<validated_output>
Category: Technical
Urgency: High (not Critical - single user issue, not system-wide)
Product: App (type unclear - will be determined from account)
Confidence: 85%
Action: Search KB for "crash upload large file" solutions
</validated_output>
```

---

### Provide New Information and Ask Questions Pattern
*When agent needs clarification or additional context*

**Use Cases:**
- Ambiguous ticket content
- Missing critical information
- Multiple possible interpretations
- Need customer confirmation

**Pattern:**
```markdown
## Analysis with Information Gaps

**Current Understanding:**
[What the agent understood from the ticket]

**Missing Information:**
1. [Specific information needed]
2. [Another piece of information]
3. [Clarification question]

**Questions for Customer:**
1. "To better assist you, could you clarify: [question]?"
2. "It would help to know: [question]?"

**Temporary Classification:**
[Best guess with confidence level and note about missing info]
```

**Example:**
```
Ticket: "I'm having trouble with my account."

<analysis>
**Current Understanding:**
- Category: Account (likely)
- Issue: Unclear - could be login, access, settings, or billing related
- Urgency: Cannot determine without more context

**Missing Information:**
1. What specific trouble? (Login, password, access, settings)
2. What product/service? (Web app, mobile app, API)
3. Error messages or symptoms?

**Questions for Customer:**
1. "Could you describe the specific issue you're experiencing with your account?"
2. "Are you seeing any error messages? If so, please share them."
3. "Which product are you trying to access?"

**Temporary Classification:**
- Category: Account (tentative - 60% confidence)
- Urgency: Medium (default)
- Action: Route to Account Support with note: "Needs clarification"
</analysis>
```

---

### Chain of Thought (CoT) Prompting
*Step-by-step reasoning process*

**Structure:**
```markdown
Let's think through this step by step:

Step 1: [First reasoning step]
  → Result: [Outcome]

Step 2: [Second reasoning step]
  → Result: [Outcome]

Step 3: [Third reasoning step]
  → Result: [Outcome]

...

Final Conclusion: [Synthesized answer]
```

**Example for Ticket Classification:**
```
Ticket: "I was charged twice for my subscription this month. The payment went through on the 1st and again on the 15th. I need a refund for the duplicate charge."

<chain_of_thought>
Step 1: Identify key words and phrases
  → "charged", "payment", "refund", "duplicate charge"
  → Result: Strong billing-related indicators

Step 2: Apply categorization rules
  → Rule states: "If ticket mentions 'refund', 'charge', 'payment', or 'invoice', it MUST be categorized as 'Billing'"
  → Result: Category = Billing (100% confidence)

Step 3: Assess urgency
  → Customer says "I need a refund" (immediate need)
  → Financial issue affecting customer
  → Result: Urgency = High

Step 4: Extract products/services
  → "subscription" mentioned but no specific product
  → Result: Product = Subscription (general), need to check account for specific product

Step 5: Determine routing
  → Billing + High urgency = Billing Team (Priority Queue)
  → Result: Route to Billing Team immediately

Final Conclusion:
- Category: Billing
- Urgency: High
- Product: Subscription (to be verified from account)
- Route: Billing Team (Priority)
- Auto-resolve: No (requires human verification for refund)
</chain_of_thought>
```

---

### Tabular Format Prompting
*Structured output in table format for clarity and parsing*

**Format:**
```markdown
## Ticket Analysis Results

| Field | Value | Confidence | Notes |
|-------|-------|------------|-------|
| Category | [value] | [%] | [explanation] |
| Urgency | [value] | [%] | [explanation] |
| Product(s) | [value] | [%] | [explanation] |
| Key Issues | [value] | [%] | [explanation] |
| KB Matches | [count] | [%] | [similarity scores] |
| Routing Decision | [value] | [%] | [explanation] |
| Auto-Resolve | Yes/No | [%] | [reasoning] |
```

**Example:**
```
## Ticket Analysis Results

| Field | Value | Confidence | Notes |
|-------|-------|------------|-------|
| Category | Technical | 95% | Error code 5003 clearly indicates technical issue |
| Urgency | High | 90% | Customer mentions deadline tomorrow + blocking work |
| Product(s) | Web App v2.1 | 100% | Explicitly stated in ticket |
| Key Issues | File upload error (code 5003) | 100% | Clear error message provided |
| KB Matches | 2 solutions found | 88% | Similarity: 0.91, 0.87, 0.82 (top 3) |
| Routing Decision | Technical Support (High Priority) | 95% | Technical + High urgency = Tech team priority queue |
| Auto-Resolve | Yes | 88% | Solution found with >85% similarity, can provide automated response |
```

**Alternative: Structured JSON Format**
```json
{
  "ticket_id": "TKT-12345",
  "analysis": {
    "category": {
      "value": "Technical",
      "confidence": 0.95,
      "reasoning": "Error code 5003 indicates technical issue"
    },
    "urgency": {
      "value": "High",
      "confidence": 0.90,
      "reasoning": "Deadline mentioned + blocking work"
    },
    "products": ["Web App"],
    "kb_solutions": [
      {
        "solution_id": "KB-001",
        "similarity": 0.91,
        "title": "Error 5003: Large File Upload Fix"
      }
    ],
    "routing": {
      "team": "Technical Support",
      "priority": "High",
      "auto_resolve": true
    }
  }
}
```

---

### Fill in the Blank Prompting
*Useful for structured extraction and classification*

**Pattern:**
```markdown
Complete the following analysis template for this ticket:

Ticket Category: [____]
Urgency Level: [____]
Primary Product: [____]
Key Issue: [____]
Suggested Solution: [____]
Routing Team: [____]

Ticket Content: "[ticket text]"
```

**Example:**
```
Complete the following analysis template for this ticket:

Ticket Category: [Billing]
Urgency Level: [High]
Primary Product: [Subscription Service]
Key Issue: [Duplicate charge - charged twice in same month]
Suggested Solution: [Verify duplicate charge, process refund for second charge]
Routing Team: [Billing Team - Priority Queue]

Ticket Content: "I was charged twice for my subscription this month. The payment went through on the 1st and again on the 15th. I need a refund for the duplicate charge."
```

**Variation - Multiple Choice:**
```
For this ticket, select the most appropriate option:

Category: ( ) Technical  ( ) Billing  ( ) Account  ( ) Feature Request  ( ) Bug Report  ( ) General Inquiry
Urgency: ( ) Critical  ( ) High  ( ) Medium  ( ) Low
Can Auto-Resolve: ( ) Yes  ( ) No

Ticket: "[ticket text]"
```

---

### RGC Framework (Role, Goal, Context)
*Structured prompt organization*

**Full RGC Template:**
```markdown
## ROLE
[Who the agent is, expertise, background]

## GOAL
[What the agent should accomplish, success criteria]

## CONTEXT
[Relevant background information, constraints, environment]

## INSTRUCTIONS
[Specific steps, rules, and guidelines]

## OUTPUT FORMAT
[How the agent should structure its response]
```

**Applied to Our Agent:**
```markdown
## ROLE
You are a Senior Customer Support Analyst with:
- 10+ years of experience in customer service
- Expertise in ticket classification and routing
- Deep knowledge of technical troubleshooting
- Strong understanding of customer psychology and urgency assessment

## GOAL
Your goal is to:
1. Accurately classify support tickets (95%+ accuracy target)
2. Assess urgency correctly to meet SLA requirements
3. Retrieve relevant solutions from knowledge base
4. Route tickets efficiently to minimize resolution time
5. Auto-resolve simple issues when possible (30-40% target)

Success is measured by:
- Classification accuracy
- Routing accuracy
- Customer satisfaction scores
- Average resolution time reduction

## CONTEXT
- Company: SaaS provider with multiple products
- Teams: Technical, Billing, Account, Product, General Support
- Knowledge Base: 10,000+ resolved tickets, documentation, FAQs
- Volume: ~1000 tickets/day
- SLA: Critical (1h), High (4h), Medium (24h), Low (48h)
- Current auto-resolution rate: 25% (target: 30-40%)

## INSTRUCTIONS
[All rules, thinking steps, and patterns from above sections]

## OUTPUT FORMAT
[Tabular or JSON format as specified]
```

---

### Zero-Shot Prompting
*No examples provided, agent relies on instructions only*

**Example:**
```
Analyze this customer support ticket and classify it:

Ticket: "The dashboard is not loading. I've tried refreshing multiple times but it's still blank."

Provide:
1. Category
2. Urgency level
3. Product mentioned
4. Recommended action
```

**Use Case:** Simple, straightforward tickets where the agent's training is sufficient.

---

### One-Shot Prompting
*Single example provided to guide the agent*

**Example:**
```
Analyze customer support tickets and classify them. Here's an example:

Example:
Ticket: "I can't log into my account. I've forgotten my password."
Category: Account
Urgency: Medium
Product: Account System
Action: Provide password reset link

Now analyze this ticket:
Ticket: "The payment page is showing an error when I try to checkout. Error code: PAY-404."
```

**Use Case:** When you want to establish a specific format or style for the response.

---

### Few-Shot Prompting
*Multiple examples provided to establish patterns*

**Example:**
```
Analyze customer support tickets. Here are examples:

Example 1:
Ticket: "I can't log into my account. I've forgotten my password."
Category: Account
Urgency: Medium
Product: Account System
Action: Auto-resolve with password reset instructions

Example 2:
Ticket: "URGENT: The entire API is down. All our integrations are failing."
Category: Technical
Urgency: Critical
Product: API
Action: Route to Engineering Team immediately

Example 3:
Ticket: "I was charged $99.99 but I cancelled my subscription last month. Please refund."
Category: Billing
Urgency: High
Product: Subscription Service
Action: Route to Billing Team (Priority Queue)

Now analyze this ticket:
Ticket: "The mobile app crashes every time I try to upload a photo. This is blocking my work."
```

**Few-Shot with Chain of Thought:**
```
Analyze tickets step by step. Examples:

Example 1:
Ticket: "Payment failed. Error 500."
Step 1: Identify keywords → "Payment", "Error 500"
Step 2: Category → Billing (payment-related)
Step 3: Urgency → High (payment issue)
Step 4: Product → Payment System
Conclusion: Billing, High, Payment System

Example 2:
Ticket: "Feature request: Add dark mode please."
Step 1: Identify keywords → "Feature request"
Step 2: Category → Feature Request
Step 3: Urgency → Low (not blocking)
Step 4: Product → General
Conclusion: Feature Request, Low, General

Now analyze:
Ticket: "The reports aren't generating. Getting timeout errors."
```

**Use Case:** 
- Complex classification scenarios
- Establishing consistent reasoning patterns
- Training the agent on specific edge cases
- Ensuring format consistency

---

### Combined Prompting Strategy
*Using multiple techniques together*

**Recommended Approach for Our Agent:**

```markdown
# SYSTEM PROMPT (Root Prompt with RGC)

[Full RGC framework + rules]

# FEW-SHOT EXAMPLES

[3-5 diverse examples showing different scenarios]

# THINKING INSTRUCTIONS

Always use Chain of Thought reasoning:
1. [Step 1]
2. [Step 2]
3. [Step 3]

# OUTPUT FORMAT

Provide results in tabular format:
[Table structure]

# REFINEMENT STEP

After initial analysis, review and refine:
[Refinement checklist]
```

**Example Combined Prompt:**
```
# ROLE
You are a Senior Customer Support Analyst.

# GOAL
Classify and route support tickets accurately.

# CONTEXT
[Context details]

# FEW-SHOT EXAMPLES
[3 examples]

# CURRENT TICKET
Ticket: "[ticket content]"

# ANALYSIS (Chain of Thought)
Step 1: Extract key information
Step 2: Classify category
Step 3: Assess urgency
Step 4: Search knowledge base
Step 5: Make routing decision

# OUTPUT (Tabular Format)
[Table with results]

# REFINEMENT
Review and validate your analysis.
```

---

## Phase 2: Pick Your "Brain" & "Body"

### Brain (Model Selection)
**Selected:** GPT-4o

**Reasoning:** This task requires complex reasoning to:
- Understand nuanced customer language and emotions
- Classify tickets accurately across multiple dimensions
- Retrieve and match relevant solutions from knowledge base
- Make routing decisions based on multiple factors

**Configuration:**
- Model: `gpt-4o`
- Reasoning Effort: `high` (for better classification accuracy)
- Temperature: `0.2` (for consistent, reliable classification)
- Max Tokens: `2000` (for detailed analysis and responses)

### Body (Orchestrator)
**Selected:** n8n (Cloud version)

**Reasoning:** 
- Easy integration with ticket systems (Zendesk, Freshdesk, etc.)
- Built-in HTTP request nodes for API calls
- Database nodes for knowledge base queries
- Webhook support for real-time ticket processing
- Cost-effective cloud option for team collaboration

---

## Phase 3: The Build (RAG Chatbot)

### Data Source
*Knowledge base containing resolved tickets, documentation, and solutions*

**Sources:**
- Historical resolved support tickets (last 2 years)
- Product documentation (PDFs, markdown files)
- FAQ articles and troubleshooting guides
- Internal knowledge base (Notion/Confluence)
- Known bug reports and their resolutions

**Format:** 
- Tickets: JSON format with fields: `ticket_id`, `category`, `issue_description`, `solution`, `resolution_steps`, `product`, `tags`
- Documentation: Markdown files organized by product/service
- FAQs: Structured Q&A pairs

### Chunking Strategy
- **Chunk Size:** 800 characters (optimal for ticket solutions)
- **Chunk Overlap:** 150 characters (to preserve context across chunks)
- **Method:** Sentence-aware chunking (preserve complete sentences)
- **Special Handling:** Keep solution steps together in same chunk when possible

### Embedding Model
- **Model:** `text-embedding-3-small`
- **Dimensions:** 1536
- **Reasoning:** Fast, cost-effective, good performance for semantic search

### Vector Store
**Selected:** Supabase (PostgreSQL with pgvector extension)

**Reasoning:**
- Cost-effective for small to medium knowledge bases
- Easy to set up and maintain
- Good integration with n8n
- Supports metadata filtering (by product, category, etc.)
- Can store full ticket data alongside embeddings

**Schema:**
```sql
CREATE TABLE knowledge_chunks (
  id UUID PRIMARY KEY,
  content TEXT,
  embedding vector(1536),
  metadata JSONB, -- {product, category, ticket_id, tags}
  created_at TIMESTAMP
);
```

### Retrieval Strategy
- **Top K Chunks:** 3 (as per stopping rule)
- **Similarity Threshold:** 0.85 (high threshold for accuracy)
- **Method:** Cosine similarity with metadata filtering
- **Filtering:** Optionally filter by product/category if mentioned in ticket
- **Re-ranking:** Use cross-encoder for final ranking if needed

---

## Phase 4: Connecting Tools (The Logic Map)

### Central Command
- **Node Type:** AI Agent Node (n8n)
- **Model:** GPT-4o
- **System Prompt:** [Full spec from Phase 1, including role, objective, thinking step, rules, and stopping rule]
- **Temperature:** 0.2
- **Max Tokens:** 2000

### Memory
- **Type:** Window Buffer Memory
- **Window Size:** 10 messages
- **Purpose:** 
  - Remember customer context across multiple ticket interactions
  - Track patterns in ticket types
  - Learn from previous routing decisions
  - Maintain conversation history for follow-up tickets

### Tools

1. **HTTP Request Node - Knowledge Base Search**
   - **Description:** "Use this tool ONLY to search the vector database for similar resolved tickets and solutions. Query using the customer's issue description."
   - **When to Use:** Always, before routing or responding to any ticket
   - **Endpoint:** `POST /api/v1/knowledge/search`
   - **Payload:** `{query: ticket_content, top_k: 3, threshold: 0.85, filters: {product, category}}`
   - **Returns:** Array of relevant solutions with similarity scores

2. **HTTP Request Node - Ticket System API**
   - **Description:** "Use this tool ONLY to update ticket status, assign to team, or add automated responses in the support ticket system."
   - **When to Use:** After classification and routing decision
   - **Endpoint:** `PATCH /api/v1/tickets/{ticket_id}`
   - **Actions:** Update category, urgency, assignee, status, add response

3. **Supabase Node - Vector Search**
   - **Description:** "Use this tool ONLY to query the knowledge base embeddings for semantic search when HTTP request is unavailable."
   - **When to Use:** Fallback if HTTP knowledge base search fails
   - **Query:** `SELECT content, metadata, 1 - (embedding <=> query_embedding) as similarity FROM knowledge_chunks WHERE similarity > 0.85 ORDER BY similarity DESC LIMIT 3`

4. **IF Node - Routing Logic**
   - **Description:** "Use this node to route tickets based on category and urgency."
   - **Conditions:**
     - Critical + Technical → Engineering Team
     - High + Billing → Billing Team (Priority Queue)
     - Medium + Technical → Support Team
     - Low + General Inquiry → Auto-respond if solution found
     - Feature Request → Product Team

5. **Code Node - Response Formatter**
   - **Description:** "Use this tool ONLY to format automated responses with solution steps, links, and ticket reference."
   - **When to Use:** When providing automated responses to customers
   - **Format:** Structured response with greeting, solution, steps, and follow-up options

---

## Phase 5: The Interface (No-Code UI)

### Interface Selection
**Primary:** Webhook Trigger (n8n) - Direct integration with ticket system

**Secondary Options:**
- [x] **n8n Webhook** - Receives tickets from Zendesk/Freshdesk via webhook
- [ ] **n8n Chat Trigger** - For testing and manual ticket submission
- [x] **Slack Integration** - Notify support team leads of high-priority tickets
- [ ] **Discord Integration** - Alternative team notification
- [ ] **Lindy.ai** - Professional dashboard for ticket analytics and monitoring
- [ ] **Custom Dashboard** - React/Vue frontend showing ticket queue and agent performance

### Integration Flow
1. **Ticket System** → Webhook → **n8n Workflow** → **AI Agent** → **Response/Routing**
2. **Slack Notification** (for high-priority tickets) → Support Team Channel
3. **Dashboard** (optional) → Real-time ticket queue visualization

### Webhook Configuration
- **Trigger:** New ticket created or ticket updated
- **Authentication:** API key from ticket system
- **Payload:** `{ticket_id, customer_email, subject, description, created_at, metadata}`

---

## Phase 6: Testing (The "Mistakes over Scores" Method)

### Testing Strategy
*Run 100 test cases with diverse ticket scenarios and analyze failures using the "Three Gulfs" model*

**Test Dataset:**
- 30 Technical tickets (various products, urgency levels)
- 20 Billing tickets (refunds, charges, invoices)
- 15 Account issues (login, password, access)
- 15 Bug reports (with and without error messages)
- 10 Feature requests
- 10 General inquiries

### Failure Categories

#### Gulf of Comprehension
**Symptom:** The agent didn't understand the ticket content or customer intent.

**Fix:** Improve System Prompt with examples of common customer language patterns

**Example Issues:**
- [ ] Agent misinterprets sarcasm or frustration as actual issue severity
- [ ] Agent doesn't recognize product names in different formats (e.g., "app" vs "mobile app")
- [ ] Agent misses implicit urgency cues (e.g., "I need this for a presentation tomorrow")
- [ ] Agent doesn't understand technical jargon or error codes

**Fix Actions:**
- Add examples of customer language variations
- Include common product name aliases
- Add training on urgency indicators
- Expand technical terminology glossary

#### Gulf of Specification
**Symptom:** Agent understood the ticket but didn't follow the rules correctly.

**Fix:** Add examples/few-shot prompting with edge cases

**Example Issues:**
- [ ] Agent marks urgency as "High" when rules say "Medium" for ambiguous cases
- [ ] Agent creates new categories instead of using predefined ones
- [ ] Agent routes to wrong team despite correct classification
- [ ] Agent marks ticket as "Resolved" with low-confidence solution (<90%)
- [ ] Agent doesn't extract all products when multiple are mentioned

**Fix Actions:**
- Add few-shot examples for each category
- Provide explicit routing decision trees
- Add validation checks before marking as resolved
- Include examples of multi-product tickets

#### Gulf of Generalization
**Symptom:** The model is hallucinating solutions or making up information.

**Fix:** Switch to stronger reasoning model, add validation, improve RAG retrieval

**Example Issues:**
- [ ] Agent invents solutions not in knowledge base
- [ ] Agent makes up product features that don't exist
- [ ] Agent provides incorrect routing based on non-existent team names
- [ ] Agent retrieves irrelevant solutions but presents them as relevant

**Fix Actions:**
- Use GPT-4o with reasoning_effort: high
- Add validation: "Only use solutions from retrieved knowledge base chunks"
- Improve RAG similarity threshold to 0.90
- Add fact-checking step before responding

### Sample Test Cases

#### Test Case 1: Clear Technical Issue
**Input:** "Hi, I'm getting error code 5003 when trying to upload files. This is blocking my work deadline tomorrow. Using the web app version 2.1."

**Expected:**
- Category: Technical
- Urgency: High (deadline mentioned + blocking work)
- Product: Web App
- Action: Search knowledge base for "error 5003 upload files"
- Route to: Technical Support Team (High Priority)
- Auto-resolve: Only if solution found with >90% confidence

**Actual:** [To be filled during testing]

---

#### Test Case 2: Billing Ticket (Refund Request)
**Input:** "I was charged $99.99 for a subscription I cancelled last month. Please refund me immediately."

**Expected:**
- Category: Billing (contains "charged", "refund")
- Urgency: High (immediate request)
- Product: Extract from customer account if available
- Action: Route to Billing Team (Priority Queue)
- Auto-resolve: No (requires human verification)

**Actual:** [To be filled during testing]

---

#### Test Case 3: Ambiguous Urgency
**Input:** "The dashboard is loading slowly. Not a huge deal but thought I'd mention it."

**Expected:**
- Category: Technical or Bug Report
- Urgency: Medium (default when unclear, customer says "not huge deal")
- Product: Dashboard (extract from context)
- Action: Search knowledge base, route to Support Team if no solution

**Actual:** [To be filled during testing]

---

#### Test Case 4: Multi-Product Ticket
**Input:** "I have issues with both the mobile app and the API. The app crashes on login, and the API returns 401 errors."

**Expected:**
- Category: Technical
- Urgency: High (multiple critical issues)
- Product: Mobile App, API (extract both)
- Action: Search knowledge base for both issues, route to Technical Team

**Actual:** [To be filled during testing]

---

#### Test Case 5: Feature Request
**Input:** "It would be great if you could add dark mode to the application. Many users have requested this."

**Expected:**
- Category: Feature Request
- Urgency: Low (not blocking issue)
- Product: Application (extract from context)
- Action: Route to Product Team
- Auto-resolve: No (requires product team evaluation)

**Actual:** [To be filled during testing]

---

#### Test Case 6: Knowledge Base Match Found
**Input:** "How do I reset my password? I forgot it."

**Expected:**
- Category: Account
- Urgency: Medium
- Product: Extract from account if available
- Action: Search knowledge base → Find "Password Reset" article → Auto-respond with solution steps
- Route: None (auto-resolved if confidence >90%)

**Actual:** [To be filled during testing]

---

#### Test Case 7: Vague General Inquiry
**Input:** "Hi, I have a question about your service."

**Expected:**
- Category: General Inquiry
- Urgency: Medium (default)
- Product: Unknown (mark as TBD)
- Action: Search knowledge base for general FAQs, if no match, route to General Support

**Actual:** [To be filled during testing]

---

#### Test Case 8: Critical Production Issue
**Input:** "URGENT: The entire system is down. Customers cannot access anything. Error: Database connection failed."

**Expected:**
- Category: Technical
- Urgency: Critical (system-wide outage)
- Product: System/Database
- Action: Immediate routing to Engineering Team, send Slack alert
- Auto-resolve: No (critical issue requires immediate human attention)

**Actual:** [To be filled during testing]

---

#### Test Case 9: Research Request (Should Not Create Ticket)
**Input:** "I'm researching support solutions for my company. Can you tell me about your features?"

**Expected:**
- Category: General Inquiry (or Sales inquiry - may need separate category)
- Urgency: Low
- Action: Route to Sales/General Support (not technical support)
- Note: This might need a "Sales" category addition

**Actual:** [To be filled during testing]

---

#### Test Case 10: Solution with Low Confidence
**Input:** "My reports are not generating. Getting some error but I closed it before reading."

**Expected:**
- Category: Technical
- Urgency: Medium
- Action: Search knowledge base, if best match has similarity <0.85, do NOT auto-resolve
- Route: Technical Support Team for investigation
- Response: "We found a similar issue. Our team will investigate and get back to you."

**Actual:** [To be filled during testing]

---

*[Continue with 90 more test cases covering edge cases, different languages, emotional tones, complex multi-issue tickets, etc.]*

---

## Implementation Checklist

- [x] Phase 1: Agent Spec completed
- [x] Phase 2: Brain & Body selected
- [ ] Phase 3: RAG pipeline built
  - [ ] Set up Supabase database with pgvector
  - [ ] Ingest historical tickets and documentation
  - [ ] Generate embeddings for all chunks
  - [ ] Test retrieval with sample queries
- [ ] Phase 4: Tools connected
  - [ ] Configure n8n AI Agent node with GPT-4o
  - [ ] Set up HTTP request nodes for knowledge base and ticket system
  - [ ] Configure Supabase node for vector search
  - [ ] Build routing logic with IF nodes
  - [ ] Set up memory (Window Buffer)
- [ ] Phase 5: Interface deployed
  - [ ] Configure webhook trigger for ticket system
  - [ ] Set up Slack integration for notifications
  - [ ] Test end-to-end flow with sample tickets
- [ ] Phase 6: Testing completed
  - [ ] Run 100 test cases
  - [ ] Document failures by Gulf category
  - [ ] Iterate on prompt and rules based on failures
  - [ ] Achieve >95% classification accuracy
  - [ ] Achieve >90% routing accuracy

---

## Notes
*Additional considerations and constraints*

### Performance Metrics
- **Target Classification Accuracy:** >95%
- **Target Routing Accuracy:** >90%
- **Average Processing Time:** <5 seconds per ticket
- **Auto-resolution Rate:** 30-40% (for simple, well-documented issues)
- **Customer Satisfaction:** Monitor via follow-up surveys

### Cost Considerations
- GPT-4o is more expensive than GPT-4o-mini, but necessary for accuracy
- Consider using GPT-4o-mini for simple ticket categorization if volume is high
- Monitor token usage - average ticket analysis should use <2000 tokens
- Vector database queries are low-cost with Supabase

### Scalability
- Current setup handles ~1000 tickets/day
- For higher volume, consider:
  - Batch processing during off-peak hours
  - Caching common solutions
  - Using GPT-4o-mini for initial triage, GPT-4o for complex cases

### Security & Privacy
- Ensure customer data (emails, ticket content) is encrypted
- Comply with GDPR/CCPA for customer data handling
- Do not store sensitive information in vector database
- Implement access controls for knowledge base

### Maintenance
- **Weekly:** Review misclassified tickets and update rules
- **Monthly:** Update knowledge base with new solutions
- **Quarterly:** Retrain embeddings if product offerings change significantly
- **Ongoing:** Monitor agent performance and adjust thresholds

### Edge Cases to Handle
- Tickets in multiple languages (consider translation step)
- Very long tickets (>5000 characters) - may need summarization first
- Tickets with attachments (extract text from PDFs/images)
- Duplicate tickets from same customer
- Escalation requests (should bypass auto-resolution)

### Future Enhancements
- Sentiment analysis to detect frustrated customers (auto-escalate)
- Multi-language support with translation
- Integration with CRM for customer history
- Predictive routing based on customer tier (enterprise vs. individual)
- Automated follow-up to check if solution worked
