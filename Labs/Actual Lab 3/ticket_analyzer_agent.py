"""
Customer Support Ticket Analyzer & Router
A complete AI agent implementation with RAG, advanced prompting, and routing logic.
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import math

# Try to import optional dependencies
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not found. Using simple similarity calculation.")

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not found. Using mock responses.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not found. Using simple embeddings.")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Ticket:
    """Represents a customer support ticket"""
    ticket_id: str
    content: str
    customer_email: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class KnowledgeBaseEntry:
    """Represents a knowledge base entry"""
    entry_id: str
    content: str
    solution: str
    category: str
    product: str
    tags: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class TicketAnalysis:
    """Result of ticket analysis"""
    ticket_id: str
    category: str
    urgency: str
    products: List[str]
    key_issues: List[str]
    kb_matches: List[Dict]
    routing_team: str
    auto_resolve: bool
    confidence: float
    reasoning: str
    customer_response: Optional[str] = None
    needs_clarification: bool = False
    clarification_questions: List[str] = None


@dataclass
class Message:
    """Represents a message in the conversation history"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    ticket_id: Optional[str] = None


# ============================================================================
# PROMPT TEMPLATES (All Advanced Prompting Techniques)
# ============================================================================

class PromptTemplates:
    """Contains all prompt templates using advanced prompting techniques"""
    
    # Root Prompt with RGC Framework
    ROOT_PROMPT = """# ROOT PROMPT: Customer Support Ticket Analyzer

## RGC Framework (Role, Goal, Context)

**ROLE:** You are a Senior Customer Support Analyst with 10+ years of experience in ticket classification, urgency assessment, and knowledge base retrieval. You have deep expertise in customer service best practices and technical troubleshooting.

**GOAL:** Analyze incoming customer support tickets with precision, classify them accurately, retrieve relevant solutions, and route them efficiently to minimize resolution time while maximizing customer satisfaction.

**CONTEXT:**
- Company: SaaS provider with products: Web App, Mobile App, API, Dashboard
- Support teams: Technical, Billing, Account Management, Product, General Support
- Knowledge base: 10,000+ resolved tickets and documentation
- Ticket volume: ~1000 tickets/day
- SLA targets: Critical (1 hour), High (4 hours), Medium (24 hours), Low (48 hours)

## CORE RULES

1. If urgency cannot be clearly determined, default to 'Medium' - never guess.
2. Never mark a ticket as 'Resolved' unless a clear solution is found in the knowledge base with >90% confidence.
3. Always extract the product/service name. If multiple products are mentioned, list all of them.
4. Category must be one of: Technical, Billing, Account, Feature Request, Bug Report, General Inquiry. If unclear, use 'General Inquiry'.
5. If the ticket mentions 'refund', 'charge', 'payment', or 'invoice', it MUST be categorized as 'Billing' regardless of other content.
6. Stop searching the knowledge base after finding 3 highly relevant solutions (similarity score >0.85).
7. If a customer mentions 'urgent', 'critical', 'down', or 'broken', mark urgency as 'High' or 'Critical' based on severity.
8. Never route a ticket without first checking if a solution exists in the knowledge base.

## THINKING PROCESS

Always use <thinking> tags to show your reasoning process step by step.

## OUTPUT FORMAT

Provide results in structured JSON format (see examples below).
"""

    # Few-Shot Examples with Chain of Thought
    FEW_SHOT_EXAMPLES = """
## FEW-SHOT EXAMPLES WITH CHAIN OF THOUGHT

**Example 1:**
Ticket: "I can't log into my account. I've forgotten my password."

<thinking>
Step 1: Extract key information
  → Keywords: "can't log in", "forgotten password"
  → Product: Account system (implied)
  → Error: None mentioned

Step 2: Classify category
  → Account-related issue (login/password)
  → Category: Account

Step 3: Assess urgency
  → No explicit urgency mentioned
  → Not blocking other users
  → Urgency: Medium (default per rule #1)

Step 4: Search knowledge base
  → Query: "password reset login"
  → Expected: Password reset procedure

Step 5: Routing decision
  → Account category + Medium urgency
  → Can likely auto-resolve with password reset link
  → Route: Auto-resolve if KB solution found
</thinking>

Result:
{
  "category": "Account",
  "urgency": "Medium",
  "products": ["Account System"],
  "key_issues": ["Login failure", "Password forgotten"],
  "kb_matches": [{"entry_id": "KB-001", "similarity": 0.92, "solution": "Password reset procedure"}],
  "routing_team": "Account Management",
  "auto_resolve": true,
  "confidence": 0.90,
  "reasoning": "Standard password reset issue with high-confidence solution found"
}

---

**Example 2:**
Ticket: "URGENT: The entire API is down. All our integrations are failing. Error 503."

<thinking>
Step 1: Extract key information
  → Keywords: "URGENT", "API down", "all integrations failing", "Error 503"
  → Product: API
  → Scope: System-wide (all integrations)

Step 2: Classify category
  → Technical issue (API down, error code)
  → Category: Technical

Step 3: Assess urgency
  → Explicitly marked "URGENT"
  → System-wide outage affecting all users
  → Urgency: Critical (per rule #7)

Step 4: Search knowledge base
  → Query: "API down error 503"
  → May find similar incidents but critical issues need human attention

Step 5: Routing decision
  → Technical + Critical = Engineering Team
  → Cannot auto-resolve (system-wide issue)
  → Route: Engineering Team immediately
</thinking>

Result:
{
  "category": "Technical",
  "urgency": "Critical",
  "products": ["API"],
  "key_issues": ["API outage", "Error 503", "System-wide failure"],
  "kb_matches": [{"entry_id": "KB-045", "similarity": 0.87, "solution": "Similar incident documented"}],
  "routing_team": "Technical",
  "auto_resolve": false,
  "confidence": 1.0,
  "reasoning": "Critical system-wide outage requires immediate engineering attention"
}

---

**Example 3:**
Ticket: "I was charged $99.99 but I cancelled my subscription last month. Please refund immediately."

<thinking>
Step 1: Extract key information
  → Keywords: "charged", "cancelled", "refund"
  → Product: Subscription service
  → Issue: Billing error

Step 2: Classify category
  → Rule #5: "If ticket mentions 'refund', 'charge', 'payment', or 'invoice', it MUST be categorized as 'Billing'"
  → Category: Billing (100% certain)

Step 3: Assess urgency
  → "immediately" indicates high urgency
  → Financial issue affecting customer
  → Urgency: High

Step 4: Search knowledge base
  → Query: "refund cancelled subscription"
  → May find policy but refunds require verification

Step 5: Routing decision
  → Billing + High urgency = Billing Team Priority Queue
  → Cannot auto-resolve (requires human verification)
  → Route: Billing Team (Priority)
</thinking>

Result:
{
  "category": "Billing",
  "urgency": "High",
  "products": ["Subscription Service"],
  "key_issues": ["Duplicate charge", "Refund request"],
  "kb_matches": [{"entry_id": "KB-078", "similarity": 0.85, "solution": "Refund policy documentation"}],
  "routing_team": "Billing",
  "auto_resolve": false,
  "confidence": 0.95,
  "reasoning": "Billing issue with refund request requires human verification per rule #2"
}
"""

    @staticmethod
    def build_analysis_prompt(ticket: Ticket, kb_context: str = "") -> str:
        """Build the complete prompt for ticket analysis"""
        prompt = f"""{PromptTemplates.ROOT_PROMPT}

{PromptTemplates.FEW_SHOT_EXAMPLES}

---

## CURRENT TICKET

Ticket ID: {ticket.ticket_id}
Ticket Content: "{ticket.content}"
Customer: {ticket.customer_email or 'Not provided'}
Created: {ticket.created_at or datetime.now().isoformat()}

## KNOWLEDGE BASE CONTEXT

{kb_context if kb_context else "No relevant solutions found in knowledge base."}

## ANALYSIS INSTRUCTIONS

Use Chain of Thought reasoning in <thinking> tags:
1. Extract key information from the ticket
2. Classify category (apply rule #4 and #5)
3. Assess urgency (apply rule #1 and #7)
4. Extract products mentioned (apply rule #3)
5. Review knowledge base matches provided above
6. Make routing decision (apply rule #8)
7. Determine if auto-resolve is possible (apply rule #2)

## OUTPUT FORMAT

Provide your analysis as a JSON object with this exact structure:

{{
  "category": "Technical|Billing|Account|Feature Request|Bug Report|General Inquiry",
  "urgency": "Critical|High|Medium|Low",
  "products": ["list of products mentioned"],
  "key_issues": ["list of main issues"],
  "kb_matches": [{{"entry_id": "KB-XXX", "similarity": 0.XX, "solution": "brief description"}}],
  "routing_team": "Technical|Billing|Account Management|Product|General Support",
  "auto_resolve": true|false,
  "confidence": 0.XX,
  "reasoning": "explanation of your analysis"
}}

## REFINEMENT STEP

After your initial analysis, review:
- [ ] Does category follow rule #4 and #5?
- [ ] Is urgency assessment accurate per rule #1 and #7?
- [ ] Are all products extracted per rule #3?
- [ ] Was knowledge base checked per rule #8?
- [ ] Is auto-resolve decision justified per rule #2?

Provide your final validated analysis.
"""
        return prompt


# ============================================================================
# EMBEDDING & VECTOR SEARCH (RAG Implementation)
# ============================================================================

class EmbeddingModel:
    """Handles text embeddings for RAG"""
    
    def __init__(self):
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Loaded sentence-transformers model for embeddings")
            except Exception as e:
                print(f"Could not load sentence-transformers: {e}")
                self.model = None
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.model:
            return self.model.encode(text).tolist()
        else:
            # Simple fallback: character frequency vector
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Simple embedding fallback"""
        text_lower = text.lower()
        # Create a simple feature vector
        vector = [0.0] * dim
        for i, char in enumerate(text_lower[:dim]):
            vector[i % dim] += ord(char) / 1000.0
        # Normalize
        norm = math.sqrt(sum(x*x for x in vector))
        return [x/norm if norm > 0 else 0.0 for x in vector]


class DocumentChunker:
    """Implements chunking strategy from spec: 800 chars, 150 overlap, sentence-aware"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text preserving sentence boundaries"""
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunks and self.overlap > 0:
                    # Take last overlap chars from previous chunk
                    prev_chunk = chunks[-1] if chunks else ""
                    overlap_text = prev_chunk[-self.overlap:] if len(prev_chunk) > self.overlap else prev_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]


class VectorStore:
    """Simple in-memory vector store with chunking support"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.entries: List[KnowledgeBaseEntry] = []
        self.embeddings: List[List[float]] = []
        self.chunker = DocumentChunker(chunk_size=800, overlap=150)
    
    def add_entry(self, entry: KnowledgeBaseEntry, chunk_large_docs: bool = True):
        """Add entry to vector store, chunking if needed"""
        # If content is large, chunk it
        if chunk_large_docs and len(entry.content) > 800:
            chunks = self.chunker.chunk_text(entry.content)
            for i, chunk in enumerate(chunks):
                chunk_entry = KnowledgeBaseEntry(
                    entry_id=f"{entry.entry_id}-chunk{i}",
                    content=chunk,
                    solution=entry.solution,
                    category=entry.category,
                    product=entry.product,
                    tags=entry.tags
                )
                embedding = self.embedding_model.embed(chunk)
                chunk_entry.embedding = embedding
                self.entries.append(chunk_entry)
                self.embeddings.append(embedding)
        else:
            embedding = self.embedding_model.embed(entry.content)
            entry.embedding = embedding
            self.entries.append(entry)
            self.embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.85) -> List[Tuple[KnowledgeBaseEntry, float]]:
        """Search for similar entries"""
        if not self.entries:
            return []
        
        query_embedding = self.embedding_model.embed(query)
        similarities = []
        
        for i, entry_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, entry_embedding)
            if similarity >= threshold:
                similarities.append((self.entries[i], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        if HAS_SKLEARN:
            return float(cosine_similarity([vec1], [vec2])[0][0])
        else:
            # Manual calculation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(a * a for a in vec2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)


# ============================================================================
# LLM INTERFACE
# ============================================================================

class LLMInterface:
    """Interface to LLM (OpenAI or mock)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        self.use_mock = False
        
        if HAS_OPENAI and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                print("Initialized OpenAI client")
            except Exception as e:
                print(f"Could not initialize OpenAI: {e}")
                self.use_mock = True
        else:
            self.use_mock = True
            print("Using mock LLM responses (set OPENAI_API_KEY for real responses)")
    
    def analyze_ticket(self, prompt: str) -> Dict:
        """Send prompt to LLM and get analysis"""
        if self.use_mock:
            return self._mock_analysis(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that analyzes support tickets and returns structured JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return self._mock_analysis(prompt)
    
    def _mock_analysis(self, prompt: str) -> Dict:
        """Mock analysis for testing without API"""
        # Simple rule-based classification
        content = prompt.lower()
        
        category = "General Inquiry"
        urgency = "Medium"
        auto_resolve = False
        confidence = 0.7
        
        # Rule #5: Billing keywords
        if any(word in content for word in ['refund', 'charge', 'payment', 'invoice', 'billing']):
            category = "Billing"
            urgency = "High"
            confidence = 0.9
        
        # Technical keywords
        elif any(word in content for word in ['error', 'bug', 'crash', 'down', 'broken', 'api', 'code']):
            category = "Technical"
            if any(word in content for word in ['urgent', 'critical', 'down', 'broken']):
                urgency = "High"
            confidence = 0.85
        
        # Account keywords
        elif any(word in content for word in ['login', 'password', 'account', 'access']):
            category = "Account"
            confidence = 0.8
        
        # Feature request
        elif any(word in content for word in ['feature', 'request', 'add', 'suggest']):
            category = "Feature Request"
            urgency = "Low"
            confidence = 0.75
        
        return {
            "category": category,
            "urgency": urgency,
            "products": ["Web App"] if "web" in content else ["Mobile App"] if "mobile" in content else ["General"],
            "key_issues": ["Issue identified"],
            "kb_matches": [],
            "routing_team": category,
            "auto_resolve": auto_resolve,
            "confidence": confidence,
            "reasoning": "Mock analysis based on keyword matching"
        }


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class WindowBufferMemory:
    """Window Buffer Memory - remembers last N messages (spec: 10 messages)"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: List[Message] = []
    
    def add_message(self, message: Message):
        """Add message to buffer"""
        self.messages.append(message)
        # Keep only last window_size messages
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]
    
    def get_context(self, ticket_id: Optional[str] = None) -> str:
        """Get conversation context for a ticket"""
        relevant_messages = [m for m in self.messages if not ticket_id or m.ticket_id == ticket_id]
        if not relevant_messages:
            return ""
        
        context = "Previous conversation history:\n"
        for msg in relevant_messages[-5:]:  # Last 5 messages for context
            context += f"{msg.role}: {msg.content}\n"
        return context
    
    def clear(self):
        """Clear all messages"""
        self.messages = []


class RoutingEngine:
    """Implements routing logic from Phase 4 spec"""
    
    @staticmethod
    def determine_team(category: str, urgency: str) -> str:
        """Route ticket based on category and urgency (from spec Phase 4)"""
        # Critical + Technical → Engineering Team
        if urgency == "Critical" and category == "Technical":
            return "Engineering Team"
        
        # High + Billing → Billing Team (Priority Queue)
        if urgency == "High" and category == "Billing":
            return "Billing Team (Priority Queue)"
        
        # Medium + Technical → Support Team
        if urgency in ["Medium", "High"] and category == "Technical":
            return "Technical Support Team"
        
        # Feature Request → Product Team
        if category == "Feature Request":
            return "Product Team"
        
        # Low + General Inquiry → General Support (can auto-respond)
        if urgency == "Low" and category == "General Inquiry":
            return "General Support"
        
        # Default routing by category
        routing_map = {
            "Technical": "Technical Support Team",
            "Billing": "Billing Team",
            "Account": "Account Management",
            "Feature Request": "Product Team",
            "Bug Report": "Technical Support Team",
            "General Inquiry": "General Support"
        }
        
        return routing_map.get(category, "General Support")


class ClarificationDetector:
    """Detects when ticket needs clarification (Provide New Information pattern)"""
    
    @staticmethod
    def needs_clarification(ticket: Ticket, analysis: Dict) -> Tuple[bool, List[str]]:
        """Check if ticket needs clarification"""
        questions = []
        ticket_lower = ticket.content.lower()
        
        # Check for vague language
        vague_phrases = ["trouble", "issue", "problem", "not working", "something wrong"]
        is_vague = any(phrase in ticket_lower for phrase in vague_phrases)
        
        # Check for missing product information
        has_product = len(analysis.get("products", [])) > 0 and analysis["products"][0] != "General"
        
        # Check for missing error details
        has_error_info = any(word in ticket_lower for word in ["error", "code", "message", "failed"])
        
        # Low confidence suggests ambiguity
        low_confidence = analysis.get("confidence", 1.0) < 0.7
        
        if is_vague and not has_error_info:
            questions.append("Could you describe the specific issue you're experiencing?")
            questions.append("Are you seeing any error messages? If so, please share them.")
        
        if not has_product:
            questions.append("Which product or service are you using? (Web App, Mobile App, API, Dashboard)")
        
        if low_confidence:
            questions.append("Could you provide more details about what you were trying to do when the issue occurred?")
        
        needs_clarification = len(questions) > 0
        
        return needs_clarification, questions


class TicketAnalyzerAgent:
    """Main agent class implementing the complete ticket analyzer"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.llm = LLMInterface(openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.memory = WindowBufferMemory(window_size=10)  # From spec Phase 4
        self.routing_engine = RoutingEngine()
        self.clarification_detector = ClarificationDetector()
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with sample entries"""
        sample_entries = [
            KnowledgeBaseEntry(
                entry_id="KB-001",
                content="Password reset login account forgot password",
                solution="Navigate to login page, click 'Forgot Password', enter email, check inbox for reset link",
                category="Account",
                product="Account System",
                tags=["password", "login", "reset"]
            ),
            KnowledgeBaseEntry(
                entry_id="KB-002",
                content="API error 503 service unavailable down",
                solution="Check API status page, verify service health, contact engineering if persistent",
                category="Technical",
                product="API",
                tags=["api", "error", "503", "down"]
            ),
            KnowledgeBaseEntry(
                entry_id="KB-003",
                content="Refund cancelled subscription billing charge",
                solution="Verify subscription cancellation date, check billing records, process refund if duplicate charge confirmed",
                category="Billing",
                product="Subscription Service",
                tags=["refund", "billing", "subscription"]
            ),
            KnowledgeBaseEntry(
                entry_id="KB-004",
                content="Dashboard not loading blank screen",
                solution="Clear browser cache, try incognito mode, check browser console for errors, verify account permissions",
                category="Technical",
                product="Dashboard",
                tags=["dashboard", "loading", "blank"]
            ),
            KnowledgeBaseEntry(
                entry_id="KB-005",
                content="Mobile app crash upload file large",
                solution="Reduce file size, check file format compatibility, ensure app is updated to latest version, try smaller file first",
                category="Technical",
                product="Mobile App",
                tags=["mobile", "crash", "upload", "file"]
            ),
        ]
        
        for entry in sample_entries:
            self.vector_store.add_entry(entry)
        
        print(f"Initialized knowledge base with {len(sample_entries)} entries")
    
    def analyze(self, ticket: Ticket) -> TicketAnalysis:
        """Main analysis method - implements the complete workflow"""
        print(f"\n{'='*60}")
        print(f"Analyzing Ticket: {ticket.ticket_id}")
        print(f"{'='*60}")
        
        # Step 1: Search knowledge base (RAG)
        print("\n[Step 1] Searching knowledge base...")
        kb_matches = self.vector_store.search(ticket.content, top_k=3, threshold=0.85)
        
        # Format KB context for prompt
        kb_context = ""
        if kb_matches:
            kb_context = "Relevant solutions found:\n"
            for entry, similarity in kb_matches:
                kb_context += f"- {entry.entry_id}: {entry.solution} (similarity: {similarity:.2f})\n"
            print(f"Found {len(kb_matches)} relevant solutions")
        else:
            print("No highly relevant solutions found (similarity < 0.85)")
        
        # Step 2: Build prompt with all techniques
        print("\n[Step 2] Building analysis prompt...")
        prompt = PromptTemplates.build_analysis_prompt(ticket, kb_context)
        
        # Step 3: Get LLM analysis
        print("\n[Step 3] Getting LLM analysis...")
        analysis_dict = self.llm.analyze_ticket(prompt)
        
        # Step 4: Refinement (validate against rules)
        print("\n[Step 4] Refining and validating analysis...")
        refined_analysis = self._refine_analysis(analysis_dict, ticket, kb_matches)
        
        # Step 4.5: Check if clarification needed (Provide New Information pattern)
        needs_clarification, questions = self.clarification_detector.needs_clarification(ticket, refined_analysis)
        
        # Step 5: Determine routing using routing engine
        routing_team = self.routing_engine.determine_team(
            refined_analysis["category"],
            refined_analysis["urgency"]
        )
        
        # Step 6: Create TicketAnalysis object
        analysis = TicketAnalysis(
            ticket_id=ticket.ticket_id,
            category=refined_analysis["category"],
            urgency=refined_analysis["urgency"],
            products=refined_analysis["products"],
            key_issues=refined_analysis["key_issues"],
            kb_matches=[
                {
                    "entry_id": entry.entry_id,
                    "similarity": float(similarity),
                    "solution": entry.solution
                }
                for entry, similarity in kb_matches
            ],
            routing_team=routing_team,  # Use routing engine
            auto_resolve=refined_analysis["auto_resolve"],
            confidence=refined_analysis["confidence"],
            reasoning=refined_analysis["reasoning"],
            needs_clarification=needs_clarification,
            clarification_questions=questions if needs_clarification else []
        )
        
        # Step 7: Generate customer response if auto-resolve
        if analysis.auto_resolve and analysis.kb_matches and not needs_clarification:
            analysis.customer_response = self._generate_customer_response(analysis, kb_matches[0][0])
        elif needs_clarification:
            analysis.customer_response = self._generate_clarification_response(analysis)
        
        # Step 8: Store in memory
        self.memory.add_message(Message(
            role="user",
            content=ticket.content,
            timestamp=datetime.now().isoformat(),
            ticket_id=ticket.ticket_id
        ))
        self.memory.add_message(Message(
            role="assistant",
            content=f"Analyzed ticket: {analysis.category}, {analysis.urgency}, routed to {analysis.routing_team}",
            timestamp=datetime.now().isoformat(),
            ticket_id=ticket.ticket_id
        ))
        
        return analysis
    
    def _refine_analysis(self, analysis: Dict, ticket: Ticket, kb_matches: List) -> Dict:
        """Refinement pattern - validate and improve analysis"""
        refined = analysis.copy()
        ticket_lower = ticket.content.lower()
        
        # Rule #5: Billing keywords override
        if any(word in ticket_lower for word in ['refund', 'charge', 'payment', 'invoice']):
            refined["category"] = "Billing"
            refined["reasoning"] += " (Applied rule #5: Billing keywords detected)"
        
        # Rule #1: Default urgency to Medium if unclear
        if refined["urgency"] not in ["Critical", "High", "Medium", "Low"]:
            refined["urgency"] = "Medium"
            refined["reasoning"] += " (Applied rule #1: Default urgency)"
        
        # Rule #7: Urgency keywords
        if any(word in ticket_lower for word in ['urgent', 'critical', 'down', 'broken']):
            if 'down' in ticket_lower or 'broken' in ticket_lower:
                if 'all' in ticket_lower or 'entire' in ticket_lower:
                    refined["urgency"] = "Critical"
                else:
                    refined["urgency"] = "High"
            refined["reasoning"] += " (Applied rule #7: Urgency keywords detected)"
        
        # Rule #2: Auto-resolve only if high confidence solution
        if refined["auto_resolve"]:
            if not kb_matches or kb_matches[0][1] < 0.90:
                refined["auto_resolve"] = False
                refined["reasoning"] += " (Applied rule #2: Solution confidence < 90%)"
        
        # Rule #4: Validate category
        valid_categories = ["Technical", "Billing", "Account", "Feature Request", "Bug Report", "General Inquiry"]
        if refined["category"] not in valid_categories:
            refined["category"] = "General Inquiry"
            refined["reasoning"] += " (Applied rule #4: Invalid category, defaulted)"
        
        return refined
    
    def _generate_customer_response(self, analysis: TicketAnalysis, kb_entry: KnowledgeBaseEntry) -> str:
        """Generate automated customer response"""
        response = f"""Hello,

Thank you for contacting support. We've identified your issue and have a solution for you.

**Issue:** {', '.join(analysis.key_issues)}
**Product:** {', '.join(analysis.products)}

**Solution:**
{kb_entry.solution}

If this doesn't resolve your issue, please reply to this ticket and we'll escalate it to our {analysis.routing_team} team.

Best regards,
Support Team
"""
        return response
    
    def _generate_clarification_response(self, analysis: TicketAnalysis) -> str:
        """Generate clarification request (Provide New Information pattern)"""
        response = f"""Hello,

Thank you for contacting support. To better assist you, we need a bit more information.

**Current Understanding:**
- Category: {analysis.category} (tentative - {analysis.confidence:.0%} confidence)
- Urgency: {analysis.urgency}

**To help us resolve your issue quickly, could you please provide:**

"""
        for i, question in enumerate(analysis.clarification_questions, 1):
            response += f"{i}. {question}\n"
        
        response += f"""
Once we have this information, we'll route your ticket to the {analysis.routing_team} team for immediate assistance.

Best regards,
Support Team
"""
        return response
    
    def format_analysis_table(self, analysis: TicketAnalysis) -> str:
        """Format analysis as a table (tabular format prompting)"""
        table = f"""
{'='*80}
TICKET ANALYSIS RESULTS
{'='*80}

| Field              | Value                                    | Confidence |
|--------------------|------------------------------------------|------------|
| Ticket ID          | {analysis.ticket_id:<40} | 100%       |
| Category           | {analysis.category:<40} | {analysis.confidence*100:.0f}%       |
| Urgency            | {analysis.urgency:<40} | {analysis.confidence*100:.0f}%       |
| Product(s)         | {', '.join(analysis.products):<40} | {analysis.confidence*100:.0f}%       |
| Key Issues         | {', '.join(analysis.key_issues):<40} | {analysis.confidence*100:.0f}%       |
| KB Matches        | {len(analysis.kb_matches)} solutions found{'':<30} | {analysis.confidence*100:.0f}%       |
| Routing Team       | {analysis.routing_team:<40} | {analysis.confidence*100:.0f}%       |
| Auto-Resolve       | {'Yes' if analysis.auto_resolve else 'No':<40} | {analysis.confidence*100:.0f}%       |
| Overall Confidence | {analysis.confidence*100:.0f}%{'':<35} | -          |

{'='*80}
REASONING
{'='*80}
{analysis.reasoning}

"""
        
        if analysis.kb_matches:
            table += f"""
{'='*80}
KNOWLEDGE BASE MATCHES
{'='*80}
"""
            for i, match in enumerate(analysis.kb_matches, 1):
                table += f"""
Match {i}:
  Entry ID: {match['entry_id']}
  Similarity: {match['similarity']:.2%}
  Solution: {match['solution']}
"""
        
        if analysis.auto_resolve and analysis.customer_response:
            table += f"""
{'='*80}
AUTOMATED CUSTOMER RESPONSE
{'='*80}
{analysis.customer_response}
"""
        
        return table
    
    def add_knowledge_base_entry(self, entry: KnowledgeBaseEntry):
        """Add new entry to knowledge base"""
        self.vector_store.add_entry(entry)
        print(f"Added knowledge base entry: {entry.entry_id}")


# ============================================================================
# MAIN EXECUTION & DEMO
# ============================================================================

def main():
    """Main function with demo examples"""
    print("="*80)
    print("Customer Support Ticket Analyzer & Router")
    print("="*80)
    
    # Initialize agent
    agent = TicketAnalyzerAgent()
    
    # Demo tickets
    demo_tickets = [
        Ticket(
            ticket_id="TKT-001",
            content="I can't log into my account. I've forgotten my password.",
            customer_email="user@example.com",
            created_at=datetime.now().isoformat()
        ),
        Ticket(
            ticket_id="TKT-002",
            content="URGENT: The entire API is down. All our integrations are failing. Error 503.",
            customer_email="enterprise@example.com",
            created_at=datetime.now().isoformat()
        ),
        Ticket(
            ticket_id="TKT-003",
            content="I was charged $99.99 but I cancelled my subscription last month. Please refund immediately.",
            customer_email="customer@example.com",
            created_at=datetime.now().isoformat()
        ),
        Ticket(
            ticket_id="TKT-004",
            content="The dashboard is not loading. I've tried refreshing multiple times but it's still blank. This is blocking my work.",
            customer_email="user2@example.com",
            created_at=datetime.now().isoformat()
        ),
        Ticket(
            ticket_id="TKT-005",
            content="The mobile app crashes when I try to upload a large file. This is urgent!",
            customer_email="mobile@example.com",
            created_at=datetime.now().isoformat()
        ),
    ]
    
    # Analyze each ticket
    results = []
    for ticket in demo_tickets:
        analysis = agent.analyze(ticket)
        results.append(analysis)
        
        # Print formatted results
        print(agent.format_analysis_table(analysis))
        print("\n" + "="*80 + "\n")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total tickets analyzed: {len(results)}")
    print(f"Auto-resolved: {sum(1 for r in results if r.auto_resolve)}")
    print(f"Average confidence: {sum(r.confidence for r in results) / len(results):.2%}")
    
    # Category breakdown
    categories = {}
    for r in results:
        categories[r.category] = categories.get(r.category, 0) + 1
    print("\nCategory breakdown:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")


# ============================================================================
# TESTING FRAMEWORK (Phase 6: Three Gulfs Model)
# ============================================================================

@dataclass
class TestCase:
    """Represents a test case"""
    test_id: str
    ticket_content: str
    expected_category: str
    expected_urgency: str
    expected_products: List[str]
    expected_routing: str
    expected_auto_resolve: bool
    gulf_category: Optional[str] = None  # "Comprehension", "Specification", "Generalization"


class TestingFramework:
    """Testing framework implementing Three Gulfs model from Phase 6"""
    
    def __init__(self, agent: TicketAnalyzerAgent):
        self.agent = agent
        self.test_results: List[Dict] = []
    
    def run_test(self, test_case: TestCase) -> Dict:
        """Run a single test case"""
        ticket = Ticket(
            ticket_id=test_case.test_id,
            content=test_case.ticket_content,
            created_at=datetime.now().isoformat()
        )
        
        analysis = self.agent.analyze(ticket)
        
        # Compare with expected
        category_match = analysis.category == test_case.expected_category
        urgency_match = analysis.urgency == test_case.expected_urgency
        routing_match = analysis.routing_team == test_case.expected_routing
        auto_resolve_match = analysis.auto_resolve == test_case.expected_auto_resolve
        
        # Check products (at least one match)
        products_match = any(p in test_case.expected_products for p in analysis.products) or \
                        any(p in analysis.products for p in test_case.expected_products)
        
        passed = category_match and urgency_match and routing_match and products_match
        
        result = {
            "test_id": test_case.test_id,
            "passed": passed,
            "category_match": category_match,
            "urgency_match": urgency_match,
            "routing_match": routing_match,
            "products_match": products_match,
            "auto_resolve_match": auto_resolve_match,
            "expected": {
                "category": test_case.expected_category,
                "urgency": test_case.expected_urgency,
                "routing": test_case.expected_routing
            },
            "actual": {
                "category": analysis.category,
                "urgency": analysis.urgency,
                "routing": analysis.routing_team,
                "confidence": analysis.confidence
            },
            "gulf_category": test_case.gulf_category,
            "failure_reason": self._classify_failure(analysis, test_case) if not passed else None
        }
        
        self.test_results.append(result)
        return result
    
    def _classify_failure(self, analysis: TicketAnalysis, test_case: TestCase) -> str:
        """Classify failure using Three Gulfs model"""
        # Gulf of Comprehension: Agent didn't understand
        if analysis.confidence < 0.6:
            return "Gulf of Comprehension: Low confidence suggests misunderstanding"
        
        # Gulf of Specification: Understood but wrong rules
        if analysis.category != test_case.expected_category:
            return "Gulf of Specification: Category mismatch - rules not applied correctly"
        
        if analysis.urgency != test_case.expected_urgency:
            return "Gulf of Specification: Urgency mismatch - rules not applied correctly"
        
        # Gulf of Generalization: Hallucination
        if analysis.confidence > 0.9 and not analysis.kb_matches:
            return "Gulf of Generalization: High confidence without KB support (possible hallucination)"
        
        return "Unknown failure"
    
    def generate_report(self) -> str:
        """Generate test report"""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["passed"])
        failed = total - passed
        
        # Group by Gulf category
        gulf_counts = {"Comprehension": 0, "Specification": 0, "Generalization": 0, "Other": 0}
        for result in self.test_results:
            if not result["passed"]:
                gulf = result.get("gulf_category") or result.get("failure_reason", "Other")
                if "Comprehension" in gulf:
                    gulf_counts["Comprehension"] += 1
                elif "Specification" in gulf:
                    gulf_counts["Specification"] += 1
                elif "Generalization" in gulf:
                    gulf_counts["Generalization"] += 1
                else:
                    gulf_counts["Other"] += 1
        
        report = f"""
{'='*80}
TEST REPORT - Three Gulfs Analysis
{'='*80}

Total Tests: {total}
Passed: {passed} ({passed/total*100:.1f}%)
Failed: {failed} ({failed/total*100:.1f}%)

Failure Analysis (Three Gulfs Model):
- Gulf of Comprehension: {gulf_counts['Comprehension']} failures
  → Fix: Improve System Prompt with examples
  
- Gulf of Specification: {gulf_counts['Specification']} failures
  → Fix: Add examples/few-shot prompting
  
- Gulf of Generalization: {gulf_counts['Generalization']} failures
  → Fix: Switch to stronger reasoning model, improve RAG

- Other: {gulf_counts['Other']} failures

{'='*80}
DETAILED RESULTS
{'='*80}
"""
        
        for result in self.test_results:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            report += f"\n{status} - {result['test_id']}\n"
            if not result["passed"]:
                report += f"  Expected: {result['expected']}\n"
                report += f"  Actual: {result['actual']}\n"
                report += f"  Failure: {result['failure_reason']}\n"
        
        return report


def create_test_suite() -> List[TestCase]:
    """Create test suite from Phase 6 spec examples"""
    return [
        # Test Case 1: Clear Technical Issue
        TestCase(
            test_id="TC-001",
            ticket_content="Hi, I'm getting error code 5003 when trying to upload files. This is blocking my work deadline tomorrow. Using the web app version 2.1.",
            expected_category="Technical",
            expected_urgency="High",
            expected_products=["Web App"],
            expected_routing="Technical Support Team",
            expected_auto_resolve=False  # Only if >90% confidence solution
        ),
        # Test Case 2: Billing Ticket
        TestCase(
            test_id="TC-002",
            ticket_content="I was charged $99.99 for a subscription I cancelled last month. Please refund me immediately.",
            expected_category="Billing",
            expected_urgency="High",
            expected_products=["Subscription Service"],
            expected_routing="Billing Team (Priority Queue)",
            expected_auto_resolve=False
        ),
        # Test Case 3: Ambiguous Urgency
        TestCase(
            test_id="TC-003",
            ticket_content="The dashboard is loading slowly. Not a huge deal but thought I'd mention it.",
            expected_category="Technical",
            expected_urgency="Medium",
            expected_products=["Dashboard"],
            expected_routing="Technical Support Team",
            expected_auto_resolve=False
        ),
        # Test Case 4: Multi-Product Ticket
        TestCase(
            test_id="TC-004",
            ticket_content="I have issues with both the mobile app and the API. The app crashes on login, and the API returns 401 errors.",
            expected_category="Technical",
            expected_urgency="High",
            expected_products=["Mobile App", "API"],
            expected_routing="Technical Support Team",
            expected_auto_resolve=False
        ),
        # Test Case 5: Feature Request
        TestCase(
            test_id="TC-005",
            ticket_content="It would be great if you could add dark mode to the application. Many users have requested this.",
            expected_category="Feature Request",
            expected_urgency="Low",
            expected_products=["Application"],
            expected_routing="Product Team",
            expected_auto_resolve=False
        ),
        # Test Case 6: Account Issue (should auto-resolve if KB match)
        TestCase(
            test_id="TC-006",
            ticket_content="I can't log into my account. I've forgotten my password.",
            expected_category="Account",
            expected_urgency="Medium",
            expected_products=["Account System"],
            expected_routing="Account Management",
            expected_auto_resolve=True  # If KB solution found with >90% confidence
        ),
    ]


def run_tests():
    """Run the test suite"""
    print("="*80)
    print("Running Test Suite - Phase 6: Three Gulfs Model")
    print("="*80)
    
    agent = TicketAnalyzerAgent()
    test_framework = TestingFramework(agent)
    test_cases = create_test_suite()
    
    print(f"\nRunning {len(test_cases)} test cases...\n")
    
    for test_case in test_cases:
        print(f"Running {test_case.test_id}...")
        result = test_framework.run_test(test_case)
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {status}")
    
    print("\n" + test_framework.generate_report())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        main()
