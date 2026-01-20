"""
DSK Memory System - GMI-style memory at the API level

Instead of injecting into model weights, we inject into prompts:
- FIFO Context: Full conversation history sent each turn
- Knowledge Graph: Concepts and relationships extracted and stored
- Causal Chains: Track cause-effect patterns
- Adapters: Learned response patterns stored as templates
- Compaction: When context overflows, summarize intelligently
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

CONFIG_DIR = Path.home() / ".dsk"
MEMORY_FILE = CONFIG_DIR / "memory.json"
GRAPH_FILE = CONFIG_DIR / "knowledge_graph.json"


class KnowledgeGraph:
    """
    Stores concepts, relationships, and causal chains.
    Injected into prompts as relevant context.
    """

    def __init__(self):
        self.nodes = {}  # concept -> {info, frequency, last_seen}
        self.edges = []  # (from, relation, to)
        self.causal_chains = []  # [(cause, effect, confidence)]
        self.load()

    def load(self):
        """Load graph from disk"""
        if GRAPH_FILE.exists():
            try:
                with open(GRAPH_FILE) as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                    self.causal_chains = data.get("causal_chains", [])
            except:
                pass

    def save(self):
        """Save graph to disk"""
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(GRAPH_FILE, "w") as f:
            json.dump({
                "nodes": self.nodes,
                "edges": self.edges,
                "causal_chains": self.causal_chains
            }, f, indent=2)

    def add_concept(self, concept, info=None):
        """Add or update a concept"""
        concept = concept.lower().strip()
        if concept in self.nodes:
            self.nodes[concept]["frequency"] += 1
            self.nodes[concept]["last_seen"] = datetime.now().isoformat()
            if info:
                self.nodes[concept]["info"] = info
        else:
            self.nodes[concept] = {
                "info": info or "",
                "frequency": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }

    def add_relation(self, from_concept, relation, to_concept):
        """Add a relationship between concepts"""
        edge = (from_concept.lower(), relation.lower(), to_concept.lower())
        if edge not in self.edges:
            self.edges.append(edge)
            self.add_concept(from_concept)
            self.add_concept(to_concept)

    def add_causal(self, cause, effect, confidence=0.5):
        """Add a causal chain"""
        chain = (cause.lower(), effect.lower(), confidence)
        # Update existing or add new
        for i, (c, e, conf) in enumerate(self.causal_chains):
            if c == cause.lower() and e == effect.lower():
                # Strengthen confidence
                self.causal_chains[i] = (c, e, min(1.0, conf + 0.1))
                return
        self.causal_chains.append(chain)

    def get_related(self, concepts, max_hops=2):
        """Get related concepts within N hops"""
        related = set(c.lower() for c in concepts)
        for _ in range(max_hops):
            new_related = set()
            for edge in self.edges:
                if edge[0] in related:
                    new_related.add(edge[2])
                if edge[2] in related:
                    new_related.add(edge[0])
            related.update(new_related)
        return related

    def get_context_injection(self, query, max_items=10):
        """
        Generate context to inject into prompt based on query.
        This is the GMI-style knowledge injection.
        """
        # Extract potential concepts from query
        words = set(re.findall(r'\b\w{3,}\b', query.lower()))

        # Find matching nodes
        relevant_nodes = []
        for concept, data in self.nodes.items():
            if concept in words or any(w in concept for w in words):
                relevant_nodes.append((concept, data))

        # Sort by frequency and recency
        relevant_nodes.sort(key=lambda x: (x[1]["frequency"], x[1]["last_seen"]), reverse=True)
        relevant_nodes = relevant_nodes[:max_items]

        if not relevant_nodes:
            return ""

        # Build context string
        context_parts = ["[Memory Context]"]

        for concept, data in relevant_nodes:
            if data["info"]:
                context_parts.append(f"- {concept}: {data['info']}")

        # Add relevant relationships
        relevant_concepts = set(n[0] for n in relevant_nodes)
        for from_c, rel, to_c in self.edges:
            if from_c in relevant_concepts or to_c in relevant_concepts:
                context_parts.append(f"- {from_c} {rel} {to_c}")

        # Add causal chains
        for cause, effect, conf in self.causal_chains:
            if cause in relevant_concepts or effect in relevant_concepts:
                if conf > 0.5:
                    context_parts.append(f"- {cause} → {effect} (causal)")

        if len(context_parts) > 1:
            return "\n".join(context_parts) + "\n\n"
        return ""

    def extract_from_conversation(self, user_msg, assistant_msg):
        """
        Extract concepts and relationships from a conversation turn.
        This learns from every interaction (like GMI adapters).
        """
        # Simple extraction - in production, could use NLP or another AI call
        # Extract capitalized words as potential concepts
        text = f"{user_msg} {assistant_msg}"

        # Find quoted terms, capitalized words, technical terms
        concepts = set()

        # Quoted terms
        concepts.update(re.findall(r'"([^"]+)"', text))
        concepts.update(re.findall(r"'([^']+)'", text))

        # Technical terms (camelCase, snake_case)
        concepts.update(re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', text))  # camelCase
        concepts.update(re.findall(r'\b\w+_\w+\b', text))  # snake_case

        # Capitalized terms (not at sentence start)
        sentences = text.split(". ")
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words[1:], 1):  # Skip first word
                if word and word[0].isupper() and len(word) > 2:
                    concepts.add(word.strip(".,!?"))

        # Add concepts
        for concept in concepts:
            if len(concept) > 2:
                self.add_concept(concept)

        # Simple relation extraction: "X is Y", "X uses Y", etc.
        patterns = [
            (r'(\w+)\s+is\s+(?:a|an|the)?\s*(\w+)', "is_a"),
            (r'(\w+)\s+uses?\s+(\w+)', "uses"),
            (r'(\w+)\s+requires?\s+(\w+)', "requires"),
            (r'(\w+)\s+contains?\s+(\w+)', "contains"),
            (r'(\w+)\s+depends?\s+on\s+(\w+)', "depends_on"),
        ]

        for pattern, relation in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    self.add_relation(match[0], relation, match[1])

        self.save()


class FIFOMemory:
    """
    FIFO Context Memory - keeps full conversation history.
    Compacts when exceeding token limits.
    """

    def __init__(self, max_tokens=100000):
        self.messages = []
        self.max_tokens = max_tokens
        self.summaries = []  # Compressed old conversations
        self.load()

    def load(self):
        """Load memory from disk"""
        if MEMORY_FILE.exists():
            try:
                with open(MEMORY_FILE) as f:
                    data = json.load(f)
                    self.messages = data.get("messages", [])
                    self.summaries = data.get("summaries", [])
            except:
                pass

    def save(self):
        """Save memory to disk"""
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(MEMORY_FILE, "w") as f:
            json.dump({
                "messages": self.messages,
                "summaries": self.summaries
            }, f, indent=2)

    def estimate_tokens(self, messages):
        """Rough token estimate (4 chars per token)"""
        total = sum(len(m.get("content", "")) for m in messages)
        return total // 4

    def add(self, role, content):
        """Add a message"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.check_overflow()
        self.save()

    def check_overflow(self):
        """
        Handle context overflow with intelligent compaction:
        Keep: 10% start + 20% middle + 30% end + summaries
        """
        tokens = self.estimate_tokens(self.messages)

        if tokens > self.max_tokens:
            total = len(self.messages)
            keep_start = max(2, total // 10)  # 10% from start
            keep_middle = total // 5  # 20% from middle
            keep_end = int(total * 0.3)  # 30% from end

            middle_start = (total - keep_middle) // 2
            middle_end = middle_start + keep_middle

            # Messages to compact (everything else)
            to_compact = []
            kept = []

            for i, msg in enumerate(self.messages):
                if i < keep_start:
                    kept.append(msg)
                elif middle_start <= i < middle_end:
                    kept.append(msg)
                elif i >= total - keep_end:
                    kept.append(msg)
                else:
                    to_compact.append(msg)

            # Create summary of compacted messages
            if to_compact:
                summary = self._create_summary(to_compact)
                self.summaries.append(summary)

            self.messages = kept

    def _create_summary(self, messages):
        """Create a summary of messages (simple version)"""
        topics = set()
        for msg in messages:
            # Extract key terms
            words = msg.get("content", "").split()
            for word in words:
                if len(word) > 5 and word[0].isupper():
                    topics.add(word)

        return {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(messages),
            "topics": list(topics)[:20],
            "summary": f"Discussed: {', '.join(list(topics)[:10])}" if topics else "General conversation"
        }

    def get_context(self):
        """
        Get full context for injection.
        Returns messages formatted for API.
        """
        context_messages = []

        # Add summaries as system context
        if self.summaries:
            summary_text = "Previous conversation summaries:\n"
            for s in self.summaries[-3:]:  # Last 3 summaries
                summary_text += f"- {s['summary']}\n"
            context_messages.append({
                "role": "system",
                "content": summary_text
            })

        # Add actual messages (without timestamps for API)
        for msg in self.messages:
            context_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return context_messages

    def clear(self):
        """Clear current session but keep summaries"""
        if self.messages:
            summary = self._create_summary(self.messages)
            self.summaries.append(summary)
        self.messages = []
        self.save()


class Adapter:
    """
    Prompt-level adapter - learned response patterns.
    Stored as prompt templates that get injected when relevant.
    """

    def __init__(self, name, trigger_keywords, template):
        self.name = name
        self.trigger_keywords = trigger_keywords
        self.template = template
        self.usage_count = 0

    def matches(self, query):
        """Check if this adapter should trigger"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.trigger_keywords)

    def get_injection(self):
        """Get the prompt injection for this adapter"""
        self.usage_count += 1
        return self.template


class AdapterBank:
    """
    Collection of learned adapters.
    Can create new adapters from successful interactions.
    """

    def __init__(self):
        self.adapters = []
        self.adapter_file = CONFIG_DIR / "adapters.json"
        self.load()

    def load(self):
        """Load adapters from disk"""
        if self.adapter_file.exists():
            try:
                with open(self.adapter_file) as f:
                    data = json.load(f)
                    for a in data:
                        self.adapters.append(Adapter(
                            a["name"],
                            a["trigger_keywords"],
                            a["template"]
                        ))
            except:
                pass

        # Add default adapters if none exist
        if not self.adapters:
            self._add_defaults()

    def _add_defaults(self):
        """Add default useful adapters"""
        defaults = [
            Adapter(
                "code_review",
                ["review", "check", "audit", "look at"],
                "When reviewing code, check for: security issues, performance, readability, error handling. Be specific about line numbers."
            ),
            Adapter(
                "debugging",
                ["debug", "error", "fix", "broken", "not working"],
                "For debugging: 1) Identify the error type 2) Locate the source 3) Explain why it happens 4) Provide the fix with before/after code."
            ),
            Adapter(
                "explain",
                ["explain", "how does", "what is", "why"],
                "Explain clearly with: 1) Simple definition 2) How it works 3) A practical example 4) Common gotchas."
            ),
            Adapter(
                "shell_command",
                ["command", "bash", "terminal", "shell", "cli"],
                "For shell commands: provide the exact command, explain each flag, show example output if helpful."
            ),
        ]
        self.adapters.extend(defaults)
        self.save()

    def save(self):
        """Save adapters to disk"""
        CONFIG_DIR.mkdir(exist_ok=True)
        with open(self.adapter_file, "w") as f:
            json.dump([{
                "name": a.name,
                "trigger_keywords": a.trigger_keywords,
                "template": a.template
            } for a in self.adapters], f, indent=2)

    def get_injections(self, query):
        """Get all relevant adapter injections for a query"""
        injections = []
        for adapter in self.adapters:
            if adapter.matches(query):
                injections.append(adapter.get_injection())
        return injections

    def create_adapter(self, name, trigger_keywords, template):
        """Create a new adapter from a successful pattern"""
        adapter = Adapter(name, trigger_keywords, template)
        self.adapters.append(adapter)
        self.save()
        return adapter


class DSKMemory:
    """
    Main memory system combining all GMI-like features.
    """

    def __init__(self):
        self.fifo = FIFOMemory()
        self.graph = KnowledgeGraph()
        self.adapters = AdapterBank()

    def process_turn(self, user_msg, assistant_msg):
        """Process a conversation turn - learn from it"""
        # Add to FIFO
        self.fifo.add("user", user_msg)
        self.fifo.add("assistant", assistant_msg)

        # Extract knowledge
        self.graph.extract_from_conversation(user_msg, assistant_msg)

    def get_enhanced_messages(self, new_query, system_prompt=None):
        """
        Get messages with all GMI-style injections:
        - Knowledge context
        - Adapter templates
        - Conversation history
        """
        messages = []

        # 1. System prompt with injections
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)

        # Adapter injections
        adapter_injections = self.adapters.get_injections(new_query)
        if adapter_injections:
            system_parts.append("\n[Active Patterns]\n" + "\n".join(adapter_injections))

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # 2. Knowledge context injection
        knowledge_context = self.graph.get_context_injection(new_query)
        if knowledge_context:
            messages.append({"role": "system", "content": knowledge_context})

        # 3. Conversation history (FIFO)
        messages.extend(self.fifo.get_context())

        return messages

    def clear_session(self):
        """Clear current session"""
        self.fifo.clear()

    def get_stats(self):
        """Get memory statistics"""
        return {
            "messages": len(self.fifo.messages),
            "summaries": len(self.fifo.summaries),
            "concepts": len(self.graph.nodes),
            "relations": len(self.graph.edges),
            "causal_chains": len(self.graph.causal_chains),
            "adapters": len(self.adapters.adapters)
        }
