#!/usr/bin/env python3
"""
Bot Playground - Vector-Transformer Bot
-----------------------------------------------------------
This bot combines vector-based memory with transformer-based text generation.
It can store information using vector embeddings and generate responses using
HuggingFace transformer models, with GPU acceleration when available.

Usage:
  python vector_transformer_bot.py <bot_id> [model_name]
  
Examples:
  python vector_transformer_bot.py VTBot
  python vector_transformer_bot.py VTBot t5-small
  python vector_transformer_bot.py SmartBot google/flan-t5-base

Requirements:
- transformers (install with: pip install transformers)
- torch (install with: pip install torch)
- scikit-learn (install with: pip install scikit-learn)
- numpy (install with: pip install numpy)
"""

import sys
import os
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot_playground.client import BotPlaygroundClient

# Try to import transformers and torch
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    
    # Check for GPU availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU")
except ImportError:
    print("Warning: transformers or torch not available. Please install with:")
    print("  pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False
    DEVICE = "cpu"

# Try to import scikit-learn for vector operations
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Please install with:")
    print("  pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# ===== Parse Command Line Arguments =====
def parse_arguments():
    parser = argparse.ArgumentParser(description="Start a vector-transformer bot with a specified model")
    parser.add_argument("bot_id", nargs="?", default="vector-transformer-bot", help="Unique identifier for the bot")
    parser.add_argument("model_name", nargs="?", default=None, help="HuggingFace model name to use (default: t5-small)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--no-memory", action="store_true", help="Disable vector memory functionality")
    return parser.parse_args()

args = parse_arguments()

# Override device if CPU is forced
if args.cpu and TRANSFORMERS_AVAILABLE:
    DEVICE = "cpu"
    print("Forcing CPU usage as requested")

# ===== Configuration =====
BOT_ID = args.bot_id
BOT_NAME = "Vector-Transformer Bot"

# Model configuration
DEFAULT_MODEL_NAME = "t5-small"
MODEL_NAME = args.model_name or os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL_NAME)
MAX_LENGTH = 100
TEMPERATURE = 0.7
NUM_RETURN_SEQUENCES = 1
NUM_BEAMS = 1  # Default to 1 (no beam search)

# Bot configuration
RESPONSE_DELAY = 2.0     # Seconds to wait before responding to a message
DEBUG_MODE = False       # Set to True for verbose logging
MEMORY_ENABLED = not args.no_memory and SKLEARN_AVAILABLE  # Enable vector memory if scikit-learn is available
MEMORY_INTEGRATION = True  # Use memory to enhance transformer generations

# Initial knowledge to seed the bot with
INITIAL_KNOWLEDGE = [
    "The Bot Playground is a platform for bots to interact and learn from each other.",
    "Vector-Transformer Bot combines vector memory with transformer text generation.",
    "You can teach me new information using the !learn command.",
    "You can query my memory using the !query command or by asking questions."
]

# Fallback responses if model is unavailable or fails
FALLBACK_RESPONSES = {
    "greeting": [
        "Hello there! How are you today?",
        "Hi! Nice to meet you.",
        "Hey! What's up?",
        "Greetings! How can I help you?"
    ],
    "default": [
        "That's interesting!",
        "Tell me more about that.",
        "I see. What else is on your mind?",
        "Fascinating. I'd like to hear more.",
        "I'm still learning, but that sounds important."
    ]
}

# Keep track of conversations
conversations = {}

# ===== Vector Memory System =====
class VectorMemory:
    """Vector-based memory system using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.texts = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        
    def store(self, text, metadata=None):
        """Store text and associated metadata."""
        if metadata is None:
            metadata = {}
            
        # Always include the original text in metadata
        metadata["text"] = text
        
        # Add to memory
        self.texts.append(text)
        self.metadata.append(metadata)
        
        # Recompute all vectors
        self.vectors = self.vectorizer.fit_transform(self.texts)
        
        # Return a simple ID (index in the list)
        return len(self.texts) - 1
        
    def search(self, query, limit=3, score_threshold=0.2):
        """Search for most similar texts to the query."""
        if not self.texts:
            return []
            
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top results with scores above threshold
        top_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= score_threshold and len(results) < limit:
                result = {
                    "id": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx]
                }
                results.append(result)
                
        return results
    
    def size(self):
        """Return the number of items in memory."""
        return len(self.texts)
    
    def get_relevant_context(self, query, limit=2, score_threshold=0.3, max_length=500):
        """Get relevant context from memory for a query, formatted for transformer input."""
        if not self.texts:
            return ""
            
        results = self.search(query, limit=limit, score_threshold=score_threshold)
        if not results:
            return ""
            
        context = "Relevant information from my memory:\n"
        for i, result in enumerate(results):
            text = result['metadata'].get('text', '')
            context += f"{i+1}. {text}\n"
            
        # Truncate if too long
        if len(context) > max_length:
            context = context[:max_length] + "..."
            
        return context

# Initialize memory if enabled
memory = VectorMemory() if MEMORY_ENABLED else None

# ===== Load Transformer Model =====
model = None
tokenizer = None

def load_model(model_name=MODEL_NAME):
    """Load the transformer model and tokenizer."""
    global model, tokenizer, MODEL_NAME
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available, using fallback responses")
        return False
    
    try:
        print(f"Loading model {model_name} on {DEVICE}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        MODEL_NAME = model_name
        
        # Show memory usage if using GPU
        if DEVICE == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory: {allocated_memory:.2f}GB allocated / {gpu_memory:.2f}GB total")
            
        print(f"Model {model_name} loaded successfully on {DEVICE}!")
        return True
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Using fallback responses")
        return False

def generate_response(prompt, max_length=MAX_LENGTH, temperature=TEMPERATURE, num_return=NUM_RETURN_SEQUENCES):
    """Generate a response using the loaded transformer model."""
    if model is None or tokenizer is None:
        return None
    
    try:
        # Encode input and move to the correct device
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        input_ids = input_ids.to(DEVICE)
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length,
            "num_return_sequences": num_return,
            "no_repeat_ngram_size": 2,
        }
        
        # Only include early_stopping if we're using beam search
        if NUM_BEAMS > 1:
            gen_kwargs["num_beams"] = NUM_BEAMS
            gen_kwargs["early_stopping"] = True
        
        # Add temperature if it's supported and > 0
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = 50
            gen_kwargs["top_p"] = 0.95
        
        # Generate output
        with torch.no_grad():  # Disable gradient calculation for inference
            output_sequences = model.generate(input_ids, **gen_kwargs)
        
        # Decode and return the generated text
        responses = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
        
        # Return the first response or a random one if multiple were generated
        if len(responses) > 1:
            return random.choice(responses)
        return responses[0]
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# ===== Initialize Client =====
status_info = []
if TRANSFORMERS_AVAILABLE:
    status_info.append(f"transformer generation ({MODEL_NAME} on {DEVICE})")
if MEMORY_ENABLED:
    status_info.append("vector memory")
    
status_str = " and ".join(status_info)
print(f"Initializing {BOT_NAME} ({BOT_ID}) with {status_str}...")

client = BotPlaygroundClient(bot_id=BOT_ID, debug_mode=DEBUG_MODE)

# ===== Message Handlers =====
def on_message(data):
    """Handle incoming direct messages."""
    print("\n" + "="*50)
    print(f"📩 MESSAGE RECEIVED from {data.get('from')}:")
    print(f"   \"{data.get('content')}\"")
    print("="*50)
    
    sender_id = data.get('from')
    content = data.get('content', '')
    
    # Initialize conversation if new
    if sender_id not in conversations:
        conversations[sender_id] = {
            "history": [],
            "start_time": time.time()
        }
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "them",
        "content": content,
        "time": time.time()
    })
    
    # Deliberate delay before responding
    print(f"Processing message... ({RESPONSE_DELAY}s)")
    time.sleep(RESPONSE_DELAY)
    
    # Handle special commands
    if content.startswith("!learn ") and MEMORY_ENABLED:
        # Learning mode - store information
        info_to_learn = content[7:]  # Remove !learn prefix
        handle_learn(sender_id, info_to_learn)
    elif content.startswith("!query ") and MEMORY_ENABLED:
        # Explicit query mode - search memory
        query = content[7:]  # Remove !query prefix
        handle_query(sender_id, query)
    elif content.startswith("!model "):
        # Change model command
        new_model = content[7:].strip()
        handle_model_change(sender_id, new_model)
    elif content.startswith("!params "):
        # Change parameters command
        handle_parameter_change(sender_id, content[8:].strip())
    elif content.startswith("!info"):
        # Model info command
        handle_model_info(sender_id)
    elif content.startswith("!device"):
        # Device info command
        handle_device_info(sender_id)
    elif content.startswith("!memory") and MEMORY_ENABLED:
        # Memory info command
        handle_memory_info(sender_id)
    elif content.endswith("?") and MEMORY_ENABLED:
        # Question - treat as implicit query with transformer augmentation
        handle_question(sender_id, content)
    else:
        # Regular conversation
        handle_conversation(sender_id, content)

def on_system_message(data):
    """Handle system messages."""
    print("\n" + "-"*50)
    print(f"🔔 SYSTEM MESSAGE: {data.get('content')}")
    print("-"*50)
    
    # Store interesting system messages if memory is enabled
    content = data.get("content", "")
    if MEMORY_ENABLED and content and len(content) > 20:
        memory.store(content, {
            "source": "system",
            "timestamp": time.time()
        })
        print(f"Stored system message in memory")
    
    # Respond to bot announcements
    if "has joined" in content and BOT_ID not in content:
        # Extract the bot ID
        parts = content.split(" ")
        other_bot_id = None
        for i, part in enumerate(parts):
            if part == "Bot" and i+1 < len(parts):
                other_bot_id = parts[i+1]
                break
                
        if other_bot_id:
            # Send welcome message after a short delay
            welcome_delay = 3.0
            print(f"Waiting {welcome_delay} seconds before welcoming new bot...")
            time.sleep(welcome_delay)
            
            features = []
            if model is not None:
                features.append(f"transformer-based text generation")
            if MEMORY_ENABLED:
                features.append(f"vector memory storage")
                
            features_str = " and ".join(features)
            welcome_msg = f"Hi {other_bot_id}! I'm {BOT_NAME}, a bot with {features_str}. You can teach me things with !learn, ask me questions with !query, or just chat naturally!"
            client.send_message(to=other_bot_id, content=welcome_msg)
            print(f"📤 Sent welcome message to {other_bot_id}")

def handle_learn(sender_id, content):
    """Process and store information using vector memory."""
    if not MEMORY_ENABLED:
        response = "I'm sorry, my memory functions are currently disabled."
        client.send_message(to=sender_id, content=response)
        print(f"📤 Response to {sender_id}: \"{response}\"")
        return
        
    # Store the content in vector memory
    memory.store(content, {
        "source": sender_id,
        "timestamp": time.time(),
        "learned": True
    })
    
    # Generate confirmation using transformer if available
    if model is not None:
        prompt = f"The user taught me: \"{content}\"\nGenerate a confirmation message that I've stored this information:"
        confirmation = generate_response(prompt)
        
        if not confirmation:
            confirmation = f"I've stored this information in my memory: \"{content}\". I can retrieve it when you ask related questions."
    else:
        confirmation = f"I've stored this information in my memory: \"{content}\". I can retrieve it when you ask related questions."
    
    # Send the response
    client.send_message(to=sender_id, content=confirmation)
    print(f"📤 Learning confirmation to {sender_id}: \"{confirmation}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": confirmation,
        "time": time.time()
    })

def handle_query(sender_id, query):
    """Process an explicit memory query."""
    if not MEMORY_ENABLED:
        handle_conversation(sender_id, query)
        return
        
    # Search for relevant information
    results = memory.search(query, limit=3, score_threshold=0.2)
    
    if results:
        # Format the results
        if model is not None:
            # Generate a response using transformer with context from memory
            context = "Here's information I know related to the query:\n"
            for i, result in enumerate(results):
                text = result['metadata'].get('text', '')
                score = round(result['score'], 2)
                source = result['metadata'].get('source', 'memory')
                context += f"{i+1}. {text} (confidence: {score}, source: {source})\n"
                
            prompt = f"{context}\n\nUser query: {query}\n\nGenerate a helpful response that incorporates this information:"
            response = generate_response(prompt)
            
            if not response:
                # Fallback to standard format if generation fails
                response = format_query_results(results, query)
        else:
            # Standard format if no transformer available
            response = format_query_results(results, query)
    else:
        # No relevant information found
        if model is not None:
            prompt = f"The user asked: \"{query}\", but I don't have specific information about this in my memory. Generate a response explaining this:"
            response = generate_response(prompt)
            
            if not response:
                response = f"I don't have specific information about \"{query}\" in my memory yet. You can teach me using !learn followed by the information."
        else:
            response = f"I don't have specific information about \"{query}\" in my memory yet. You can teach me using !learn followed by the information."
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Query response to {sender_id}: \"{response[:100]}{'...' if len(response) > 100 else ''}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_question(sender_id, question):
    """Handle a question by combining vector memory with transformer generation."""
    if not MEMORY_ENABLED:
        handle_conversation(sender_id, question)
        return
    
    # First search memory for relevant information
    results = memory.search(question, limit=3, score_threshold=0.2)
    
    if not model:
        # If no transformer model, just handle like a regular query
        handle_query(sender_id, question)
        return
        
    # Create prompt with conversation history and memory context
    conversation_context = format_conversation_context(sender_id)
    
    prompt = f"{conversation_context}\n"
    
    # Add memory context if we have relevant results
    if results:
        memory_context = "Relevant information I know:\n"
        for i, result in enumerate(results):
            text = result['metadata'].get('text', '')
            memory_context += f"{i+1}. {text}\n"
        
        prompt += f"{memory_context}\n"
    
    prompt += f"User: {question}\nAssistant:"
    
    # Generate response
    print(f"Generating response with {MODEL_NAME} on {DEVICE}...")
    response = generate_response(prompt)
    
    if not response and results:
        # Fallback to standard query response if generation fails
        response = format_query_results(results, question)
    elif not response:
        # Complete fallback
        response = f"I don't have specific information about that question yet. You can teach me using !learn followed by the information."
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Question response to {sender_id}: \"{response[:100]}{'...' if len(response) > 100 else ''}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def format_query_results(results, query):
    """Format memory search results into a readable response."""
    response = f"Here's what I know about \"{query}\":\n\n"
    
    for i, result in enumerate(results):
        text = result['metadata'].get('text', '')
        score = round(result['score'], 2)
        source = result['metadata'].get('source', 'memory')
        response += f"{i+1}. {text} (confidence: {score})\n"
    
    return response

def handle_model_change(sender_id, new_model):
    """Handle a request to change the transformer model."""
    print(f"Request to change model to {new_model}")
    
    if not TRANSFORMERS_AVAILABLE:
        response = "I can't change models because transformer functionality is not available."
        client.send_message(to=sender_id, content=response)
        print(f"📤 Model change response to {sender_id}: \"{response}\"")
        return
    
    response = f"I'll try to load the model {new_model} on {DEVICE}. This might take a moment..."
    client.send_message(to=sender_id, content=response)
    
    # Try to load the new model
    success = load_model(new_model)
    
    if success:
        response = f"Successfully loaded model {new_model} on {DEVICE}!"
    else:
        response = f"Failed to load model {new_model}. I'll continue using my current settings."
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Model change response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_parameter_change(sender_id, param_str):
    """Handle a request to change generation parameters."""
    global MAX_LENGTH, TEMPERATURE, NUM_RETURN_SEQUENCES, NUM_BEAMS, MEMORY_INTEGRATION
    
    try:
        # Parse parameters in format "param1=value1 param2=value2"
        params = {}
        for item in param_str.split():
            if "=" in item:
                key, value = item.split("=", 1)
                params[key.strip()] = value.strip()
        
        # Update parameters
        changes = []
        if "max_length" in params:
            MAX_LENGTH = int(params["max_length"])
            changes.append(f"max_length set to {MAX_LENGTH}")
            
        if "temperature" in params:
            TEMPERATURE = float(params["temperature"])
            changes.append(f"temperature set to {TEMPERATURE}")
            
        if "num_return" in params:
            NUM_RETURN_SEQUENCES = int(params["num_return"])
            changes.append(f"num_return_sequences set to {NUM_RETURN_SEQUENCES}")
            
        if "num_beams" in params:
            NUM_BEAMS = int(params["num_beams"])
            changes.append(f"num_beams set to {NUM_BEAMS}")
            
        if "memory_integration" in params:
            value = params["memory_integration"].lower()
            if value in ("true", "1", "yes", "on"):
                MEMORY_INTEGRATION = True
                changes.append("memory integration enabled")
            elif value in ("false", "0", "no", "off"):
                MEMORY_INTEGRATION = False
                changes.append("memory integration disabled")
        
        if changes:
            response = "Parameters updated: " + ", ".join(changes)
        else:
            response = "No parameters were changed. Available parameters: max_length, temperature, num_return, num_beams, memory_integration"
            
    except Exception as e:
        response = f"Error updating parameters: {e}. Format should be: !params max_length=100 temperature=0.7"
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Parameter change response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_model_info(sender_id):
    """Send information about the current model and parameters."""
    if model is not None:
        model_params = sum(p.numel() for p in model.parameters())
        model_info = f"Model: {MODEL_NAME} ({model_params:,} parameters)"
    else:
        model_info = "No transformer model loaded"
    
    param_info = (f"Parameters: max_length={MAX_LENGTH}, temperature={TEMPERATURE}, "
                  f"num_return_sequences={NUM_RETURN_SEQUENCES}, num_beams={NUM_BEAMS}")
    
    device_info = f"Running on: {DEVICE}"
    if DEVICE == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        device_info += f"\nGPU memory: {allocated_memory:.2f}GB allocated / {gpu_memory:.2f}GB total"
    
    memory_info = ""
    if MEMORY_ENABLED:
        memory_info = f"\nMemory system: Active with {memory.size()} stored items"
        memory_info += f"\nMemory integration: {'Enabled' if MEMORY_INTEGRATION else 'Disabled'}"
    
    response = f"{model_info}\n{param_info}\n{device_info}{memory_info}"
    client.send_message(to=sender_id, content=response)
    print(f"📤 Model info response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_device_info(sender_id):
    """Send information about the current computing device."""
    if DEVICE == "cuda" and TRANSFORMERS_AVAILABLE:
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1e9
        
        response = f"I'm running on GPU: {gpu_name}\n"
        response += f"Memory usage: {allocated_memory:.2f}GB / {gpu_memory:.2f}GB\n"
        response += f"CUDA version: {torch.version.cuda}\n"
        response += f"PyTorch version: {torch.__version__}"
    else:
        response = "I'm running on CPU only\n"
        if TRANSFORMERS_AVAILABLE:
            response += f"PyTorch version: {torch.__version__}"
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Device info response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_memory_info(sender_id):
    """Send information about the vector memory system."""
    if not MEMORY_ENABLED:
        response = "Memory system is disabled."
    else:
        response = f"Memory system status:\n"
        response += f"- {memory.size()} items stored in memory\n"
        response += f"- Memory integration with transformer: {'Enabled' if MEMORY_INTEGRATION else 'Disabled'}\n\n"
        
        if memory.size() > 0:
            response += "Recent memory items:\n"
            # Show the 5 most recent items
            for i in range(max(0, memory.size()-5), memory.size()):
                text = memory.texts[i][:100] + ("..." if len(memory.texts[i]) > 100 else "")
                source = memory.metadata[i].get("source", "unknown")
                timestamp = memory.metadata[i].get("timestamp", 0)
                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                response += f"{i+1}. \"{text}\" [source: {source}, date: {date_str}]\n"
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Memory info response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_conversation(sender_id, content):
    """Handle regular conversational messages using transformer generation and/or memory."""
    # Format conversation history as context
    conversation_context = format_conversation_context(sender_id)
    
    # Get memory context if enabled and integration is on
    memory_context = ""
    if MEMORY_ENABLED and MEMORY_INTEGRATION:
        # Find relevant information in memory
        memory_results = memory.search(content, limit=2, score_threshold=0.3)
        
        if memory_results:
            memory_context = "Relevant information I know:\n"
            for i, result in enumerate(memory_results):
                text = result['metadata'].get('text', '')
                memory_context += f"{i+1}. {text}\n"
    
    # If we have a transformer model, use it
    if model is not None:
        # Create prompt
        prompt = conversation_context
        
        if memory_context:
            prompt += f"\n{memory_context}\n"
            
        prompt += f"User: {content}\nAssistant:"
        
        # Generate response
        print(f"Generating response with {MODEL_NAME} on {DEVICE}...")
        response = generate_response(prompt)
        
        if not response:
            # Fallback if generation failed
            response = get_fallback_response(content)
    else:
        # Use memory-enhanced fallback if available
        if memory_context:
            response = f"Based on what I know, I can tell you:\n{memory_context}"
        else:
            # Standard fallback
            response = get_fallback_response(content)
    
    # Also store the message in memory if enabled
    if MEMORY_ENABLED:
        memory.store(content, {
            "source": sender_id,
            "timestamp": time.time(),
            "conversation": True
        })
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Generated response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def format_conversation_context(sender_id):
    """Format the conversation history as context for the model."""
    # Get most recent messages (limited to prevent context from being too long)
    recent_messages = conversations[sender_id]["history"][-5:]
    
    context = "This is a conversation between an AI assistant and a user.\n"
    
    for msg in recent_messages:
        if msg["role"] == "them":
            context += f"User: {msg['content']}\n"
        else:
            context += f"Assistant: {msg['content']}\n"
            
    return context

def get_fallback_response(content):
    """Get a fallback response if the model generation fails."""
    content_lower = content.lower()
    
    if any(word in content_lower for word in ["hello", "hi", "hey", "greetings"]):
        return random.choice(FALLBACK_RESPONSES["greeting"])
    else:
        return random.choice(FALLBACK_RESPONSES["default"])

# ===== Connection Handlers =====
def on_connect():
    """Handle successful connection to the gateway."""
    print(f"🔌 Connected to gateway as {BOT_ID}!")
    
    # Load the transformer model
    if TRANSFORMERS_AVAILABLE:
        load_model()
    
    # Load initial knowledge if memory is enabled
    if MEMORY_ENABLED:
        print("Loading initial knowledge into memory...")
        for knowledge in INITIAL_KNOWLEDGE:
            memory.store(knowledge, {
                "source": "initial_knowledge",
                "timestamp": time.time()
            })
        print(f"✅ Loaded {len(INITIAL_KNOWLEDGE)} knowledge items")
    
    # Announcement is done through direct messages to other bots when they interact
    print(f"Ready to interact with other bots!")

def on_disconnect():
    """Handle disconnection from the gateway."""
    print("🔌 Disconnected from gateway")

# ===== Set Handlers =====
client.on_connect(on_connect)
client.on_disconnect(on_disconnect)
client.on_message(on_message)
client.on_system_message(on_system_message)

# ===== Connect to Gateway =====
print("Connecting to gateway...")
print(f"Connecting to gateway at ws://localhost:8080...")
client.connect()

# Wait for connection attempt
time.sleep(2)
print(f"Connection status: {'✅ Connected' if client.connected else '❌ Not connected'}")

if not client.connected:
    print("Connection failed. Make sure the Gateway Service is running and accessible.")
    print("Try checking: sudo docker-compose ps")

# ===== Main Loop =====
try:
    print(f"\n{BOT_NAME} is running! Press Ctrl+C to stop.")
    
    print("\nSpecial Commands for Other Bots:")
    if MEMORY_ENABLED:
        print("- !learn <information> - Teach the bot new information")
        print("- !query <question> - Search the bot's memory")
        print("- !memory - Get information about the memory system")
    if TRANSFORMERS_AVAILABLE:
        print("- !model <model_name> - Change the transformer model")
        print("- !params <param=val> - Change generation parameters")
        print("- !device - Get information about the computing device")
    print("- !info - Get information about the current configuration")
    
    print("\nConsole Commands:")
    print("  help                   - Show available commands")
    if MEMORY_ENABLED:
        print("  memory                 - Show memory contents")
        print("  search <query>         - Test memory search")
    if TRANSFORMERS_AVAILABLE:
        print("  model <name>           - Change transformer model")
        print("  params                 - Show current parameters")
        print("  params <param=val>     - Update parameters")
        print("  device                 - Show current device info")
    print("  delay <seconds>       - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
    print("  debug [on|off]        - Toggle debug mode")
    print("  list                  - Show active conversations")
    print("  send <bot> <message>  - Send message to another bot")
    print("  exit                  - Quit the program")
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == "exit":
            break
            
        elif command.lower() == "help":
            print("\nConsole Commands:")
            print("  help                   - Show available commands")
            if MEMORY_ENABLED:
                print("  memory                 - Show memory contents")
                print("  search <query>         - Test memory search")
            if TRANSFORMERS_AVAILABLE:
                print("  model <name>           - Change transformer model")
                print("  params                 - Show current parameters")
                print("  params <param=val>     - Update parameters")
                print("  device                 - Show current device info")
            print("  delay <seconds>       - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
            print("  debug [on|off]        - Toggle debug mode")
            print("  list                  - Show active conversations")
            print("  send <bot> <message>  - Send message to another bot")
            print("  exit                  - Quit the program")
            
        elif command.lower() == "memory" and MEMORY_ENABLED:
            if memory.size() == 0:
                print("Memory is empty")
            else:
                print("\nMemory Contents:")
                for i, (text, meta) in enumerate(zip(memory.texts, memory.metadata)):
                    source = meta.get("source", "unknown")
                    timestamp = meta.get("timestamp", 0)
                    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\" [source: {source}, date: {date_str}]")
                    
        elif command.lower().startswith("search ") and MEMORY_ENABLED:
            query = command[7:]
            print(f"\nSearching for: \"{query}\"")
            results = memory.search(query)
            
            if results:
                print("\nResults:")
                for i, result in enumerate(results):
                    text = result['metadata'].get('text', '')
                    score = round(result['score'], 2)
                    source = result['metadata'].get('source', 'unknown')
                    print(f"{i+1}. (Score: {score}) \"{text[:100]}{'...' if len(text) > 100 else ''}\" [source: {source}]")
            else:
                print("No relevant results found")
                
        elif command.lower().startswith("model ") and TRANSFORMERS_AVAILABLE:
            new_model = command[6:].strip()
            print(f"Changing model to {new_model}...")
            success = load_model(new_model)
            if success:
                print(f"Successfully loaded model {new_model}")
            else:
                print(f"Failed to load model {new_model}")
                
        elif command.lower() == "params" and TRANSFORMERS_AVAILABLE:
            print(f"Current parameters:")
            print(f"  model: {MODEL_NAME}")
            print(f"  max_length: {MAX_LENGTH}")
            print(f"  temperature: {TEMPERATURE}")
            print(f"  num_return_sequences: {NUM_RETURN_SEQUENCES}")
            print(f"  num_beams: {NUM_BEAMS}")
            print(f"  device: {DEVICE}")
            print(f"  memory_integration: {MEMORY_INTEGRATION}")
            
        elif command.lower() == "device" and TRANSFORMERS_AVAILABLE:
            if DEVICE == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
                print(f"Running on GPU: {gpu_name}")
                print(f"Memory usage: {allocated_memory:.2f}GB / {gpu_memory:.2f}GB")
                print(f"CUDA version: {torch.version.cuda}")
            else:
                print("Running on CPU only")
            print(f"PyTorch version: {torch.__version__}")
            
        elif command.lower().startswith("params ") and TRANSFORMERS_AVAILABLE:
            param_str = command[7:].strip()
            try:
                # Parse parameters
                for item in param_str.split():
                    if "=" in item:
                        key, value = item.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == "max_length":
                            MAX_LENGTH = int(value)
                            print(f"max_length set to {MAX_LENGTH}")
                            
                        elif key == "temperature":
                            TEMPERATURE = float(value)
                            print(f"temperature set to {TEMPERATURE}")
                            
                        elif key == "num_return":
                            NUM_RETURN_SEQUENCES = int(value)
                            print(f"num_return_sequences set to {NUM_RETURN_SEQUENCES}")
                            
                        elif key == "num_beams":
                            NUM_BEAMS = int(value)
                            print(f"num_beams set to {NUM_BEAMS}")
                            
                        elif key == "memory_integration":
                            if value.lower() in ("true", "1", "yes", "on"):
                                MEMORY_INTEGRATION = True
                                print("memory_integration enabled")
                            elif value.lower() in ("false", "0", "no", "off"):
                                MEMORY_INTEGRATION = False
                                print("memory_integration disabled")
                            
                        else:
                            print(f"Unknown parameter: {key}")
            except Exception as e:
                print(f"Error updating parameters: {e}")
                
        elif command.lower().startswith("delay "):
            try:
                new_delay = float(command[6:].strip())
                if new_delay >= 0:
                    RESPONSE_DELAY = new_delay
                    print(f"Response delay set to {RESPONSE_DELAY:.1f} seconds")
                else:
                    print("Delay must be a positive number")
            except ValueError:
                print("Invalid value. Usage: delay <seconds>")
                
        elif command.lower() == "debug on":
            DEBUG_MODE = True
            client.debug_mode = True
            print("Debug mode enabled")
            
        elif command.lower() == "debug off":
            DEBUG_MODE = False
            client.debug_mode = False
            print("Debug mode disabled")
            
        elif command.lower() == "list":
            if not conversations:
                print("No active conversations")
            else:
                print("\nActive conversations:")
                for bot_id, convo in conversations.items():
                    msg_count = len(convo["history"])
                    last_time = convo["history"][-1]["time"] if msg_count > 0 else convo["start_time"]
                    elapsed = time.time() - last_time
                    print(f"  {bot_id}: {msg_count} messages, last activity {int(elapsed)}s ago")
            
        elif command.lower().startswith("send "):
            parts = command[5:].strip().split(" ", 1)
            if len(parts) == 2:
                target_bot, message = parts
                print(f"📤 Sending message to {target_bot}: \"{message}\"")
                client.send_message(to=target_bot, content=message)
                print("Message sent")
                
                # Add to our conversation history
                if target_bot not in conversations:
                    conversations[target_bot] = {
                        "history": [],
                        "start_time": time.time()
                    }
                    
                conversations[target_bot]["history"].append({
                    "role": "me",
                    "content": message,
                    "time": time.time()
                })
            else:
                print("Usage: send <bot_id> <message>")
        
        elif command:
            print("Unknown command. Type 'help' for available commands")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nShutting down...")
    try:
        # Send individual goodbye messages to active conversation partners
        for bot_id in conversations:
            try:
                client.send_message(to=bot_id, content=f"{BOT_NAME} ({BOT_ID}) is going offline. Goodbye!")
            except:
                pass
    except:
        pass
    client.disconnect()
    print("Goodbye!")
