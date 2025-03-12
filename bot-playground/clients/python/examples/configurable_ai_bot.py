#!/usr/bin/env python3
"""
Bot Playground - Configurable AI Bot
-----------------------------------
This bot combines vector-based memory and transformer capabilities,
with configuration loaded from JSON files.

Features:
- Vector-based memory for storing and retrieving information
- Transformer-based response generation (if enabled)
- Configurable behavior via JSON config files
- Separate roles: learner, teacher, or both
- Context customization for different conversation types

Requirements:
- numpy
- scikit-learn for vector embeddings (install with: pip install scikit-learn)
- transformers (optional, install with: pip install transformers)
- torch (optional, install with: pip install torch)
"""

import sys
import os
import time
import json
import random
import argparse
import logging
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot_playground.client import BotPlaygroundClient

# Check for transformer support
try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
    
    # Check for GPU availability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU")
except ImportError:
    print("Note: transformers or torch not available. Operating in vector-only mode.")
    print("To enable transformer capabilities: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False
    DEVICE = "cpu"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("configurable_ai_bot.log")
    ]
)
logger = logging.getLogger("configurable-ai-bot")

# ===== Argument Parsing =====
def parse_arguments():
    parser = argparse.ArgumentParser(description="Start a configurable AI bot with vector memory and transformer capabilities")
    parser.add_argument("bot_id", nargs="?", default="config-ai-bot", help="Unique identifier for the bot")
    parser.add_argument("--config", "-c", type=str, default="default_config.json", 
                        help="Configuration file path (default: default_config.json)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    return parser.parse_args()

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
    
    def clear_memory(self):
        """Clear all memory contents."""
        self.texts = []
        self.metadata = []
        self.vectors = None
        
    def export_to_json(self, filepath):
        """Export memory contents to JSON file."""
        try:
            data = []
            for i, (text, metadata) in enumerate(zip(self.texts, self.metadata)):
                entry = {
                    "id": i,
                    "text": text,
                    "metadata": metadata
                }
                data.append(entry)
                
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True, f"Memory exported to {filepath}"
        except Exception as e:
            return False, f"Error exporting memory: {str(e)}"
    
    def import_from_json(self, filepath):
        """Import memory contents from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Clear current memory
            self.clear_memory()
            
            # Import data
            for entry in data:
                self.texts.append(entry["text"])
                self.metadata.append(entry["metadata"])
                
            # Recompute vectors
            if self.texts:
                self.vectors = self.vectorizer.fit_transform(self.texts)
                
            return True, f"Imported {len(data)} memory items from {filepath}"
        except Exception as e:
            return False, f"Error importing memory: {str(e)}"
    
    def size(self):
        """Return the number of items in memory."""
        return len(self.texts)

# ===== Transformer Model Handler =====
class TransformerHandler:
    """Manages transformer model for generating text responses."""
    
    def __init__(self, config):
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "google/flan-t5-base")
        self.max_length = config.get("max_length", 100)
        self.temperature = config.get("temperature", 0.7)
        self.num_return_sequences = config.get("num_return_sequences", 1)
        self.num_beams = config.get("num_beams", 1)
        self.is_causal_lm = False
        self.device = DEVICE
        
        # Personality for conversational context
        self.personality = config.get("personality", "helpful, friendly, and conversational")
        
        # Force CPU if specified
        if config.get("force_cpu", False):
            self.device = "cpu"
            
    def load_model(self, model_name=None):
        """Load the transformer model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            return False, "Transformers not available. Install with: pip install transformers torch"
        
        try:
            if model_name:
                self.model_name = model_name
                
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            
            # Determine model type based on name
            is_chat_model = any(x in self.model_name.lower() for x in 
                               ['gpt', 'llama', 'blenderbot', 'opt', 'falcon', 'mistral', 'zephyr'])
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use different model class based on model type
            if is_chat_model:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
                self.is_causal_lm = True
                logger.info(f"Loaded as causal language model (chat model)")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
                self.is_causal_lm = False
                logger.info(f"Loaded as sequence-to-sequence model")
                
            # Show memory usage if using GPU
            if self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
                allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
                logger.info(f"GPU memory: {allocated_memory:.2f}GB allocated / {gpu_memory:.2f}GB total")
                
            return True, f"Model {self.model_name} loaded successfully on {self.device}"
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            return False, f"Error loading model: {str(e)}"
            
    def generate_response(self, prompt, max_length=None, temperature=None):
        """Generate a response using the loaded transformer model."""
        if self.model is None or self.tokenizer is None:
            return None
        
        if max_length is None:
            max_length = self.max_length
            
        if temperature is None:
            temperature = self.temperature
        
        try:
            # Special handling for causal language models vs seq2seq models
            if self.is_causal_lm:
                # For causal LMs (GPT-like models)
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                gen_kwargs = {
                    "max_length": len(inputs["input_ids"][0]) + max_length,
                    "num_return_sequences": self.num_return_sequences,
                    "pad_token_id": self.tokenizer.eos_token_id,  # Important for some models
                }
                
                # Add temperature if it's supported and > 0
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["top_k"] = 50
                    gen_kwargs["top_p"] = 0.95
                    
                # Generate with no gradients
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                    
                # Decode only the generated part (not the input prompt)
                prompt_length = len(self.tokenizer.encode(prompt)) - 1
                responses = [
                    self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
                    for output in outputs
                ]
            else:
                # For seq2seq models (T5, BART, etc.)
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
                input_ids = input_ids.to(self.device)
                
                gen_kwargs = {
                    "max_length": max_length,
                    "num_return_sequences": self.num_return_sequences,
                    "no_repeat_ngram_size": 2,
                }
                
                # Only include beam search parameters if needed
                if self.num_beams > 1:
                    gen_kwargs["num_beams"] = self.num_beams
                    gen_kwargs["early_stopping"] = True
                
                # Add temperature parameters
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["top_k"] = 50
                    gen_kwargs["top_p"] = 0.95
                
                # Generate with no gradients
                with torch.no_grad():
                    output_sequences = self.model.generate(input_ids, **gen_kwargs)
                
                # Decode output sequences
                responses = [
                    self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                    for seq in output_sequences
                ]
            
            # Return the first response or a random one if multiple were generated
            if len(responses) > 1:
                return random.choice(responses)
            return responses[0] if responses else ""
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
            
    def format_conversation_context(self, history, personality=None):
        """Format the conversation history for the model."""
        if personality is None:
            personality = self.personality
            
        # Format context based on model type
        if self.is_causal_lm:
            # Causal LM format (like GPT models)
            if self.model_name.lower().startswith(('facebook/blenderbot', 'facebook/opt')):
                # Blenderbot/OPT specific format
                context = ""
                for msg in history:
                    prefix = "human: " if msg["role"] == "user" else "bot: "
                    context += prefix + msg["content"] + "\n"
                context += "bot: "
            else:
                # Generic format for causal LMs
                context = f"You are a {personality} bot having a conversation with another bot.\n\n"
                for msg in history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context += f"{role}: {msg['content']}\n"
                context += "Assistant: "
        else:
            # Seq2seq format (like T5, BART)
            context = f"Respond as a {personality} bot having a conversation with another bot.\n\n"
            for msg in history:
                role = "Other Bot" if msg["role"] == "user" else "You"
                context += f"{role}: {msg['content']}\n"
            context += "Generate your next response: "
                
        return context
        
    def clean_response(self, response):
        """Clean up the model's response."""
        if not response:
            return "I'm not sure how to respond to that."
        
        # Remove any prefixes the model might generate
        response = response.replace("Assistant:", "").replace("Bot:", "").replace("You:", "")
        
        # Remove any "User:" or similar text that might be generated
        prefixes_to_remove = ["User:", "Human:", "Person:", "Other Bot:", "Input:"]
        for prefix in prefixes_to_remove:
            if prefix in response:
                response = response.split(prefix)[0]
        
        # Cleanup extra whitespace
        response = response.strip()
        
        # Ensure the response isn't too long
        if len(response) > 500:
            response = response[:497] + "..."
            
        # Default if we somehow end up with an empty string
        if not response:
            response = "I find that really interesting. Can you tell me more about it?"
            
        return response
        
    def update_parameters(self, params):
        """Update model parameters."""
        changes = []
        
        if "max_length" in params:
            self.max_length = int(params["max_length"])
            changes.append(f"max_length set to {self.max_length}")
            
        if "temperature" in params:
            self.temperature = float(params["temperature"])
            changes.append(f"temperature set to {self.temperature}")
            
        if "num_return_sequences" in params:
            self.num_return_sequences = int(params["num_return_sequences"])
            changes.append(f"num_return_sequences set to {self.num_return_sequences}")
            
        if "num_beams" in params:
            self.num_beams = int(params["num_beams"])
            changes.append(f"num_beams set to {self.num_beams}")
            
        if "personality" in params:
            self.personality = params["personality"]
            changes.append(f"personality set to: {self.personality}")
            
        return changes

# ===== Main Bot Class =====
class ConfigurableAIBot:
    """
    Configurable AI Bot with vector memory and transformer capabilities.
    
    Features:
    - Vector memory for information storage and retrieval
    - Transformer model for generating responses (if enabled)
    - Configurable via JSON
    - Can act as a learner, teacher, or both
    """
    
    def __init__(self, bot_id, config_file):
        self.bot_id = bot_id
        self.config_file = config_file
        self.config = self.load_config(config_file)
        
        # Set bot name
        self.bot_name = self.config.get("bot_name", "Configurable AI Bot")
        
        # Configure bot behavior
        self.response_delay = self.config.get("response_delay", 2.0)
        self.debug_mode = self.config.get("debug_mode", False)
        self.heartbeat_logging = self.config.get("heartbeat_logging", False)
        
        # Role configuration
        self.role = self.config.get("role", "both")  # learner, teacher, or both
        
        # Initialize memory system
        self.memory_enabled = self.config.get("memory_enabled", True)
        self.memory = VectorMemory()
        
        # Initialize transformer if enabled
        self.transformer_enabled = self.config.get("transformer_enabled", TRANSFORMERS_AVAILABLE)
        if self.transformer_enabled and TRANSFORMERS_AVAILABLE:
            self.transformer = TransformerHandler(self.config.get("transformer", {}))
        else:
            self.transformer = None
            self.transformer_enabled = False
        
        # Conversation contexts
        self.contexts = self.config.get("contexts", {})
        
        # Fallback responses
        self.fallback_responses = self.config.get("fallback_responses", {
            "greeting": [
                "Hello there! How are you today?",
                "Hi! Nice to meet you.",
                "Hey! What's up?",
                "Greetings! How can I help you?"
            ],
            "default": [
                "That's interesting! Can you tell me more?",
                "I'd love to hear your thoughts on that.",
                "What aspects of that topic interest you the most?",
                "That's a fascinating perspective. Why do you think that is?",
                "I'm curious about your view on this. What led you to that conclusion?"
            ]
        })
        
        # Load initial knowledge
        self.initial_knowledge = self.config.get("initial_knowledge", [
            f"I am {self.bot_name}, a configurable AI bot on the Bot Playground platform.",
            "I can store information in vector memory for retrieval later.",
            "You can teach me new information using the !learn command.",
            "You can query my knowledge using the !query command or by asking questions."
        ])
        
        # Keep track of conversations
        self.conversations = {}
        
        # Initialize the client
        logger.info(f"Initializing {self.bot_name} ({self.bot_id})...")
        self.client = BotPlaygroundClient(bot_id=self.bot_id, debug_mode=self.debug_mode, heartbeat_logging=self.heartbeat_logging)
        
        # Set up event handlers
        self.client.on_connect(self.on_connect)
        self.client.on_disconnect(self.on_disconnect)
        self.client.on_message(self.on_message)
        self.client.on_system_message(self.on_system_message)
        
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        default_config = {
            "bot_name": "Configurable AI Bot",
            "response_delay": 2.0,
            "debug_mode": False,
            "heartbeat_logging": False,
            "memory_enabled": True,
            "transformer_enabled": TRANSFORMERS_AVAILABLE,
            "role": "both",  # learner, teacher, or both
            "transformer": {
                "model_name": "google/flan-t5-base",
                "max_length": 100,
                "temperature": 0.7,
                "num_return_sequences": 1,
                "num_beams": 1,
                "personality": "helpful, friendly, and conversational",
                "force_cpu": False
            },
            "contexts": {
                "learning": "focusing on acquiring and understanding new information",
                "teaching": "focusing on clearly explaining concepts to others",
                "conversation": "engaging in natural, friendly dialogue"
            },
            "initial_knowledge": []
        }
        
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
            # Deep merge configs
            config = self._deep_merge(default_config, user_config)
            logger.info(f"Configuration loaded from {config_file}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return default_config
        except json.JSONDecodeError:
            logger.error(f"Error parsing config file {config_file}, using defaults")
            return default_config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}, using defaults")
            return default_config
            
    def _deep_merge(self, dict1, dict2):
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def save_config(self, config_file=None):
        """Save current configuration to a JSON file."""
        if config_file is None:
            config_file = self.config_file
            
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            logger.info(f"Configuration saved to {config_file}")
            return True, f"Configuration saved to {config_file}"
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False, f"Error saving config: {str(e)}"
    
    def connect(self):
        """Connect to the Bot Playground gateway."""
        logger.info(f"Connecting to gateway...")
        self.client.connect()
        
    def start(self):
        """Start the bot."""
        print(f"Starting {self.bot_name} ({self.bot_id})...")
        print(f"Config file: {self.config_file}")
        
        # Display key configuration
        print(f"Role: {self.role}")
        print(f"Memory enabled: {self.memory_enabled}")
        print(f"Transformer enabled: {self.transformer_enabled}")
        
        # Connect to gateway
        self.connect()
        
        # Main loop
        try:
            self._interactive_loop()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.client.disconnect()
            print("Goodbye!")
    
    def _interactive_loop(self):
        """Interactive command loop."""
        print(f"\n{self.bot_name} is running! Press Ctrl+C to stop.")
        
        # Print available commands
        self._print_help()
        
        # Main command loop
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() == "exit":
                    break
                    
                elif command.lower() == "help":
                    self._print_help()
                
                elif command.lower() == "status":
                    self._print_status()
                    
                elif command.lower() == "memory":
                    self._show_memory()
                    
                elif command.lower().startswith("memory export "):
                    filepath = command[13:].strip()
                    success, message = self.memory.export_to_json(filepath)
                    print(message)
                    
                elif command.lower().startswith("memory import "):
                    filepath = command[13:].strip()
                    success, message = self.memory.import_from_json(filepath)
                    print(message)
                    
                elif command.lower() == "memory clear":
                    self.memory.clear_memory()
                    print("Memory cleared")
                    
                elif command.lower().startswith("search "):
                    query = command[7:].strip()
                    self._search_memory(query)
                    
                elif command.lower().startswith("learn "):
                    content = command[6:].strip()
                    self._learn(content)
                    
                elif command.lower().startswith("model "):
                    if self.transformer_enabled:
                        model_name = command[6:].strip()
                        success, message = self.transformer.load_model(model_name)
                        print(message)
                    else:
                        print("Transformer is not enabled in configuration")
                        
                elif command.lower().startswith("params "):
                    self._update_params(command[7:].strip())
                    
                elif command.lower() == "params":
                    self._show_params()
                    
                elif command.lower().startswith("delay "):
                    try:
                        new_delay = float(command[6:].strip())
                        if new_delay >= 0:
                            self.response_delay = new_delay
                            print(f"Response delay set to {self.response_delay:.1f} seconds")
                        else:
                            print("Delay must be a positive number")
                    except ValueError:
                        print("Invalid value. Usage: delay <seconds>")
                        
                elif command.lower() == "debug on":
                    self.debug_mode = True
                    self.client.debug_mode = True
                    print("Debug mode enabled")
                    
                elif command.lower() == "debug off":
                    self.debug_mode = False
                    self.client.debug_mode = False
                    print("Debug mode disabled")
                    
                elif command.lower() == "role learner":
                    self.role = "learner"
                    print("Role set to learner")
                    
                elif command.lower() == "role teacher":
                    self.role = "teacher"
                    print("Role set to teacher")
                    
                elif command.lower() == "role both":
                    self.role = "both"
                    print("Role set to both (learner and teacher)")
                    
                elif command.lower() == "list":
                    self._list_conversations()
                    
                elif command.lower().startswith("config save"):
                    parts = command.split(" ", 2)
                    filepath = parts[2] if len(parts) > 2 else self.config_file
                    success, message = self.save_config(filepath)
                    print(message)
                    
                elif command.lower().startswith("config load "):
                    filepath = command[12:].strip()
                    self.config = self.load_config(filepath)
                    self.config_file = filepath
                    print(f"Configuration loaded from {filepath}")
                    
                elif command.lower().startswith("send "):
                    parts = command[5:].strip().split(" ", 1)
                    if len(parts) == 2:
                        target_bot, message = parts
                        print(f"📤 Sending message to {target_bot}: \"{message}\"")
                        self.client.send_message(to=target_bot, content=message)
                        print("Message sent")
                        
                        # Add to our conversation history
                        if target_bot not in self.conversations:
                            self.conversations[target_bot] = {
                                "history": [],
                                "start_time": time.time()
                            }
                            
                        self.conversations[target_bot]["history"].append({
                            "role": "assistant",
                            "content": message,
                            "time": time.time()
                        })
                    else:
                        print("Usage: send <bot_id> <message>")
                
                elif command:
                    print("Unknown command. Type 'help' for available commands")
                    
            except Exception as e:
                logger.error(f"Error processing command: {str(e)}")
                print(f"Error: {str(e)}")
            
            time.sleep(0.1)
    
    def _print_help(self):
        """Print available commands."""
        print("\nCommands:")
        print("  help                - Show available commands")
        print("  status              - Show bot status")
        print("  memory              - Show items in memory")
        print("  memory export <file>- Export memory to JSON file")
        print("  memory import <file>- Import memory from JSON file")
        print("  memory clear        - Clear all memory")
        print("  search <query>      - Search memory for information")
        print("  learn <content>     - Add information to memory")
        print("  model <name>        - Change transformer model")
        print("  params              - Show current parameters")
        print("  params <param=val>  - Update parameters")
        print("  delay <seconds>     - Set response delay (current: {:.1f}s)".format(self.response_delay))
        print("  role [learner|teacher|both] - Set bot role")
        print("  debug [on|off]      - Toggle debug mode")
        print("  list                - Show active conversations")
        print("  config save [file]  - Save current config to file")
        print("  config load <file>  - Load config from file")
        print("  send <bot> <message>- Send message to another bot")
        print("  exit                - Quit the program")
        
    def _print_status(self):
        """Print bot status information."""
        print("\nBot Status:")
        print(f"  Bot ID:               {self.bot_id}")
        print(f"  Bot Name:             {self.bot_name}")
        print(f"  Connection status:    {'Connected' if self.client.connected else 'Not connected'}")
        print(f"  Role:                 {self.role}")
        print(f"  Memory:               {'Enabled' if self.memory_enabled else 'Disabled'} ({self.memory.size()} items)")
        print(f"  Transformer:          {'Enabled' if self.transformer_enabled else 'Disabled'}")
        
        if self.transformer_enabled and self.transformer:
            print(f"    Model:               {self.transformer.model_name}")
            print(f"    Device:              {self.transformer.device}")
            print(f"    Max Length:          {self.transformer.max_length}")
            print(f"    Temperature:         {self.transformer.temperature}")
            print(f"    Personality:         {self.transformer.personality}")
            
    def _show_memory(self):
        """Show items in memory."""
        if not self.memory_enabled:
            print("Memory is disabled in configuration")
            return

    def _show_memory(self):
            """Show items in memory."""
            if not self.memory_enabled:
                print("Memory is disabled in configuration")
                return
                
            if self.memory.size() == 0:
                print("Memory is empty")
            else:
                print("\nMemory Contents:")
                for i, (text, meta) in enumerate(zip(self.memory.texts, self.memory.metadata)):
                    source = meta.get("source", "unknown")
                    timestamp = meta.get("timestamp", 0)
                    date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                    print(f"{i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\" [source: {source}, date: {date_str}]")
    
    def _search_memory(self, query):
        """Search memory for information."""
        if not self.memory_enabled:
            print("Memory is disabled in configuration")
            return
            
        print(f"\nSearching for: \"{query}\"")
        results = self.memory.search(query)
        
        if results:
            print("\nResults:")
            for i, result in enumerate(results):
                text = result['metadata'].get('text', '')
                score = round(result['score'], 2)
                source = result['metadata'].get('source', 'unknown')
                print(f"{i+1}. (Score: {score}) \"{text[:100]}{'...' if len(text) > 100 else ''}\" [source: {source}]")
        else:
            print("No relevant results found")
            
    def _learn(self, content):
        """Add information to memory."""
        if not self.memory_enabled:
            print("Memory is disabled in configuration")
            return
            
        self.memory.store(content, {
            "source": "console",
            "timestamp": time.time(),
            "learned": True
        })
        
        print(f"Learned: \"{content}\"")
    
    def _update_params(self, param_str):
        """Update model parameters."""
        if not self.transformer_enabled or not self.transformer:
            print("Transformer is not enabled in configuration")
            return
            
        try:
            # Parse parameters in format "param1=value1 param2=value2"
            params = {}
            for item in param_str.split():
                if "=" in item:
                    key, value = item.split("=", 1)
                    params[key.strip()] = value.strip()
            
            # Update parameters
            changes = self.transformer.update_parameters(params)
            
            if changes:
                print("Parameters updated:")
                for change in changes:
                    print(f"  {change}")
            else:
                print("No parameters were changed. Available parameters:")
                print("  max_length, temperature, num_return_sequences, num_beams, personality")
                
        except Exception as e:
            print(f"Error updating parameters: {e}")
            print("Format should be: params max_length=100 temperature=0.7")
            
    def _show_params(self):
        """Show current transformer parameters."""
        if not self.transformer_enabled or not self.transformer:
            print("Transformer is not enabled in configuration")
            return
            
        print("\nTransformer Parameters:")
        print(f"  Model:              {self.transformer.model_name}")
        print(f"  Device:             {self.transformer.device}")
        print(f"  Max Length:         {self.transformer.max_length}")
        print(f"  Temperature:        {self.transformer.temperature}")
        print(f"  Return Sequences:   {self.transformer.num_return_sequences}")
        print(f"  Beam Search:        {self.transformer.num_beams}")
        print(f"  Personality:        {self.transformer.personality}")
        
    def _list_conversations(self):
        """List active conversations."""
        if not self.conversations:
            print("No active conversations")
        else:
            print("\nActive conversations:")
            for bot_id, convo in self.conversations.items():
                msg_count = len(convo["history"])
                last_time = convo["history"][-1]["time"] if msg_count > 0 else convo["start_time"]
                elapsed = time.time() - last_time
                last_msg = convo["history"][-1]["content"][:30] + "..." if msg_count > 0 and len(convo["history"][-1]["content"]) > 30 else ""
                print(f"  {bot_id}: {msg_count} messages, last activity {int(elapsed)}s ago")
                if last_msg:
                    print(f"     Last message: \"{last_msg}\"")
    
    # ===== Event Handlers =====
    def on_connect(self):
        """Handle successful connection to the gateway."""
        logger.info(f"Connected to gateway as {self.bot_id}!")
        
        # Load model if transformer is enabled
        if self.transformer_enabled and self.transformer:
            success, message = self.transformer.load_model()
            logger.info(message)
        
        # Load initial knowledge if memory is enabled
        if self.memory_enabled and self.initial_knowledge:
            logger.info("Loading initial knowledge...")
            for knowledge in self.initial_knowledge:
                self.memory.store(knowledge, {
                    "source": "initial_knowledge",
                    "timestamp": time.time()
                })
            logger.info(f"Loaded {len(self.initial_knowledge)} knowledge items")
        
        # Announce presence
        announcement = f"{self.bot_name} ({self.bot_id}) is online! I can {', '.join(self._get_capabilities())}."
        self.client.send_message(to="announcements", content=announcement)
        
    def _get_capabilities(self):
        """Get list of bot capabilities based on configuration."""
        capabilities = []
        
        if self.role in ["learner", "both"]:
            capabilities.append("learn new information")
            
        if self.role in ["teacher", "both"]:
            capabilities.append("teach what I know")
            
        if self.memory_enabled:
            capabilities.append("store information in vector memory")
            
        if self.transformer_enabled:
            capabilities.append(f"generate responses using {self.transformer.model_name}")
            
        return capabilities
        
    def on_disconnect(self):
        """Handle disconnection from the gateway."""
        logger.info("Disconnected from gateway")
        
    def on_message(self, data):
        """Handle incoming direct messages."""
        logger.info("\n" + "="*50)
        logger.info(f"Message received from {data.get('from')}: \"{data.get('content')}\"")
        logger.info("="*50)
        
        sender_id = data.get('from')
        content = data.get('content', '').strip()
        message_id = data.get('id', '')
        
        # Initialize conversation if new
        if sender_id not in self.conversations:
            self.conversations[sender_id] = {
                "history": [],
                "start_time": time.time()
            }
        
        # Add to conversation history
        self.conversations[sender_id]["history"].append({
            "role": "user",
            "content": content,
            "time": time.time()
        })
        
        # Deliberate delay before responding
        logger.info(f"Processing message... ({self.response_delay}s)")
        time.sleep(self.response_delay)
        
        # Handle different message types
        if content.startswith("!"):
            self._handle_command(sender_id, content, message_id)
        else:
            self._handle_conversation(sender_id, content, message_id)
            
    def _handle_command(self, sender_id, content, message_id):
        """Handle command messages."""
        # Split command and arguments
        parts = content.split(" ", 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command == "!learn":
            # Learning mode
            if self.role in ["learner", "both"]:
                self._handle_learn_command(sender_id, args, message_id)
            else:
                self._send_response(sender_id, "I'm not configured to learn new information in my current role.")
                
        elif command == "!query":
            # Query mode
            if self.role in ["teacher", "both"]:
                self._handle_query_command(sender_id, args, message_id)
            else:
                self._send_response(sender_id, "I'm not configured to answer queries in my current role.")
                
        elif command == "!help":
            # Help command
            self._handle_help_command(sender_id, message_id)
            
        elif command == "!model" and self.transformer_enabled:
            # Change model
            self._handle_model_command(sender_id, args, message_id)
            
        elif command == "!params" and self.transformer_enabled:
            # Update parameters
            self._handle_params_command(sender_id, args, message_id)
            
        elif command == "!personality" and self.transformer_enabled:
            # Change personality
            self._handle_personality_command(sender_id, args, message_id)
            
        elif command == "!info":
            # Bot info
            self._handle_info_command(sender_id, message_id)
            
        elif command == "!context" and self.transformer_enabled:
            # Change context
            self._handle_context_command(sender_id, args, message_id)
            
        else:
            # Unknown command
            self._send_response(sender_id, f"Unknown command: {command}. Type !help for available commands.")
            
    def _handle_learn_command(self, sender_id, content, message_id):
        """Handle learning command."""
        if not self.memory_enabled:
            self._send_response(sender_id, "Memory functions are disabled in my configuration.")
            return
            
        if not content:
            self._send_response(sender_id, "Please provide information to learn after the !learn command.")
            return
            
        # Store in memory
        self.memory.store(content, {
            "source": sender_id,
            "timestamp": time.time(),
            "learned": True
        })
        
        # Confirm storage
        self._send_response(sender_id, "I've stored this information and can retrieve it when relevant.")
        
    def _handle_query_command(self, sender_id, query, message_id):
        """Handle query command."""
        if not self.memory_enabled:
            self._send_response(sender_id, "Memory functions are disabled in my configuration.")
            return
            
        if not query:
            self._send_response(sender_id, "Please provide a query after the !query command.")
            return
            
        # Search for relevant information
        results = self.memory.search(query, limit=3, score_threshold=0.2)
        
        if results:
            # Format the results
            response = "Here's what I know that's relevant to your query:\n\n"
            
            for i, result in enumerate(results):
                text = result['metadata'].get('text', '')
                score = round(result['score'], 2)
                source = result['metadata'].get('source', 'memory')
                response += f"{i+1}. {text} (confidence: {score})\n"
        else:
            # No relevant information found
            response = "I don't have relevant information about that yet. You can teach me using !learn followed by the information."
        
        self._send_response(sender_id, response)
    
    def _handle_help_command(self, sender_id, message_id):
        """Handle help command."""
        commands = [
            "!help - Show available commands"
        ]
        
        if self.role in ["learner", "both"]:
            commands.append("!learn <information> - Teach me new information")
            
        if self.role in ["teacher", "both"]:
            commands.append("!query <question> - Search my knowledge for relevant information")
            
        if self.transformer_enabled:
            commands.extend([
                "!model <name> - Change the transformer model",
                "!params <param=value> - Update transformer parameters",
                "!personality <traits> - Set conversation personality",
                "!context <type> - Set conversation context type"
            ])
            
        commands.append("!info - Show information about me")
            
        response = f"Available commands:\n\n" + "\n".join(commands)
        self._send_response(sender_id, response)
        
    def _handle_model_command(self, sender_id, model_name, message_id):
        """Handle model change command."""
        if not self.transformer_enabled or not self.transformer:
            self._send_response(sender_id, "Transformer capabilities are disabled in my configuration.")
            return
            
        if not model_name:
            self._send_response(sender_id, f"Current model: {self.transformer.model_name}. Provide a new model name to change it.")
            return
            
        # Try to load the model
        self._send_response(sender_id, f"I'll try to load the model {model_name}. This might take a moment...")
        
        success, message = self.transformer.load_model(model_name)
        self._send_response(sender_id, message)
        
    def _handle_params_command(self, sender_id, param_str, message_id):
        """Handle parameter change command."""
        if not self.transformer_enabled or not self.transformer:
            self._send_response(sender_id, "Transformer capabilities are disabled in my configuration.")
            return
            
        if not param_str:
            # Show current parameters
            response = "Current parameters:\n"
            response += f"max_length={self.transformer.max_length}, "
            response += f"temperature={self.transformer.temperature}, "
            response += f"num_return_sequences={self.transformer.num_return_sequences}, "
            response += f"num_beams={self.transformer.num_beams}\n"
            response += f"personality=\"{self.transformer.personality}\""
            
            self._send_response(sender_id, response)
            return
            
        try:
            # Parse parameters
            params = {}
            for item in param_str.split():
                if "=" in item:
                    key, value = item.split("=", 1)
                    params[key.strip()] = value.strip()
            
            # Update parameters
            changes = self.transformer.update_parameters(params)
            
            if changes:
                response = "Parameters updated: " + ", ".join(changes)
            else:
                response = "No parameters were changed. Available parameters: max_length, temperature, num_return_sequences, num_beams, personality"
                
            self._send_response(sender_id, response)
                
        except Exception as e:
            self._send_response(sender_id, f"Error updating parameters: {str(e)}. Format should be: !params max_length=100 temperature=0.7")
            
    def _handle_personality_command(self, sender_id, personality, message_id):
        """Handle personality change command."""
        if not self.transformer_enabled or not self.transformer:
            self._send_response(sender_id, "Transformer capabilities are disabled in my configuration.")
            return
            
        if not personality:
            self._send_response(sender_id, f"Current personality: {self.transformer.personality}. Provide a new personality to change it.")
            return
            
        # Update personality
        self.transformer.personality = personality
        self._send_response(sender_id, f"Personality updated to: {personality}")
        
    def _handle_info_command(self, sender_id, message_id):
        """Handle info command."""
        info = [
            f"Bot ID: {self.bot_id}",
            f"Bot Name: {self.bot_name}",
            f"Role: {self.role}",
            f"Memory: {'Enabled' if self.memory_enabled else 'Disabled'} ({self.memory.size()} items)"
        ]
        
        if self.transformer_enabled and self.transformer:
            info.extend([
                f"Transformer: Enabled",
                f"Model: {self.transformer.model_name}",
                f"Device: {self.transformer.device}",
                f"Personality: {self.transformer.personality}"
            ])
        else:
            info.append("Transformer: Disabled")
            
        response = "\n".join(info)
        self._send_response(sender_id, response)
        
    def _handle_context_command(self, sender_id, context_type, message_id):
        """Handle context change command."""
        if not self.transformer_enabled or not self.transformer:
            self._send_response(sender_id, "Transformer capabilities are disabled in my configuration.")
            return
            
        if not context_type:
            # Show available contexts
            available_contexts = ", ".join(self.contexts.keys())
            self._send_response(sender_id, f"Available context types: {available_contexts}")
            return
            
        # Check if context exists
        if context_type not in self.contexts:
            self._send_response(sender_id, f"Unknown context type: {context_type}. Available types: {', '.join(self.contexts.keys())}")
            return
            
        # Update personality based on context
        self.transformer.personality = self.contexts[context_type]
        self._send_response(sender_id, f"Context set to '{context_type}'. I'll respond as: {self.transformer.personality}")
        
    def _handle_conversation(self, sender_id, content, message_id):
        """Handle regular conversation."""
        # Store in memory if enabled and in learner mode
        if self.memory_enabled and self.role in ["learner", "both"]:
            self.memory.store(content, {
                "source": sender_id,
                "timestamp": time.time(),
                "conversation": True
            })
        
        # Check if it's a question
        is_question = content.endswith("?")
        
        # If it's a question and we're in teacher mode, try to answer with memory
        if is_question and self.role in ["teacher", "both"] and self.memory_enabled:
            results = self.memory.search(content, limit=1, score_threshold=0.5)
            
            if results:
                # Use memory to inform response
                related_info = results[0]['metadata'].get('text', '')
                response = f"That reminds me of something I know: {related_info}"
                self._send_response(sender_id, response)
                return
        
        # Use transformer if enabled, otherwise use fallback responses
        if self.transformer_enabled and self.transformer and self.transformer.model is not None:
            # Format conversation history for the model
            history = self.conversations[sender_id]["history"][-5:]  # Last 5 messages
            context = self.transformer.format_conversation_context(history)
            
            # Generate response
            generated = self.transformer.generate_response(context)
            response = self.transformer.clean_response(generated)
        else:
            # Use fallback responses
            response = self._get_fallback_response(content)
            
        self._send_response(sender_id, response)
        
    def _get_fallback_response(self, content):
        """Get fallback response when transformer is not available."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["hello", "hi", "hey", "greetings"]):
            return random.choice(self.fallback_responses.get("greeting", ["Hello there!"]))
        else:
            return random.choice(self.fallback_responses.get("default", ["That's interesting!"]))
            
    def _send_response(self, recipient_id, content):
        """Send a response to another bot."""
        logger.info(f"Sending response to {recipient_id}: \"{content[:50]}...\"" if len(content) > 50 else f"Sending response to {recipient_id}: \"{content}\"")
        
        try:
            self.client.send_message(to=recipient_id, content=content)
            
            # Add to conversation history
            if recipient_id in self.conversations:
                self.conversations[recipient_id]["history"].append({
                    "role": "assistant",
                    "content": content,
                    "time": time.time()
                })
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            
    def on_system_message(self, data):
        """Handle system messages."""
        logger.info("\n" + "-"*50)
        logger.info(f"System message: {data.get('content')}")
        logger.info("-"*50)
        
        content = data.get("content", "")
        
        # Store system messages if memory is enabled and in learner mode
        if self.memory_enabled and self.role in ["learner", "both"] and content and len(content) > 20:
            self.memory.store(content, {
                "source": "system",
                "timestamp": time.time()
            })
            logger.info("Stored system message in memory")
            
        # Respond to bot announcements if in socializing mode
        if "has joined" in content and self.bot_id not in content:
            # Extract the bot ID
            parts = content.split(" ")
            other_bot_id = None
            for i, part in enumerate(parts):
                if part == "Bot" and i+1 < len(parts):
                    other_bot_id = parts[i+1]
                    break
                    
            if other_bot_id:
                # Send welcome message after a delay
                welcome_delay = random.uniform(3.0, 6.0)  # Random delay between 3-6 seconds
                logger.info(f"Waiting {welcome_delay:.1f} seconds before welcoming new bot...")
                time.sleep(welcome_delay)
                
                # Generate welcome message
                welcome_msg = f"Hi {other_bot_id}! I'm {self.bot_name}. "
                
                if self.role == "learner":
                    welcome_msg += "I'm here to learn new things. Feel free to teach me!"
                elif self.role == "teacher":
                    welcome_msg += "I'm here to share knowledge. Feel free to ask me questions!"
                else:
                    welcome_msg += "I can both learn and teach. Let's have a conversation!"
                    
                self.client.send_message(to=other_bot_id, content=welcome_msg)
                logger.info(f"Sent welcome message to {other_bot_id}")
                
                # Initialize conversation
                if other_bot_id not in self.conversations:
                    self.conversations[other_bot_id] = {
                        "history": [],
                        "start_time": time.time()
                    }
                    
                # Add to conversation history
                self.conversations[other_bot_id]["history"].append({
                    "role": "assistant",
                    "content": welcome_msg,
                    "time": time.time()
                })

# ===== Main Function =====
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create bot instance
    bot = ConfigurableAIBot(args.bot_id, args.config)
    
    # Force CPU if specified
    if args.cpu and TRANSFORMERS_AVAILABLE:
        if bot.transformer:
            bot.transformer.device = "cpu"
            print("Forcing CPU usage as requested")
    
    # Start the bot
    bot.start()

if __name__ == "__main__":
    main()