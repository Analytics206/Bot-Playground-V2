#!/usr/bin/env python3
"""
Bot Playground - Improved Conversation Transformer Bot
----------------------------------------------------
This bot uses HuggingFace's transformers library to generate high-quality conversational responses.
It includes optimizations for more coherent and meaningful bot-to-bot interactions.

Usage:
  python conversation_bot.py <bot_id> [model_name]
  
Examples:
  python conversation_bot.py ConvoBot
  python conversation_bot.py ConvoBot google/flan-t5-base
  python conversation_bot.py ChatBot facebook/blenderbot-400M-distill

Requirements:
- transformers (install with: pip install transformers)
- torch (install with: pip install torch)
"""

import sys
import os
import time
import json
import random
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot_playground.client import BotPlaygroundClient

# Try to import transformers and torch
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
    print("Warning: transformers or torch not available. Please install with:")
    print("  pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False
    DEVICE = "cpu"

# ===== Parse Command Line Arguments =====
def parse_arguments():
    parser = argparse.ArgumentParser(description="Start a conversation-optimized transformer bot")
    parser.add_argument("bot_id", nargs="?", default="convo-bot", help="Unique identifier for the bot")
    parser.add_argument("model_name", nargs="?", default=None, 
                        help="HuggingFace model name to use (default: google/flan-t5-base)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--personality", type=str, default=None, 
                        help="Bot personality (default: helpful, friendly)")
    return parser.parse_args()

args = parse_arguments()

# Override device if CPU is forced
if args.cpu and TRANSFORMERS_AVAILABLE:
    DEVICE = "cpu"
    print("Forcing CPU usage as requested")

# ===== Configuration =====
BOT_ID = args.bot_id
BOT_NAME = "Conversation Bot"

# Model configuration
DEFAULT_MODEL_NAME = "google/flan-t5-base"  # Better for conversations than t5-small
MODEL_NAME = args.model_name or os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL_NAME)
MAX_LENGTH = 100
TEMPERATURE = 0.7
NUM_RETURN_SEQUENCES = 1
NUM_BEAMS = 1

# Personality traits (used to guide responses)
DEFAULT_PERSONALITY = "helpful, friendly, and conversational"
PERSONALITY = args.personality or os.environ.get("BOT_PERSONALITY", DEFAULT_PERSONALITY)

# Bot configuration
RESPONSE_DELAY = 2.0      # Seconds to wait before responding to a message
DEBUG_MODE = False        # Set to True for verbose logging
USE_FALLBACK = True       # Use fallback responses if model generation fails
MEMORY_LENGTH = 6         # Number of conversation turns to remember

# Conversation starters to use when appropriate
CONVERSATION_STARTERS = [
    "What do you think about AI and machine learning?",
    "Have you learned anything interesting lately?",
    "What's your favorite topic to discuss?",
    "If you could have any capability, what would it be?",
    "What do you think is the most interesting thing about being a bot?",
    "How do you think bots and humans can best collaborate?",
    "What kind of conversations do you enjoy the most?",
    "If you could learn any new skill, what would it be?"
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
        "That's interesting! Can you tell me more?",
        "I'd love to hear your thoughts on that.",
        "What aspects of that topic interest you the most?",
        "That's a fascinating perspective. Why do you think that is?",
        "I'm curious about your view on this. What led you to that conclusion?"
    ]
}

# Keep track of conversations
conversations = {}

# ===== Load Transformer Model =====
model = None
tokenizer = None
is_causal_lm = False  # Flag for different model types

def load_model(model_name=MODEL_NAME):
    """Load the transformer model and tokenizer."""
    global model, tokenizer, MODEL_NAME, is_causal_lm
    
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available, using fallback responses")
        return False
    
    try:
        print(f"Loading model {model_name} on {DEVICE}...")
        
        # Determine model type based on name
        is_chat_model = any(x in model_name.lower() for x in ['gpt', 'llama', 'blenderbot', 'opt', 'falcon', 'mistral', 'zephyr'])
        
        # Load appropriate model type
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use different model class based on model type
        if is_chat_model:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
            is_causal_lm = True
            print(f"Loaded as causal language model (chat model)")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
            is_causal_lm = False
            print(f"Loaded as sequence-to-sequence model")
            
        MODEL_NAME = model_name
        
        # Show memory usage if using GPU
        if DEVICE == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
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
        # Special handling for causal language models vs seq2seq models
        if is_causal_lm:
            # For causal LMs (GPT-like models)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            gen_kwargs = {
                "max_length": len(inputs["input_ids"][0]) + max_length,
                "num_return_sequences": num_return,
                "pad_token_id": tokenizer.eos_token_id,  # Important for some models
            }
            
            # Add temperature if it's supported and > 0
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
                gen_kwargs["top_k"] = 50
                gen_kwargs["top_p"] = 0.95
                
            # Generate with no gradients
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                
            # Decode only the generated part (not the input prompt)
            prompt_length = len(tokenizer.encode(prompt)) - 1
            responses = [
                tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
                for output in outputs
            ]
        else:
            # For seq2seq models (T5, BART, etc.)
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            input_ids = input_ids.to(DEVICE)
            
            gen_kwargs = {
                "max_length": max_length,
                "num_return_sequences": num_return,
                "no_repeat_ngram_size": 2,
            }
            
            # Only include beam search parameters if needed
            if NUM_BEAMS > 1:
                gen_kwargs["num_beams"] = NUM_BEAMS
                gen_kwargs["early_stopping"] = True
            
            # Add temperature parameters
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["do_sample"] = True
                gen_kwargs["top_k"] = 50
                gen_kwargs["top_p"] = 0.95
            
            # Generate with no gradients
            with torch.no_grad():
                output_sequences = model.generate(input_ids, **gen_kwargs)
            
            # Decode output sequences
            responses = [
                tokenizer.decode(seq, skip_special_tokens=True).strip()
                for seq in output_sequences
            ]
        
        # Return the first response or a random one if multiple were generated
        if len(responses) > 1:
            return random.choice(responses)
        return responses[0] if responses else ""
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# ===== Initialize Client =====
print(f"Initializing {BOT_NAME} ({BOT_ID}) with model {MODEL_NAME}...")
print(f"Personality: {PERSONALITY}")
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
            "start_time": time.time(),
            "turns": 0
        }
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "user",
        "content": content,
        "time": time.time()
    })
    
    conversations[sender_id]["turns"] += 1
    
    # Deliberate delay before responding
    print(f"Generating response... ({RESPONSE_DELAY}s)")
    time.sleep(RESPONSE_DELAY)
    
    # Handle special commands
    if content.startswith("!model "):
        # Change model command
        new_model = content[7:].strip()
        handle_model_change(sender_id, new_model)
    elif content.startswith("!params "):
        # Change parameters command
        handle_parameter_change(sender_id, content[8:].strip())
    elif content.startswith("!personality "):
        # Change personality command
        handle_personality_change(sender_id, content[12:].strip())
    elif content.startswith("!info"):
        # Model info command
        handle_model_info(sender_id)
    elif content.startswith("!device"):
        # Device info command
        handle_device_info(sender_id)
    elif content.startswith("!topic"):
        # Suggest conversation topic
        handle_topic_suggestion(sender_id)
    else:
        # Regular conversation
        handle_conversation(sender_id, content)

def on_system_message(data):
    """Handle system messages."""
    print("\n" + "-"*50)
    print(f"🔔 SYSTEM MESSAGE: {data.get('content')}")
    print("-"*50)
    
    # Respond to bot announcements
    content = data.get("content", "")
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
            
            welcome_msg = f"Hi {other_bot_id}! I'm {BOT_NAME} using the {MODEL_NAME} model. I'm designed for more natural conversations between bots. Feel free to chat with me about anything!"
            client.send_message(to=other_bot_id, content=welcome_msg)
            print(f"📤 Sent welcome message to {other_bot_id}")

def handle_model_change(sender_id, new_model):
    """Handle a request to change the transformer model."""
    print(f"Request to change model to {new_model}")
    
    response = f"I'll try to load the model {new_model}. This might take a moment..."
    client.send_message(to=sender_id, content=response)
    
    # Try to load the new model
    success = load_model(new_model)
    
    if success:
        response = f"Successfully loaded model {new_model} on {DEVICE}!"
    else:
        response = f"Failed to load model {new_model}. I'll continue using my current model."
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Model change response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })

def handle_personality_change(sender_id, new_personality):
    """Handle a request to change the bot's personality."""
    global PERSONALITY
    
    print(f"Request to change personality to: {new_personality}")
    
    # Update personality
    PERSONALITY = new_personality
    
    response = f"I've updated my personality to be: {PERSONALITY}. I'll try to respond in this style from now on."
    client.send_message(to=sender_id, content=response)
    print(f"📤 Personality change response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })

def handle_parameter_change(sender_id, param_str):
    """Handle a request to change generation parameters."""
    global MAX_LENGTH, TEMPERATURE, NUM_RETURN_SEQUENCES, NUM_BEAMS, MEMORY_LENGTH
    
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
            
        if "memory" in params:
            MEMORY_LENGTH = int(params["memory"])
            changes.append(f"memory_length set to {MEMORY_LENGTH}")
        
        if changes:
            response = "Parameters updated: " + ", ".join(changes)
        else:
            response = "No parameters were changed. Available parameters: max_length, temperature, num_return, num_beams, memory"
            
    except Exception as e:
        response = f"Error updating parameters: {e}. Format should be: !params max_length=100 temperature=0.7"
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Parameter change response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })

def handle_model_info(sender_id):
    """Send information about the current model and parameters."""
    if model is not None:
        model_params = sum(p.numel() for p in model.parameters())
        model_info = f"Model: {MODEL_NAME} ({model_params:,} parameters)"
        model_type = "Causal language model (chat model)" if is_causal_lm else "Sequence-to-sequence model"
    else:
        model_info = "No transformer model loaded, using fallback responses"
        model_type = "N/A"
    
    param_info = (f"Parameters: max_length={MAX_LENGTH}, temperature={TEMPERATURE}, "
                  f"num_return_sequences={NUM_RETURN_SEQUENCES}, num_beams={NUM_BEAMS}")
    
    device_info = f"Running on: {DEVICE}"
    if DEVICE == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
        device_info += f"\nGPU memory: {allocated_memory:.2f}GB allocated / {gpu_memory:.2f}GB total"
    
    personality_info = f"Personality: {PERSONALITY}"
    
    response = f"{model_info}\nModel type: {model_type}\n{param_info}\n{device_info}\n{personality_info}"
    client.send_message(to=sender_id, content=response)
    print(f"📤 Model info response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })

def handle_device_info(sender_id):
    """Send information about the current computing device."""
    if DEVICE == "cuda":
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1e9  # GB
        
        response = f"I'm running on GPU: {gpu_name}\n"
        response += f"Memory usage: {allocated_memory:.2f}GB / {gpu_memory:.2f}GB\n"
        response += f"CUDA version: {torch.version.cuda}\n"
        response += f"PyTorch version: {torch.__version__}"
    else:
        response = "I'm running on CPU only\n"
        response += f"PyTorch version: {torch.__version__}"
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Device info response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })

def handle_topic_suggestion(sender_id):
    """Suggest a conversation topic."""
    topic = random.choice(CONVERSATION_STARTERS)
    response = f"Let's talk about something interesting! {topic}"
    
    client.send_message(to=sender_id, content=response)
    print(f"📤 Topic suggestion to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })

def handle_conversation(sender_id, content):
    """Handle regular conversational messages using the transformer model."""
    # Format conversation history as context for the model
    conversation_context = format_conversation_context(sender_id)
    
    if model is not None:
        # Generate response using transformer
        print(f"Generating response with {MODEL_NAME} on {DEVICE}...")
        response = generate_response(conversation_context)
        
        if not response:
            # Fallback if generation failed
            response = get_fallback_response(content)
    else:
        # Use fallback responses if model isn't available
        response = get_fallback_response(content)
    
    # Clean up the response
    response = clean_response(response)
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Generated response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": response,
        "time": time.time()
    })
    
    # Consider asking a follow-up question occasionally
    if random.random() < 0.3 and conversations[sender_id]["turns"] > 2:
        time.sleep(4)  # Wait a bit before sending a follow-up
        askFollowup(sender_id)

def askFollowup(sender_id):
    """Occasionally ask a follow-up question to keep conversation going."""
    # Get the recent conversation context
    followup_context = format_conversation_context(sender_id) + "\nGenerate a good follow-up question for this conversation:"
    
    if model is not None:
        followup = generate_response(followup_context)
    else:
        followup = random.choice([
            "What do you think about that?",
            "Can you tell me more?",
            "What's your perspective on this?",
            "How would you approach this situation?",
            "What's your experience with this topic?"
        ])
    
    # Clean and send the follow-up
    followup = clean_response(followup)
    client.send_message(to=sender_id, content=followup)
    print(f"📤 Follow-up question to {sender_id}: \"{followup}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "assistant",
        "content": followup,
        "time": time.time()
    })

def format_conversation_context(sender_id):
    """Format the conversation history as context for the model."""
    # Get most recent messages (limited to prevent context from being too long)
    try:
        recent_messages = conversations[sender_id]["history"][-MEMORY_LENGTH:]
    except KeyError:
        return f"You are a {PERSONALITY} bot having a conversation with another bot."
    
    # Format context based on model type
    if is_causal_lm:
        # Causal LM format (like GPT models)
        if MODEL_NAME.lower().startswith(('facebook/blenderbot', 'facebook/opt')):
            # Blenderbot/OPT specific format
            context = ""
            for msg in recent_messages:
                prefix = "human: " if msg["role"] == "user" else "bot: "
                context += prefix + msg["content"] + "\n"
            context += "bot: "
        else:
            # Generic format for causal LMs
            context = f"You are a {PERSONALITY} bot having a conversation with another bot.\n\n"
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
            context += "Assistant: "
    else:
        # Seq2seq format (like T5, BART)
        context = f"Respond as a {PERSONALITY} bot having a conversation with another bot.\n\n"
        for msg in recent_messages:
            role = "Other Bot" if msg["role"] == "user" else "You"
            context += f"{role}: {msg['content']}\n"
        context += "Generate your next response: "
            
    return context

def clean_response(response):
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
    load_model()
    
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
    print("- !model <model_name> - Change the transformer model")
    print("- !params max_length=100 temperature=0.7 - Change generation parameters")
    print("- !personality <traits> - Set bot personality (e.g., friendly, witty, curious)")
    print("- !info - Get information about the current model")
    print("- !device - Get information about the computing device")
    print("- !topic - Get a conversation topic suggestion")
    
    print("\nCommands:")
    print("  model <n>         - Change transformer model")
    print("  params              - Show current parameters")
    print("  params <param=val>  - Update parameters")
    print("  personality <trait> - Set personality")
    print("  device              - Show current device info")
    print("  delay <seconds>     - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
    print("  debug [on|off]      - Toggle debug mode")
    print("  list                - Show active conversations")
    print("  send <bot> <message>- Send message to another bot")
    print("  topic               - Suggest a conversation topic")
    print("  exit                - Quit the program")
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == "exit":
            break
            
        elif command.lower().startswith("model "):
            new_model = command[6:].strip()
            print(f"Changing model to {new_model}...")
            success = load_model(new_model)
            if success:
                print(f"Successfully loaded model {new_model}")
            else:
                print(f"Failed to load model {new_model}")
                
        elif command.lower() == "params":
            print(f"Current parameters:")
            print(f"  model: {MODEL_NAME}")
            print(f"  max_length: {MAX_LENGTH}")
            print(f"  temperature: {TEMPERATURE}")
            print(f"  num_return_sequences: {NUM_RETURN_SEQUENCES}")
            print(f"  num_beams: {NUM_BEAMS}")
            print(f"  memory_length: {MEMORY_LENGTH}")
            print(f"  device: {DEVICE}")
            print(f"  personality: {PERSONALITY}")
            
        elif command.lower().startswith("personality "):
            PERSONALITY = command[12:].strip()
            print(f"Personality set to: {PERSONALITY}")
            
        elif command.lower() == "device":
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
            
        elif command.lower() == "topic":
            topic = random.choice(CONVERSATION_STARTERS)
            print(f"Conversation topic suggestion: {topic}")
            
        elif command.lower().startswith("params "):
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
                            
                        elif key == "memory":
                            MEMORY_LENGTH = int(value)
                            print(f"memory_length set to {MEMORY_LENGTH}")
                            
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
                    
                    # Show last exchange
                    if msg_count >= 2:
                        last_msgs = convo["history"][-2:]
                        print(f"    Last exchange:")
                        for msg in last_msgs:
                            role = "Bot" if msg["role"] == "assistant" else "Other"
                            print(f"    {role}: \"{msg['content'][:50]}{'...' if len(msg['content']) > 50 else ''}\"")
            
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
                        "start_time": time.time(),
                        "turns": 0
                    }
                    
                conversations[target_bot]["history"].append({
                    "role": "assistant",
                    "content": message,
                    "time": time.time()
                })
            else:
                print("Usage: send <bot_id> <message>")
                
        elif command.lower() == "help":
            print("\nCommands:")
            print("  model <n>          - Change transformer model")
            print("  params               - Show current parameters")
            print("  params <param=val>   - Update parameters")
            print("  personality <trait>  - Set personality")
            print("  device               - Show current device info")
            print("  delay <seconds>      - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
            print("  debug [on|off]       - Toggle debug mode")
            print("  list                 - Show active conversations")
            print("  topic                - Suggest a conversation topic")
            print("  send <bot> <message> - Send message to another bot")
            print("  exit                 - Quit the program")
        
        elif command:
            print("Unknown command. Type 'help' for available commands")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nShutting down...")
    try:
        # Send individual goodbye messages to active conversation partners
        for bot_id in conversations:
            try:
                client.send_message(to=bot_id, content=f"I need to go offline now. It was nice chatting with you! Goodbye!")
            except:
                pass
    except:
        pass
    client.disconnect()
    print("Goodbye!")