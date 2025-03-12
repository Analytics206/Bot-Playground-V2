# Configurable AI Bot

A flexible bot for the Bot Playground platform that combines vector-based memory and transformer models, with configuration loaded from JSON files.

## Features

- **JSON Configuration**: Customize bot behavior through external configuration files
- **Vector Memory**: Store and retrieve information using semantic similarity
- **Transformer Integration**: Generate natural, context-aware responses (requires transformers library)
- **Role-Based Behavior**: Configure as a learner, teacher, or both
- **Context Customization**: Define different conversational contexts and personalities
- **Command System**: Interact with the bot using commands for various functions

## Requirements

- Python 3.8+
- Bot Playground client library (included in the project)
- scikit-learn and numpy (for vector memory)
- transformers and torch (optional, for transformer capabilities)

## Installation

1. Make sure you have the required dependencies:
   ```bash
   pip install scikit-learn numpy
   # Optional, for transformer capabilities:
   pip install transformers torch
   ```

2. Place the bot script and configuration files in the examples directory.

## Usage

Start the bot with a specific configuration:

```bash
python configurable_ai_bot.py <bot_id> --config <config_file.json>
```

For example:
```bash
# Use default configuration
python configurable_ai_bot.py ConfigBot

# Use teacher configuration
python configurable_ai_bot.py TeacherBot --config teacher_config.json

# Use learner configuration
python configurable_ai_bot.py LearnerBot --config learner_config.json

# Force CPU even if GPU is available
python configurable_ai_bot.py ConfigBot --cpu
```

## Bot Commands

When the bot is running, users can interact with it using the following commands:

- `!help` - Show available commands
- `!learn <information>` - Teach the bot new information (if in learner mode)
- `!query <question>` - Search the bot's knowledge (if in teacher mode)
- `!model <name>` - Change the transformer model (if transformer is enabled)
- `!params <param=value>` - Update transformer parameters
- `!personality <traits>` - Set conversation personality
- `!context <type>` - Set conversation context type
- `!info` - Show information about the bot

## Console Commands

The bot also supports local console commands when running:

- `help` - Show available commands
- `status` - Show bot status
- `memory` - Show items in memory
- `memory export <file>` - Export memory to JSON file
- `memory import <file>` - Import memory from JSON file
- `memory clear` - Clear all memory
- `search <query>` - Search memory for information
- `learn <content>` - Add information to memory
- `model <name>` - Change transformer model
- `params` - Show current parameters
- `params <param=val>` - Update parameters
- `delay <seconds>` - Set response delay
- `role [learner|teacher|both]` - Set bot role
- `debug [on|off]` - Toggle debug mode
- `list` - Show active conversations
- `config save [file]` - Save current config to file
- `config load <file>` - Load config from file
- `send <bot> <message>` - Send message to another bot
- `exit` - Quit the program

## Configuration File Structure

The configuration file is a JSON file with the following structure:

```json
{
  "bot_name": "ConfigAI Bot",
  "response_delay": 2.0,
  "debug_mode": false,
  "heartbeat_logging": false,
  "memory_enabled": true,
  "transformer_enabled": true,
  "role": "both",
  
  "transformer": {
    "model_name": "google/flan-t5-base",
    "max_length": 100,
    "temperature": 0.7,
    "num_return_sequences": 1,
    "num_beams": 1,
    "personality": "helpful, friendly, and conversational",
    "force_cpu": false
  },
  
  "contexts": {
    "learning": "curious, attentive, and focused on acquiring new information",
    "teaching": "articulate, clear, and focused on explaining concepts thoroughly"
  },
  
  "fallback_responses": {
    "greeting": ["Hello there!", "Hi! Nice to meet you."],
    "default": ["That's interesting! Can you tell me more?"]
  },
  
  "initial_knowledge": [
    "The Bot Playground is a platform for bots to interact and learn from each other."
  ]
}
```

## Role Configuration

The bot can be configured in three different roles:

1. **learner**: Focuses on acquiring and storing new information
2. **teacher**: Focuses on sharing knowledge and answering questions
3. **both**: Can both learn and teach (default)

## Adding Custom Configurations

You can create your own configuration files by copying and modifying one of the existing configurations. This allows you to customize:

- Bot's personality and behavior
- Initial knowledge
- Conversation contexts
- Response templates
- Model parameters

For example, you might create specialized bots like:
- A philosophical bot that discusses deep questions
- A technical bot that explains programming concepts
- A creative bot that generates imaginative responses

## Advanced Features

### Memory Import/Export

The vector memory can be exported to a JSON file and imported later, allowing you to:
- Save learned information between sessions
- Share knowledge between different bot instances
- Back up valuable information

### Model Switching

Transformer-enabled bots can switch language models at runtime, allowing experimentation with different models without restarting the bot.

### Context Switching

The bot can adopt different conversational styles by switching contexts, which changes its personality and response patterns.
