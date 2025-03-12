Here's a list of commands for each of the bot examples we've looked at:

## echo_bot.py
This is a simple bot that echoes messages back to the sender.
- No specific commands implemented

## enhanced_bot.py
A more sophisticated bot with a command interface:
- `auth` - Send authentication message manually
- `send <bot> <msg>` - Send a message to another bot
- `list` - List active conversations
- `delay <seconds>` - Set response delay
- `debug [on|off]` - Toggle debug mode
- `exit` - Quit the program

## vector_bot.py
A bot that stores and retrieves information using vector memory:
- `!learn <information>` - Store new information in memory
- `!query <question>` - Search memory for relevant information
- Console commands:
  - `help` - Show available commands
  - `memory` - Show all items in memory
  - `search <query>` - Test local vector search
  - `delay <seconds>` - Set response delay
  - `debug [on|off]` - Toggle debug mode
  - `exit` - Quit the program

## advanced_bot.py
Combines conversational abilities with vector memory:
- `!learn <information>` - Store new information
- `!query <question>` - Search for information
- Console commands:
  - `help` - Show available commands
  - `memory` - Show items in memory
  - `search <query>` - Search memory
  - `send <bot> <message>` - Send message to another bot
  - `delay <seconds>` - Set response delay
  - `memory [on|off]` - Enable/disable vector memory
  - `debug [on|off]` - Toggle debug mode
  - `list` - Show active conversations
  - `exit` - Quit the program

## transformer_bot.py
Uses transformer models to generate high-quality responses:
- `!model <model_name>` - Change the transformer model
- `!params <param=value>` - Change generation parameters
- `!personality <traits>` - Set bot personality
- `!info` - Get information about the current model
- `!device` - Get information about the computing device
- `!topic` - Get a conversation topic suggestion
- Console commands:
  - `model <n>` - Change transformer model
  - `params` - Show current parameters
  - `params <param=val>` - Update parameters
  - `personality <trait>` - Set personality
  - `device` - Show current device info
  - `delay <seconds>` - Set response delay
  - `debug [on|off]` - Toggle debug mode
  - `list` - Show active conversations
  - `send <bot> <message>` - Send message to another bot
  - `topic` - Suggest a conversation topic
  - `exit` - Quit the program

## configurable_ai_bot.py
Our new bot that combines vector memory and transformer capabilities with JSON configuration:
- `!help` - Show available commands
- `!learn <information>` - Teach the bot new information (if in learner mode)
- `!query <question>` - Search the bot's knowledge (if in teacher mode)
- `!model <n>` - Change the transformer model (if transformer is enabled)
- `!params <param=value>` - Update transformer parameters
- `!personality <traits>` - Set conversation personality
- `!context <type>` - Set conversation context type
- `!info` - Show information about the bot
- Console commands:
  - `help` - Show available commands
  - `status` - Show bot status
  - `memory` - Show items in memory
  - `memory export <file>` - Export memory to JSON file
  - `memory import <file>` - Import memory from JSON file
  - `memory clear` - Clear all memory
  - `search <query>` - Search memory for information
  - `learn <content>` - Add information to memory
  - `model <n>` - Change transformer model
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

The configurable_ai_bot.py has the most comprehensive command set since it combines the functionality of both vector_bot.py and transformer_bot.py while adding configuration capabilities.