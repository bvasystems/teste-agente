"""
Example: WhatsApp Bot with Welcome Image

This example demonstrates how to send an image as the first message
in a WhatsApp conversation using the new welcome_image feature.
"""

from agentle.agents.whatsapp.whatsapp_bot import WhatsAppBot
from agentle.agents.whatsapp.models.whatsapp_bot_config import WhatsAppBotConfig
from agentle.agents.whatsapp.providers.evolution.evolution_api_provider import (
    EvolutionAPIProvider,
)
from agentle.agents.whatsapp.providers.evolution.evolution_api_config import (
    EvolutionAPIConfig,
)

# Example 1: Welcome image from URL
config_with_url = WhatsAppBotConfig.production(
    welcome_message="Welcome! 👋 How can I help you today?",
    welcome_image_url="https://example.com/welcome-banner.jpg",
)

# Example 2: Welcome image from base64
# Note: Base64 image requires file_storage_manager to be configured
# Otherwise, it will fall back to sending the text caption only
config_with_base64 = WhatsAppBotConfig.production(
    welcome_message="Welcome to our service!",
    welcome_image_base64="iVBORw0KGgoAAAANSUhEUgAAAAUA...",  # Your base64 image data
)

# Example 3: Welcome image without caption (image only)
config_image_only = WhatsAppBotConfig.production(
    welcome_image_url="https://example.com/banner.png",
    # No welcome_message means the image will be sent without a caption
)

# Example 4: Traditional text-only welcome (backward compatible)
config_text_only = WhatsAppBotConfig.production(
    welcome_message="Hello! I'm your assistant.",
    # No welcome_image_url or welcome_image_base64 means text-only
)

# Setup provider and bot
evolution_config = EvolutionAPIConfig(
    base_url="https://your-evolution-api.com",
    api_key="your-api-key",
    instance_name="your-instance",
)

provider = EvolutionAPIProvider(config=evolution_config)

# Create bot with welcome image
bot = WhatsAppBot(
    agent=your_agent,  # Your configured agent
    provider=provider,
    config=config_with_url,  # Use any of the configs above
    file_storage_manager=your_storage_manager,  # Required for base64 images
)

# The welcome image will be automatically sent when a user
# sends their first message to the bot

"""
How it works:

1. When a user sends their first message:
   - Bot checks if it's the first interaction (conversation history length == 0)
   - If welcome_image_url is set: Sends the image from the URL
   - If welcome_image_base64 is set: Uploads to storage first, then sends
   - welcome_message is used as the image caption (optional)
   - If no image is configured, sends text-only message (backward compatible)

2. Priority:
   - welcome_image_url takes precedence over welcome_image_base64
   - If both image and text are configured, text becomes the caption
   - If only text is configured, sends text-only (backward compatible)

3. Base64 handling:
   - Requires file_storage_manager to be configured
   - Automatically uploads base64 to storage and gets URL
   - Falls back to text-only if upload fails

4. Error handling:
   - If image sending fails, falls back to text caption
   - If no caption available, logs warning and skips
   - All errors are logged for debugging
"""
