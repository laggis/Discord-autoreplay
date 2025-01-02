import discord
from discord.ext import commands
import io
import os
import json
import time
import yaml
import datetime
import pytesseract
import requests
from bs4 import BeautifulSoup
import asyncio
import numpy as np
import cv2
from PIL import Image, ImageStat, ImageEnhance
from moderation import ContentModerator
import random
import re
import aiohttp

# Intents are required to receive certain events
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

def check_tesseract_installation():
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not check_tesseract_installation():
    installation_guide = """
Tesseract OCR is not installed. Please follow these steps to install it:

1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (select "Add to PATH" during installation)
3. Restart your computer after installation

If you've already installed it, make sure it's added to your system PATH.
Default installation path is: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
    print(installation_guide)
    exit()

bot = commands.Bot(command_prefix='!', intents=intents)

class LearningCache:
    def __init__(self):
        self.keywords = {}
        self.image_rules = {}
        self.swear_words = []
        self.learned_responses = {}
        self.learning_state = {}  # Tracks questions waiting for answers
        self.load_config()

    def load_config(self):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.keywords = config.get('keywords', {})
                self.image_rules = config.get('image_rules', {})
                self.swear_words = config.get('swear_words', [])
                self.learned_responses = config.get('learned_responses', {})
        except Exception as e:
            print(f"Error loading config: {e}")
            # Initialize with empty values if config fails to load
            self.keywords = {}
            self.image_rules = {}
            self.swear_words = []
            self.learned_responses = {}

    def save_config(self):
        try:
            # Read existing config to preserve comments and formatting
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Update with new values
            config['keywords'] = self.keywords
            config['image_rules'] = self.image_rules
            config['swear_words'] = self.swear_words
            config['learned_responses'] = self.learned_responses

            # Write back to file
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error saving config: {e}")

    def start_learning(self, question, user_id):
        """Start learning mode for a question"""
        clean_question = question.lower().strip('?! ')
        if clean_question not in self.learned_responses and clean_question not in self.keywords:
            self.learning_state[user_id] = clean_question
            return True
        return False

    def add_learned_response(self, question, answer):
        """Add a new learned response"""
        self.learned_responses[question] = {
            'answer': answer,
            'learned_at': str(datetime.datetime.utcnow()),
            'uses': 0
        }
        self.save_config()

    def get_learned_response(self, question):
        """Get a learned response if it exists"""
        clean_question = question.lower().strip()
        if clean_question in self.learned_responses:
            response = self.learned_responses[clean_question]
            response['uses'] += 1
            self.save_config()  # Save updated usage stats
            return response['answer']
        return None

# Initialize the learning cache
cache = LearningCache()

class DiscordBot(commands.Bot):
    def __init__(self):
        # Initialize bot with intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix='!', intents=intents)

        # Initialize message tracking
        self._message_locks = {}
        self._pending_questions = {}

        # Load config
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            print("Config loaded successfully")
        except FileNotFoundError:
            print("No config file found, creating new one")
            self.config = {}
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {}
        
        # Initialize moderator after config is loaded
        if 'moderation' not in self.config:
            self.config['moderation'] = {
                'enabled': True,
                'warn_on_swear': True,
                'timeout_duration': 5,
                'max_warnings': 3,
                'settings': {
                    'auto_moderation': True,
                    'severity_levels': {
                        'low': 0.3,
                        'medium': 0.6,
                        'high': 0.8
                    },
                    'actions': {
                        'low': 'warn',
                        'medium': 'timeout',
                        'high': 'ban'
                    }
                }
            }
            self.save_config()
        
        self.moderator = ContentModerator(self.config)
        print("Content moderator initialized")
        
        # Add message cache with TTL
        self.message_cache = {}
        self.cache_ttl = 5  # seconds
        # Initialize cooldown tracking
        self._cooldowns = {}
        self._cooldown_duration = 60  # Default 60 seconds cooldown
        # Initialize keyword usage tracking
        self._keyword_usage = {}
        
    def _clean_cache(self):
        """Clean old entries from message cache"""
        current_time = time.time()
        self.message_cache = {
            k: v for k, v in self.message_cache.items() 
            if current_time - v['timestamp'] < self.cache_ttl
        }

    def check_cooldown(self, user_id: int, command: str) -> bool:
        """
        Check if a user is on cooldown for a specific command
        Returns True if the user can use the command, False if they're on cooldown
        """
        current_time = time.time()
        cooldown_key = f"{user_id}:{command}"
        
        if cooldown_key in self._cooldowns:
            last_used = self._cooldowns[cooldown_key]
            if current_time - last_used < self._cooldown_duration:
                return False
                
        self._cooldowns[cooldown_key] = current_time
        return True

    async def reload_config(self):
        """Reload bot configuration"""
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.moderator.reload_config()
            return True
        except Exception as e:
            print(f"Error reloading config: {e}")
            return False

    async def on_ready(self):
        print(f"Logged in as {self.user}!")

    async def on_member_join(self, member):
        welcome_channel = discord.utils.get(member.guild.channels, name='welcome')
        if welcome_channel:
            embed = discord.Embed(
                title=f"Welcome to {member.guild.name}!",
                description=f"Hello {member.mention}! Welcome to our server!",
                color=discord.Color.blue()
            )
            embed.set_thumbnail(url=member.avatar.url if member.avatar else member.default_avatar.url)
            await welcome_channel.send(embed=embed)

    async def on_message(self, message):
        try:
            # Don't respond to our own messages
            if message.author == self.user:
                return

            # Process commands first
            await self.process_commands(message)

            # Check moderation first
            if self.moderator and message.content:
                try:
                    result = self.moderator.check_content(message.content, str(message.author.id))
                    if result.action != "allow":
                        await message.delete()
                        
                        # Create moderation embed
                        embed = discord.Embed(
                            title="Message Removed",
                            description=f"{message.author.mention}, your message was removed due to inappropriate content.",
                            color=discord.Color.red()
                        )
                        embed.add_field(name="Reason", value=result.message, inline=False)
                        embed.add_field(name="Action", value=result.action.title(), inline=True)
                        if result.timeout_duration:
                            embed.add_field(name="Timeout Duration", value=f"{result.timeout_duration} minutes", inline=True)
                        
                        embed.set_footer(text="Please keep the chat clean and friendly!")
                        
                        # Send the embed
                        await message.channel.send(embed=embed)
                        
                        # Apply timeout if specified
                        if result.timeout_duration:
                            await message.author.timeout(
                                duration=datetime.timedelta(minutes=result.timeout_duration),
                                reason=result.message
                            )
                        return
                except Exception as e:
                    print(f"Error in moderation: {e}")

            # Check for attachments first
            if message.attachments:
                try:
                    for attachment in message.attachments:
                        if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                            print(f"Processing image: {attachment.filename}")
                            await self.process_image(message, attachment)
                            return
                except Exception as e:
                    print(f"Error processing attachment: {e}")

            # Check if this message is already being processed
            if message.id in self._message_locks:
                return
            
            # Mark this message as being processed
            self._message_locks[message.id] = True

            try:
                # Get the message content (don't convert to lower case yet)
                content = message.content
                print(f"Processing message: {content}")

                # First check if this is a reply to a pending question
                if message.reference:
                    try:
                        reference_msg = await message.channel.fetch_message(message.reference.message_id)
                        print(f"Reference message found: {reference_msg.content}")
                        print(f"Bot message IDs: {[msg_id for msg_id in self._pending_questions.keys()]}")
                        
                        if message.reference.message_id in self._pending_questions:
                            await self.handle_pending_question_reply(message)
                            return
                        else:
                            print(f"Message is a reply but not to a pending question. Reference ID: {message.reference.message_id}")
                    except discord.NotFound:
                        print("Could not find referenced message")
                    except Exception as e:
                        print(f"Error processing reply: {e}")

                # Process normal message
                await self.process_normal_message(message, content)

            except Exception as e:
                print(f"Error processing message content: {e}")
            finally:
                # Always clean up the message lock
                if message.id in self._message_locks:
                    del self._message_locks[message.id]

        except Exception as e:
            print(f"Error in on_message: {e}")

    async def process_normal_message(self, message, content):
        """Process a normal (non-reply) message"""
        try:
            # Convert to lower case for matching
            content_lower = content.lower()
            
            # Check if it's a question
            if is_question(content_lower):
                print(f"Message identified as question: {content_lower}")
                await self.handle_question(message, content)
            else:
                # Check for keywords in the message
                await self.handle_text_response(message, content_lower)
        except Exception as e:
            print(f"Error in process_normal_message: {e}")

    async def handle_text_response(self, message, content: str):
        """Handle general text responses"""
        try:
            # Check for keyword matches in sorted order (longest first)
            sorted_keywords = sorted(self.config.get('keywords', {}).items(), 
                                  key=lambda x: len(x[0]), reverse=True)
            
            for keyword, data in sorted_keywords:
                if keyword.lower() in content.lower():
                    # Check cooldown
                    if 'last_used' in data:
                        last_used = data['last_used']
                        cooldown = data.get('cooldown', 30)  # Default 30 seconds
                        if last_used and time.time() - last_used < cooldown:
                            continue

                    # Create embed response
                    embed = discord.Embed(
                        title="🐧 Penguin Svar",
                        description=data['response'],
                        color=discord.Color.blue()
                    )
                    
                    if 'tags' in data and data['tags']:
                        embed.add_field(
                            name="Kategorier",
                            value=" | ".join(f"#{tag}" for tag in data['tags']),
                            inline=False
                        )
                    
                    embed.set_footer(text="PenguinBot | Ditt Hosting Support System")
                    
                    # Update usage statistics
                    data['uses'] = data.get('uses', 0) + 1
                    data['last_used'] = time.time()
                    
                    # Save config
                    self.save_config()
                    
                    # Send response
                    await message.channel.send(embed=embed)
                    return  # Only send one response per message

            # If no keyword match was found and it's not a command
            if not content.startswith('!'):
                print(f"No keyword match found for message: {content}")
                
        except Exception as e:
            print(f"Error in handle_text_response: {e}")

    def save_config(self):
        """Save the current configuration to file"""
        try:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True)
            print("Config saved successfully")
        except Exception as e:
            print(f"Error saving config: {e}")

    async def handle_question(self, message, content):
        """Handle a potential question"""
        try:
            # Check if this exact question or very similar exists in keywords
            normalized_content = content.lower().strip()
            found_match = False
            matching_keyword = None
            
            for keyword in self.config.get('keywords', {}):
                # Compare normalized versions of the questions
                if normalized_content == keyword.lower().strip():
                    print(f"Found exact match with keyword: {keyword}")
                    found_match = True
                    matching_keyword = keyword
                    break
                    
                # Check if the question is very similar
                content_words = set(normalized_content.split())
                keyword_words = set(keyword.lower().strip().split())
                
                # Extract key terms (words that define the subject matter)
                content_key_terms = {word for word in content_words 
                                  if word not in {'kan', 'du', 'hjälpa', 'mig', 'att', 'logga', 'in', 'på', 'i', 'och', 'eller', 'men', 'har', 'mitt', 'till'}}
                keyword_key_terms = {word for word in keyword_words 
                                  if word not in {'kan', 'du', 'hjälpa', 'mig', 'att', 'logga', 'in', 'på', 'i', 'och', 'eller', 'men', 'har', 'mitt', 'till'}}
                
                # Calculate matches for both all words and key terms
                word_match = len(content_words & keyword_words) / len(content_words)
                key_term_match = len(content_key_terms & keyword_key_terms) / max(len(content_key_terms), 1)
                
                print(f"Word match with '{keyword}': {word_match:.2%}")
                print(f"Key term match with '{keyword}': {key_term_match:.2%}")
                
                # Only consider it a match if both overall words AND key terms are similar
                if word_match >= 0.8 and key_term_match >= 0.8:
                    print(f"Found similar match with keyword: {keyword}")
                    found_match = True
                    matching_keyword = keyword
                    break
            
            if not found_match:
                print("No matching keyword found, entering learning mode")
                # Send the learning mode message and store both IDs
                bot_response = await message.channel.send(
                    embed=discord.Embed(
                        title="Ny Fråga Upptäckt 🤔",
                        description="Jag känner inte till svaret på denna fråga än. En medlem från support-teamet kommer att hjälpa till med ett svar snart!",
                        color=discord.Color.blue()
                    ).add_field(
                        name="Information",
                        value="Endast support-teamet kan lära mig nya svar för att säkerställa kvaliteten på informationen.",
                        inline=False
                    ).set_footer(text="PenguinBot | Lärande Mode")
                )
                
                # Store both the original question ID and the bot's response ID
                self._pending_questions[bot_response.id] = {
                    "question": content,
                    "timestamp": time.time(),
                    "original_message_id": message.id
                }
                print(f"Added question to pending. Bot response ID: {bot_response.id}")
            else:
                print("Found matching keyword, sending embed response")
                keyword_data = self.config['keywords'][matching_keyword]
                
                # Create a modern looking embed
                embed = discord.Embed(
                    title="🐧 Penguin Svar",
                    description=keyword_data['response'],
                    color=discord.Color.blue()
                )
                
                # Add tags if they exist
                if 'tags' in keyword_data and keyword_data['tags']:
                    embed.add_field(
                        name="Kategorier",
                        value=" | ".join(f"#{tag}" for tag in keyword_data['tags']),
                        inline=False
                    )
                
                # Add footer
                embed.set_footer(text="PenguinBot | Ditt Hosting Support System")
                
                # Send the embed
                await message.channel.send(embed=embed)
        except Exception as e:
            print(f"Error handling question: {e}")

    async def handle_stuck_message(self, message):
        """Handle 'sitter fast i lufen' messages"""
        try:
            # Get stuck message configurations
            stuck_config = self.config['stuck_messages']
            
            # Check cooldown
            user_id = str(message.author.id)
            current_time = time.time()
            
            if hasattr(self, '_stuck_cooldowns') and user_id in self._stuck_cooldowns:
                if current_time - self._stuck_cooldowns[user_id] < stuck_config['cooldown']:
                    return  # Still in cooldown
            
            # Initialize cooldown dict if it doesn't exist
            if not hasattr(self, '_stuck_cooldowns'):
                self._stuck_cooldowns = {}
            
            # Update cooldown
            self._stuck_cooldowns[user_id] = current_time
            
            # Create response embed
            embed = discord.Embed(
                title="🛟 Hjälp med Fastsittning",
                description=random.choice(stuck_config['responses']['initial']),
                color=discord.Color.blue()
            )
            
            # Add solutions
            solutions = "\n".join(f"{i+1}. {solution}" for i, solution in enumerate(stuck_config['solutions']))
            embed.add_field(
                name="Lösningar",
                value=solutions,
                inline=False
            )
            
            # Add additional info
            additional_info = "\n".join(f"• {info}" for info in stuck_config['additional_info'])
            embed.add_field(
                name="Extra Information",
                value=additional_info,
                inline=False
            )
            
            # Send the response
            await message.channel.send(embed=embed)
            
        except Exception as e:
            print(f"Error handling stuck message: {e}")
            error_config = self.config['responses']['errors']['processing']
            await message.channel.send(
                embed=discord.Embed(
                    title=error_config['title'],
                    description=error_config['message'],
                    color=discord.Color.red()
                )
            )

    async def handle_time_message(self, message):
        """Handle time-related queries"""
        try:
            current_time = datetime.datetime.now().strftime("%H:%M")
            await message.channel.send(f"Klockan är {current_time}")
        except Exception as e:
            print(f"Error handling time message: {e}")

    async def handle_pending_question_reply(self, message):
        """Handle a reply to a pending question"""
        try:
            # Check if user has the support role
            support_role = discord.utils.get(message.guild.roles, id=1229151956346605730)
            if support_role not in message.author.roles:
                await message.channel.send(
                    embed=discord.Embed(
                        title="❌ Behörighet Saknas",
                        description="Endast support-teamet kan lära mig nya svar.",
                        color=discord.Color.red()
                    ).set_footer(text="PenguinBot | Behörighetskontroll")
                )
                return
                
            print("Found reply to pending question")
            bot_msg_id = message.reference.message_id
            question_data = self._pending_questions[bot_msg_id]
            
            # Add the new response to keywords
            if "keywords" not in self.config:
                self.config["keywords"] = {}
            
            print(f"Adding new response for question: {question_data['question']}")
            print(f"Response: {message.content}")
            
            self.config["keywords"][question_data['question']] = {
                "tags": ["learned"],
                "response": message.content,  # Keep original case
                "cooldown": 30,
                "uses": 0,
                "last_used": None
            }
            
            # Save the updated config
            try:
                with open('config.yaml', 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, allow_unicode=True)
                print("Successfully saved config")
            except Exception as e:
                print(f"Error saving config: {e}")
            
            # Remove from pending questions
            del self._pending_questions[bot_msg_id]
            print(f"Removed question from pending. Remaining: {list(self._pending_questions.keys())}")
            
            # Send confirmation
            await message.channel.send(
                embed=discord.Embed(
                    title="✨ Nytt Svar Lärt!",
                    description="Tack för ditt svar! Jag har lärt mig och kommer använda denna information för att hjälpa andra.",
                    color=discord.Color.green()
                ).add_field(
                    name="Fråga",
                    value=question_data['question'],
                    inline=False
                ).add_field(
                    name="Svar",
                    value=message.content,
                    inline=False
                ).set_footer(text="PenguinBot | Lärande System")
            )
        except Exception as e:
            print(f"Error handling pending question reply: {e}")

    async def handle_learning_answer(self, message, original_question):
        """Handle answer to a learning question"""
        try:
            if len(message.content) > 5:  # Ensure answer is substantial
                cache.add_learned_response(original_question, message.content)
                del cache.learning_state[message.author.id]
                await send_embed_message(
                    message.channel,
                    "Tack för svaret! 📚",
                    f"Jag har lärt mig att svara på frågan: '{original_question}'",
                    "success"
                )
        except Exception as e:
            print(f"Error handling learning answer: {e}")

    def censor_swear_words(self, text):
        """Censor swear words in text with asterisks"""
        # Get swear words from config
        swear_words = self.config.get('moderation', {}).get('swear_words', [])
        
        censored = text
        for word in swear_words:
            # Create a pattern that matches the word with word boundaries
            pattern = r'\b' + re.escape(word) + r'\b'
            # Replace with asterisks of the same length
            censored = re.sub(pattern, '*' * len(word), censored, flags=re.IGNORECASE)
        
        return censored

    async def process_image(self, message, attachment):
        """Process an image attachment and extract text"""
        try:
            # Download the image
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as response:
                    if response.status != 200:
                        return None
                    image_data = await response.read()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)  # Increase contrast
            
            # Use pytesseract with custom config for better accuracy
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Clean and format the extracted text
            text = self.clean_extracted_text(text)
            print(f"Extracted text from image: {text}")
            
            if not text.strip():
                await message.channel.send(
                    embed=discord.Embed(
                        title="🤔 Ingen Text Hittad",
                        description="Jag kunde inte hitta någon text i bilden. Var god försök igen med en tydligare bild.",
                        color=discord.Color.orange()
                    ).set_footer(text="PenguinBot | Bildanalys")
                )
                return None
            
            # Check for known error patterns
            for error_type, error_data in self.config.get('fivem_errors', {}).items():
                if any(trigger.lower() in text.lower() for trigger in error_data.get('triggers', [])):
                    embed = discord.Embed(
                        title=f"🔍 Fel Identifierat",
                        description=error_data.get('response', 'Ingen lösning tillgänglig'),
                        color=discord.Color.blue()
                    )
                    embed.set_footer(text="PenguinBot | Felanalys")
                    await message.channel.send(embed=embed)
                    return True
            
            # Check keywords
            content = text.lower()
            found_match = False
            for keyword, data in self.config.get('keywords', {}).items():
                if keyword.lower() in content:
                    embed = discord.Embed(
                        title="🔍 Matchande Svar Hittat",
                        description=data['response'],
                        color=discord.Color.blue()
                    )
                    if 'tags' in data:
                        embed.add_field(
                            name="Kategorier",
                            value=" | ".join(f"#{tag}" for tag in data['tags']),
                            inline=False
                        )
                    embed.set_footer(text="PenguinBot | Bildanalys")
                    await message.channel.send(embed=embed)
                    found_match = True
                    break
            
            # If no match was found, enter learning mode
            if not found_match:
                # Send the learning mode message and store both IDs
                bot_response = await message.channel.send(
                    embed=discord.Embed(
                        title="📸 Ny Fråga Från Bild 🤔",
                        description="Jag hittade text i bilden men känner inte igen problemet. En medlem från support-teamet kan lära mig svaret!",
                        color=discord.Color.blue()
                    ).add_field(
                        name="Extraherad Text",
                        value=f"```{text}```",
                        inline=False
                    ).add_field(
                        name="Information",
                        value="Endast support-teamet kan lära mig nya svar för att säkerställa kvaliteten på informationen.",
                        inline=False
                    ).set_footer(text="PenguinBot | Bildanalys & Lärande")
                )
                
                # Store both the original question ID and the bot's response ID
                self._pending_questions[bot_response.id] = {
                    "question": text,
                    "timestamp": time.time(),
                    "original_message_id": message.id,
                    "is_image": True
                }
                return False
            
        except Exception as e:
            print(f"Error processing image: {e}")
            await message.channel.send(
                embed=discord.Embed(
                    title="❌ Bildbehandlingsfel",
                    description="Ett fel uppstod när jag försökte analysera bilden. Var god försök igen eller beskriv problemet i text.",
                    color=discord.Color.red()
                ).set_footer(text="PenguinBot | Bildanalys")
            )
            return None

    def clean_extracted_text(self, text: str) -> str:
        """Clean and format extracted text to be more meaningful"""
        try:
            # Split into lines
            lines = text.split('\n')
            cleaned_lines = []
            
            # Track repeated lines to avoid spam
            last_line = None
            repeat_count = 0
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Clean the line
                line = line.strip()
                
                # Remove common OCR artifacts
                line = re.sub(r'[^\w\s\-.,!?()[\]{}:;\'"@#$%&*+=/<>]', '', line)
                
                # Skip lines that are just numbers or very short
                if line.isdigit() or len(line) < 3:
                    continue
                
                # Check for repeated content
                if line == last_line:
                    repeat_count += 1
                    if repeat_count > 2:  # Only show up to 3 repetitions
                        continue
                else:
                    repeat_count = 0
                
                # Add the line if it's meaningful
                if line:
                    cleaned_lines.append(line)
                    last_line = line
            
            # Join lines back together
            text = '\n'.join(cleaned_lines)
            
            # If we have repeated lines at the end, add a note
            if repeat_count > 2:
                text += f"\n(och {repeat_count-2} liknande rader)"
            
            return text.strip()
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text

def is_question(text: str) -> bool:
    """Check if the text is likely a question"""
    # Common Swedish question words and patterns
    question_words = {
        'hur', 'vad', 'när', 'var', 'vem', 'vilken', 'vilket', 'vilka',
        'varför', 'kan du', 'skulle du kunna', 'vet du', 'har du'
    }
    
    # Convert to lower case and strip whitespace
    text = text.lower().strip()
    
    # Check if text ends with question mark
    if text.endswith('?'):
        return True
        
    # Check for question words at start of text
    words = text.split()
    if words:
        if words[0] in question_words:
            return True
        # Check for two-word combinations
        if len(words) >= 2 and ' '.join(words[:2]) in question_words:
            return True
            
    return False

def analyze_image_features(image):
    """Analyze image features to detect specific patterns"""
    try:
        img = Image.open(image)
        
        # Convert to grayscale for analysis
        gray_img = img.convert('L')
        
        # Get image statistics
        stats = ImageStat.Stat(gray_img)
        mean = stats.mean[0]
        
        # Convert to numpy array for advanced analysis
        img_array = np.array(gray_img)
        
        # More specific detection for night city views
        # Check if image is predominantly dark (night scene)
        is_night_scene = 20 < mean < 80  # Tighter range for night scenes
        
        # Detect grid patterns more precisely
        edges = cv2.Canny(img_array, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150, minLineLength=150, maxLineGap=20)
        
        # Count horizontal and vertical lines separately
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                angles.append(angle)
            
            # Count lines that are roughly horizontal (0±15°) or vertical (90±15°)
            horizontal_lines = sum(1 for angle in angles if angle < 15 or angle > 165)
            vertical_lines = sum(1 for angle in angles if 75 < angle < 105)
            
            # Must have both horizontal and vertical lines for a true grid
            has_grid_pattern = horizontal_lines > 5 and vertical_lines > 5
        else:
            has_grid_pattern = False
        
        # Check for bright spots distribution (city lights pattern)
        bright_threshold = 200
        bright_spots = img_array > bright_threshold
        bright_spot_count = np.sum(bright_spots)
        
        # Calculate the ratio of bright spots
        total_pixels = img_array.size
        bright_ratio = bright_spot_count / total_pixels
        
        # Only consider it a city lights pattern if the bright spots are well distributed
        has_city_lights = 0.001 < bright_ratio < 0.1  # Adjust these thresholds as needed
        
        return {
            "is_night_scene": is_night_scene and has_city_lights,  # Must have both dark background and city lights
            "has_grid_pattern": has_grid_pattern,
            "brightness_mean": mean,
            "has_bright_spots": has_city_lights
        }
    except Exception as e:
        print(f"Error analyzing image features: {e}")
        return None

def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if image is in color
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # Apply thresholding to get black and white image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        
        # Convert back to PIL Image
        return Image.fromarray(dilated)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return image

def extract_text_from_image(image):
    """Extract text from image with improved accuracy"""
    try:
        # Convert to PIL Image if it's not already
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        
        # Create multiple versions of the image for better OCR
        preprocessed = preprocess_image_for_ocr(image)
        
        # OCR Configuration
        custom_config = r'--oem 3 --psm 6'
        
        # Try OCR on both original and preprocessed images
        text_original = pytesseract.image_to_string(image, config=custom_config).strip()
        text_preprocessed = pytesseract.image_to_string(preprocessed, config=custom_config).strip()
        
        # Combine results (preprocessed version often works better for error messages)
        combined_text = text_preprocessed if len(text_preprocessed) > len(text_original) else text_original
        
        # Clean up the text
        cleaned_text = ' '.join(combined_text.split())  # Remove extra whitespace
        return cleaned_text.lower()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def detect_image_error(image):
    if not check_tesseract_installation():
        installation_guide = """
Tesseract OCR is not installed. Please follow these steps to install it:

1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (select "Add to PATH" during installation)
3. Restart your computer after installation

If you've already installed it, make sure it's added to your system PATH.
Default installation path is: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
        return installation_guide

    try:
        # First check for specific error patterns using improved OCR
        img = Image.open(image)
        extracted_text = extract_text_from_image(img)
        
        # Check for specific error keywords
        error_keywords = [
            "error", "rejected", "failed", "exception",
            "fel", "misslyckades", "anslutning", "server"
        ]
        
        # Look for error-related text
        if any(keyword in extracted_text for keyword in error_keywords):
            # Check against configured rules for specific error messages
            for rule, response in cache.image_rules.items():
                if rule.lower() in extracted_text:
                    return response
            return None  # If no specific error message matches, don't process as aerial view
        
        # If no error text found, proceed with feature analysis
        features = analyze_image_features(image)
        if features and features["is_night_scene"] and features["has_grid_pattern"]:
            return "Det ser ut som att du sitter fast i lufen, Följ denna [Youtube Guide](https://youtu.be/1bmr0ce2Pmc?si=HEoEgI9a6OaCC0Es)"
        
        return None
    except Exception as e:
        return f"Error processing image: {e}"

# Function to recursively fetch directory structure for Fenix only
async def fetch_directory_structure(url, depth=0):
    """
    Recursively fetches directories and files only under the 'Fenix' directory.
    Logs progress for debugging purposes.
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch: {url}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True)]

        valid_links = []
        for link in links:
            if link in ["/", "../"]:  # Skip unnecessary links
                continue
            full_url = url + link
            if link.endswith('/'):  # This is a directory
                # Recursively fetch contents of this directory
                print(f"Fetching directory: {full_url}")
                valid_links += await fetch_directory_structure(full_url, depth + 1)
            else:
                # This is a file
                valid_links.append(full_url)

        print(f"Found links at depth {depth}: {valid_links}")
        return valid_links
    except Exception as e:
        print(f"Error fetching directory at URL {url}: {e}")
        return []


# Cache for directory structure
async def get_cached_directory():
    """
    Fetches the cached directory structure or scrapes it if necessary.
    Only focuses on 'Fenix' subfolders.
    Logs status to debug scraping and cache performance.
    """
    global cache
    current_time = asyncio.get_event_loop().time()
    
    if current_time - cache.last_fetch_time > 600:  # Every 10 minutes
        print("Scraping directory structure from server...")
        cache.directory_links = await fetch_directory_structure(base_url)
        cache.last_fetch_time = current_time
        print("Directory structure cached. Total items found:", len(cache.directory_links))
    else:
        print("Using cached directory structure.")
    return cache.directory_links


# Function to find a matching script path
def find_script_path(query, directory_links):
    """
    Searches only under the cached links for user-provided queries.
    Includes debug logging to trace query progress.
    """
    print(f"Searching for query: '{query}'")
    query = query.lower()
    for link in directory_links:
        if query in link.lower():
            print(f"Match found: {link}")
            return link
    print("No match found.")
    return None


def format_message(message):
    """Format message with markdown links if present"""
    import re
    # Match markdown links: [text](url)
    pattern = r'\[(.*?)\]\((.*?)\)'
    
    def replace_link(match):
        text, url = match.groups()
        return f"{text} ({url})"
    
    return re.sub(pattern, replace_link, message)


def create_error_embed(title, description, error_type="error"):
    """Create a formatted embed for error messages"""
    embed = discord.Embed()
    
    if error_type == "error":
        embed.color = discord.Color.from_rgb(231, 76, 60)  # Bright red
        embed.title = "🚫 " + title
    elif error_type == "warning":
        embed.color = discord.Color.from_rgb(241, 196, 15)  # Warm yellow
        embed.title = "⚠️ " + title
    elif error_type == "success":
        embed.color = discord.Color.from_rgb(46, 204, 113)  # Emerald green
        embed.title = "✅ " + title
    elif error_type == "info":
        embed.color = discord.Color.from_rgb(52, 152, 219)  # Bright blue
        embed.title = "💡 " + title
    
    # Add timestamp
    embed.timestamp = datetime.datetime.utcnow()
    
    # Format description with emojis based on type
    if error_type == "error":
        description = f"```diff\n- {description}\n```"
    elif error_type == "warning":
        description = f"```fix\n{description}\n```"
    elif error_type == "success":
        description = f"```diff\n+ {description}\n```"
    elif error_type == "info":
        description = f"```yaml\n{description}\n```"
    
    embed.description = description
    embed.set_footer(text="LaGgls Server | Support Bot", icon_url="https://i.imgur.com/XqQR0vN.png")
    return embed

def create_stuck_embed():
    """Create a formatted embed for the stuck in air message"""
    embed = discord.Embed(color=discord.Color.from_rgb(114, 137, 218))  # Discord blurple color
    embed.title = "🌟 Fastnat i luften?"
    
    # Main description with fancy formatting
    embed.description = """
**Problem upptäckt:** Du verkar ha fastnat i luften.
**Ingen fara!** Följ guiden nedan för att lösa problemet.
"""
    
    # Add solution field with emoji
    embed.add_field(
        name="🎮 Lösning",
        value="[**Klicka här för att se videoguiden**](https://youtu.be/1bmr0ce2Pmc?si=HEoEgI9a6OaCC0Es)\n*En enkel guide som hjälper dig att komma ner på marken igen*",
        inline=False
    )
    
    # Add tips field
    embed.add_field(
        name="💡 Tips",
        value="Se till att följa alla steg i guiden noggrant",
        inline=False
    )
    
    # Add timestamp
    embed.timestamp = datetime.datetime.utcnow()
    
    # Add footer with server icon
    embed.set_footer(
        text="LaGgls Server | Support Bot",
        icon_url="https://i.imgur.com/XqQR0vN.png"
    )
    
    # Add a thumbnail
    embed.set_thumbnail(url="https://i.imgur.com/XqQR0vN.png")
    
    return embed

async def send_embed_message(channel, title, description, error_type="error"):
    """Send a formatted embed message"""
    embed = create_error_embed(title, description, error_type)
    await channel.send(embed=embed)

# Role Management Commands
@bot.command()
@commands.has_permissions(manage_roles=True)
async def addrole(ctx, member: discord.Member, *, role: discord.Role):
    """Add a role to a member"""
    try:
        await member.add_roles(role)
        await ctx.send(f"Added role {role.name} to {member.name}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to manage roles!")

@bot.command()
@commands.has_permissions(manage_roles=True)
async def removerole(ctx, member: discord.Member, *, role: discord.Role):
    """Remove a role from a member"""
    try:
        await member.remove_roles(role)
        await ctx.send(f"Removed role {role.name} from {member.name}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to manage roles!")

# Moderation Commands
@bot.command()
@commands.has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason=None):
    """Kick a member from the server"""
    try:
        await member.kick(reason=reason)
        await ctx.send(f"{member.name} has been kicked. Reason: {reason}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to kick members!")

@bot.command()
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason=None):
    """Ban a member from the server"""
    try:
        await member.ban(reason=reason)
        await ctx.send(f"{member.name} has been banned. Reason: {reason}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to ban members!")

@bot.command()
@commands.has_permissions(moderate_members=True)
async def timeout(ctx, member: discord.Member, minutes: int, *, reason=None):
    """Timeout a member for specified minutes"""
    try:
        duration = datetime.timedelta(minutes=minutes)
        await member.timeout_for(duration, reason=reason)
        await ctx.send(f"{member.name} has been timed out for {minutes} minutes. Reason: {reason}")
    except discord.Forbidden:
        await ctx.send("I don't have permission to timeout members!")

bot = DiscordBot()
bot.run("Your discord token here")