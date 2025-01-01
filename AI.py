import discord
from discord.ext import commands
import io
import os
import json
import yaml
import datetime
import pytesseract
import requests
from bs4 import BeautifulSoup
import asyncio
import numpy as np
import cv2
from PIL import Image, ImageStat
from moderation import ContentModerator
import time
import random
import re

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
        clean_question = question.lower().strip('?! ')
        if clean_question in self.learned_responses:
            response = self.learned_responses[clean_question]
            response['uses'] += 1
            self.save_config()
            return response['answer']
        return None

# Initialize the learning cache
cache = LearningCache()

class DiscordBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix='!', intents=intents)
        
        # Load config
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {}
        
        self.moderator = ContentModerator(self.config)
        print("Content moderator initialized")
        
        # Add message cache with TTL
        self.message_cache = {}
        self.cache_ttl = 5  # seconds
        
    def _clean_cache(self):
        """Clean old entries from message cache"""
        current_time = time.time()
        self.message_cache = {
            k: v for k, v in self.message_cache.items() 
            if current_time - v['timestamp'] < self.cache_ttl
        }

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
        if message.author == self.user:
            return

        # Check if this message is already being processed
        if message.id in self._message_locks:
            return
            
        # Mark this message as being processed
        self._message_locks[message.id] = True

        try:
            # Check message content with moderator
            result = self.moderator.check_content(message.content, str(message.author.id))
            
            # Handle moderation actions
            if result.action == "timeout":
                try:
                    await message.author.timeout(
                        duration=datetime.timedelta(seconds=result.timeout_duration),
                        reason="Automated moderation action"
                    )
                    await message.channel.send(
                        embed=discord.Embed(
                            title="🚫 Moderation Action",
                            description=f"{message.author.mention} {result.message}",
                            color=discord.Color.red()
                        )
                    )
                    await message.delete()
                    return
                except discord.Forbidden:
                    print("Bot doesn't have permission to timeout users")
                
            elif result.action == "delete":
                try:
                    await message.delete()
                    await message.channel.send(
                        embed=discord.Embed(
                            title="⚠️ Content Warning",
                            description=f"{message.author.mention} {result.message}",
                            color=discord.Color.orange()
                        )
                    )
                    return
                except discord.Forbidden:
                    print("Bot doesn't have permission to delete messages")
                
            elif result.action == "warn":
                await message.channel.send(
                    embed=discord.Embed(
                        title="⚠️ Warning",
                        description=f"{message.author.mention} {result.message}",
                        color=discord.Color.yellow()
                    )
                )

            # Process commands and messages
            await self.process_commands(message)
            
            # Handle images first if present, then skip text processing
            if message.attachments:
                await self.handle_image_search(message)
                return  # Skip text processing if we handled an image
            
            if message.content:
                content = message.content.lower()
                
                # Handle specific commands
                if "sitter fast i lufen" in content:
                    await self.handle_stuck_message(message)
                elif "vad är klockan" in content:
                    await self.handle_time_message(message)
                
                # Check for learning state
                if message.author.id in cache.learning_state:
                    await self.handle_learning_answer(message, cache.learning_state[message.author.id])
                    return

                # Check for learned responses
                learned_response = cache.get_learned_response(content)
                if learned_response:
                    await send_embed_message(
                        message.channel,
                        "Svar 💡",
                        learned_response,
                        "info"
                    )
                    return

                # Handle learning mode for questions
                if is_question(content) and not any(keyword.lower() in content for keyword in cache.keywords):
                    if cache.start_learning(content, message.author.id):
                        await send_embed_message(
                            message.channel,
                            "Ny Fråga Upptäckt 🤔",
                            "Jag känner inte till svaret på denna fråga än. Om någon vet svaret, skriv det så lär jag mig!",
                            "info"
                        )
                        return

                # Handle general text responses
                await self.handle_text_response(message, content)
                
        except Exception as e:
            print(f"Error in on_message: {e}")
            error_config = self.config['responses']['errors']['general']
            await message.channel.send(
                embed=discord.Embed(
                    title=error_config['title'],
                    description=error_config['message'],
                    color=discord.Color.red()
                )
            )
        finally:
            # Remove the message lock when done processing
            if message.id in self._message_locks:
                del self._message_locks[message.id]

    async def handle_text_response(self, message, content: str):
        """Handle general text responses"""
        try:
            # Get response configurations
            responses = self.config['responses']
            
            # Check for greetings
            if any(trigger in content for trigger in responses['greetings']['triggers']):
                response = random.choice(responses['greetings']['responses'])
                # Format the response with username
                response = response.format(username=message.author.name)
                await message.channel.send(response)
                return

            # Check for keywords in the message
            for keyword, data in self.config['keywords'].items():
                if keyword.lower() in content:
                    # Check cooldown
                    if not self.check_cooldown(message.author.id, keyword):
                        continue
                    
                    # Handle multi-response type
                    if isinstance(data, dict) and data.get('response_type') == 'multi':
                        embed = discord.Embed(
                            title="Information",
                            description="Here's what I found:",
                            color=0x3498db
                        )
                        
                        # Add each response section
                        for section in data.get('responses', []):
                            if isinstance(section, dict):
                                title = section.get('title', '')
                                content = section.get('content', '')
                                if title and content:
                                    # Format code blocks if needed
                                    if '```' in content:
                                        content = content.replace('```', '')
                                    embed.add_field(
                                        name=title,
                                        value=content,
                                        inline=False
                                    )
                        
                        await message.reply(embed=embed)
                    else:
                        # Handle single response type
                        response = data.get('response', '') if isinstance(data, dict) else data
                        await message.reply(response)
                    
                    # Update usage statistics
                    self.update_usage(keyword)
                    return

            # Handle file search queries
            if any(trigger in content for trigger in responses['file_search']['triggers']):
                directory_links = await get_cached_directory()
                match = find_script_path(content, directory_links)

                if match:
                    await send_embed_message(
                        message.channel,
                        responses['file_search']['responses']['success']['title'],
                        responses['file_search']['responses']['success']['message'].format(path=match),
                        "success"
                    )
                else:
                    await send_embed_message(
                        message.channel,
                        responses['file_search']['responses']['not_found']['title'],
                        responses['file_search']['responses']['not_found']['message'],
                        "error"
                    )

        except Exception as e:
            print(f"Error in handle_text_response: {e}")
            error_config = self.config['responses']['errors']['processing']
            await message.channel.send(
                embed=discord.Embed(
                    title=error_config['title'],
                    description=error_config['message'],
                    color=discord.Color.red()
                )
            )

    async def download_image(self, attachment) -> str:
        """Download an image from a Discord attachment"""
        try:
            # Create temp directory if it doesn't exist
            if not os.path.exists('temp'):
                os.makedirs('temp')
            
            # Generate a unique filename
            filename = f"temp/image_{int(time.time())}_{attachment.filename}"
            
            # Download the file
            await attachment.save(filename)
            return filename
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    async def handle_image_search(self, message):
        """Handle image search"""
        try:
            if message.attachments:
                # Clean old cache entries
                self._clean_cache()
                
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif']):
                        # Generate cache key from attachment
                        cache_key = f"{attachment.id}_{attachment.filename}"
                        
                        # Check cache
                        if cache_key in self.message_cache:
                            continue
                        
                        # Download and process image
                        image_path = await self.download_image(attachment)
                        if image_path:
                            try:
                                text = self.extract_text_from_image(image_path)
                                
                                # Add to cache
                                self.message_cache[cache_key] = {
                                    'timestamp': time.time(),
                                    'text': text
                                }
                                
                                # Analyze the error type
                                error_type = self.analyze_error(text)
                                
                                if error_type:
                                    embed = self.create_error_embed(error_type, text)
                                else:
                                    # Create a regular text embed
                                    embed = discord.Embed(
                                        title="📝 Text från Bild",
                                        description=text,
                                        color=discord.Color.blue()
                                    )
                                
                                embed.set_footer(text="OCR Text Extraction")
                                await message.channel.send(embed=embed)
                            finally:
                                # Always clean up the temporary file
                                if os.path.exists(image_path):
                                    os.remove(image_path)
                            
        except Exception as e:
            print(f"Error processing image: {e}")
            error_embed = discord.Embed(
                title="❌ Bildbehandlingsfel",
                description="Kunde inte behandla bilden. Försök igen senare.",
                color=discord.Color.red()
            )
            await message.channel.send(embed=error_embed)

    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using pytesseract"""
        try:
            # Open and process the image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Increase contrast
            alpha = 1.5  # Contrast control
            beta = 10    # Brightness control
            adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # Apply thresholding
            _, binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Convert back to PIL Image
            processed_img = Image.fromarray(denoised)
            
            # Extract text with custom configuration for error messages
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?()[]{}:;\'"@#$%&*+=/<>_- "'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Clean up the text
            text = self.clean_text(text)
            
            return text if text else "Ingen text hittades i bilden."
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return "Ett fel uppstod vid textextrahering från bilden."

    def clean_text(self, text: str) -> str:
        """Clean up extracted text"""
        try:
            # Remove any null bytes and normalize whitespace
            text = text.replace('\x00', '').strip()
            
            # Split into lines and process each line
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Clean the line
                line = line.strip()
                
                # Fix common OCR mistakes
                line = line.replace('|', 'I')            # Vertical bar to I
                line = line.replace('1', 'l')            # One to lowercase L in specific cases
                line = line.replace('0', 'O')            # Zero to O in specific cases
                
                # Handle code paths
                if any(x in line for x in ['\\', '/', '_']):
                    # Preserve paths as is
                    cleaned_lines.append(line)
                    continue
                
                # Handle error messages
                if any(x in line.lower() for x in ['error', 'warning', 'exception', 'traceback']):
                    # Preserve error messages with minimal cleaning
                    cleaned_lines.append(line)
                    continue
                
                # General text cleaning
                line = re.sub(r'[^\w\s\-.,!?()[\]{}:;\'"@#$%&*+=/<>]', '', line)
                line = re.sub(r'\s+', ' ', line)
                
                if line.strip():
                    cleaned_lines.append(line)
            
            # Join lines back together
            text = '\n'.join(cleaned_lines)
            
            # Final cleanup
            text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
            text = text.strip()
            
            return text
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text  # Return original text if cleaning fails

    def analyze_error(self, text: str) -> dict:
        """Analyze the error text and return error type and details"""
        text_lower = text.lower()
        
        # Get FiveM error configurations
        fivem_errors = self.config['fivem_errors']
        
        # Check each error type
        for error_type, error_config in fivem_errors.items():
            # Check if any trigger matches
            if any(trigger.lower() in text_lower for trigger in error_config['triggers']):
                return {
                    "type": error_type,
                    "title": error_config['title'],
                    "description": error_config['description'],
                    "solution": [
                        f"{i+1}. {solution}" for i, solution in enumerate(error_config['solutions'])
                    ]
                }
        
        return None

    def create_error_embed(self, error_info: dict, original_text: str) -> discord.Embed:
        """Create an error embed based on the error type"""
        embed = discord.Embed(
            title=error_info["title"],
            description=error_info["description"],
            color=discord.Color.gold()
        )
        
        # Add the original error message
        embed.add_field(
            name="Felmeddelande",
            value=f"```{original_text}```",
            inline=False
        )
        
        # Add solutions
        embed.add_field(
            name="Möjliga lösningar",
            value="\n".join(error_info["solution"]),
            inline=False
        )
        
        return embed

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

def is_question(message):
    """Check if a message is likely a question"""
    question_indicators = ['vad', 'hur', 'när', 'var', 'vilken', 'vem', 'what', 'how', 'when', 'where', 'which', 'who']
    clean_msg = message.lower().strip()
    return any(clean_msg.startswith(q) for q in question_indicators) or clean_msg.endswith('?')

# Function to detect errors in images using OCR
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
bot.run("Your Discord Bot Token")