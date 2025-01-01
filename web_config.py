from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import yaml
import os
from functools import wraps
from datetime import datetime, timedelta
import re
import hashlib
import secrets

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session encryption
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # For "remember me" functionality

# Admin credentials - Change these to secure values
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}${hash_obj.hex()}"

def verify_password(stored_password, provided_password):
    salt = stored_password.split('$')[0]
    return stored_password == hash_password(provided_password, salt)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def load_users():
    try:
        with open('users.yaml', 'r') as f:
            return yaml.safe_load(f) or {'users': {}}
    except FileNotFoundError:
        return {'users': {}}

def save_users(users_data):
    with open('users.yaml', 'w') as f:
        yaml.dump(users_data, f)

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USERNAME or auth.password != ADMIN_PASSWORD:
            return ('Could not verify your access level for that URL.\n'
                   'You have to login with proper credentials', 401,
                   {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        users_data = load_users()
        if users_data['users'].get(session.get('username', ''), {}).get('role') != 'admin':
            flash('Admin access required', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def update_stats(config, stat_type):
    """Update statistics in config"""
    if 'stats' not in config:
        config['stats'] = {
            'total_messages': 0,
            'total_images_analyzed': 0,
            'total_responses': 0,
            'last_reset': str(datetime.utcnow()),
            'response_history': [],
            'unanswered_questions': []
        }
    
    config['stats'][f'total_{stat_type}'] += 1
    save_config(config)

def get_top_responses(config, limit=10):
    """Get most used responses"""
    responses = []
    
    # Collect keyword responses
    for keyword, data in config.get('keywords', {}).items():
        if isinstance(data, dict):  # New format
            responses.append({
                'keyword': keyword,
                'uses': data.get('uses', 0),
                'last_used': data.get('last_used'),
                'tags': data.get('tags', [])
            })
        else:  # Old format
            responses.append({
                'keyword': keyword,
                'uses': 0,
                'last_used': None,
                'tags': []
            })
    
    # Sort by uses
    responses.sort(key=lambda x: x['uses'], reverse=True)
    return responses[:limit]

def get_unanswered_questions(config):
    """Get recent unanswered questions"""
    return config.get('stats', {}).get('unanswered_questions', [])

def simulate_bot_response(message):
    """Simulate how the bot would respond to a message"""
    config = load_config()
    
    # Check keywords
    for keyword, data in config.get('keywords', {}).items():
        if keyword.lower() in message.lower():
            if isinstance(data, dict):
                if 'response_type' in data and data['response_type'] == 'multi':
                    # Format multi-response message
                    response_text = ""
                    for resp in data['responses']:
                        if isinstance(resp, dict):
                            response_text += f"**{resp.get('title', '')}**\n{resp.get('content', '')}\n\n"
                        else:
                            response_text += f"{resp}\n\n"
                    return response_text.strip()
                else:
                    return data.get('response', '')
            else:
                return data
    
    # Check learned responses
    for question, data in config.get('learning', {}).get('responses', {}).items():
        if question.lower() in message.lower():
            return data.get('answer', '')
    
    return "Jag förstår inte frågan. Kan du omformulera den?"

@app.context_processor
def inject_user_data():
    users_data = load_users()
    return {
        'users': users_data['users']
    }

@app.route('/')
@login_required
def index():
    config = load_config()
    add_response = request.args.get('add_response')
    return render_template('index.html', config=config, add_response=add_response)

@app.route('/stats')
@login_required
def stats():
    config = load_config()
    stats_data = {
        'total_messages': config.get('statistics', {}).get('total_messages', 0),
        'total_images': config.get('statistics', {}).get('total_images', 0),
        'total_responses': config.get('statistics', {}).get('total_responses', 0),
        'keywords': {
            k: v.get('uses', 0) if isinstance(v, dict) else 0 
            for k, v in config.get('keywords', {}).items()
        },
        'image_rules': {
            k: v.get('uses', 0) if isinstance(v, dict) else 0 
            for k, v in config.get('image_rules', {}).items()
        },
        'learned_responses': {
            k: v.get('uses', 0) if isinstance(v, dict) else 0 
            for k, v in config.get('learning', {}).get('responses', {}).items()
        },
        'unanswered_questions': config.get('statistics', {}).get('unanswered_questions', [])
    }
    return render_template('stats.html', stats=stats_data)

@app.route('/test_response', methods=['POST'])
@login_required
def test_response():
    message = request.form.get('test_message', '')
    response = simulate_bot_response(message)
    return jsonify({'response': response})

@app.route('/update_keywords', methods=['POST'])
@login_required
def update_keywords():
    config = load_config()
    keyword = request.form.get('keyword')
    responses = request.form.getlist('responses[]')  # Get all responses
    tags = request.form.get('tags', '').split(',')
    tags = [tag.strip() for tag in tags if tag.strip()]
    cooldown = int(request.form.get('cooldown', 5))
    
    if keyword and responses:
        if 'keywords' not in config:
            config['keywords'] = {}
            
        # Create keyword entry with multi-response support
        config['keywords'][keyword] = {
            'tags': tags,
            'cooldown': cooldown,
            'uses': 0,
            'last_used': None
        }
        
        # Handle single vs multi response
        if len(responses) > 1:
            config['keywords'][keyword]['response_type'] = 'multi'
            config['keywords'][keyword]['responses'] = responses
        else:
            config['keywords'][keyword]['response'] = responses[0]
            
        save_config(config)
        flash('Keyword updated successfully!', 'success')
    else:
        flash('Keyword and at least one response are required!', 'error')
    
    return redirect(url_for('index'))

@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    config = load_config()
    settings = request.form.to_dict()
    
    if 'settings' not in config:
        config['settings'] = {}
    
    # Update settings
    for key, value in settings.items():
        if key in ['cooldown', 'max_response_length']:
            value = int(value)
        elif key == 'learning_enabled':
            value = value.lower() == 'true'
        config['settings'][key] = value
    
    save_config(config)
    return redirect(url_for('index'))

@app.route('/reset_stats', methods=['POST'])
@login_required
def reset_stats():
    config = load_config()
    config['stats'] = {
        'total_messages': 0,
        'total_images_analyzed': 0,
        'total_responses': 0,
        'last_reset': str(datetime.utcnow()),
        'response_history': [],
        'unanswered_questions': []
    }
    save_config(config)
    return redirect(url_for('stats'))

@app.route('/delete_keyword', methods=['POST'])
@login_required
def delete_keyword():
    config = load_config()
    keyword = request.form.get('keyword')
    
    if 'keywords' in config and keyword in config['keywords']:
        del config['keywords'][keyword]
        save_config(config)
    
    return redirect(url_for('index'))

@app.route('/delete_image_rule', methods=['POST'])
@login_required
def delete_image_rule():
    config = load_config()
    rule = request.form.get('rule')
    
    if 'image_rules' in config and rule in config['image_rules']:
        del config['image_rules'][rule]
        save_config(config)
    
    return redirect(url_for('index'))

@app.route('/delete_swear_word', methods=['POST'])
@login_required
def delete_swear_word():
    config = load_config()
    word = request.form.get('word')
    
    if 'moderation' in config and 'swear_words' in config['moderation'] and word in config['moderation']['swear_words']:
        config['moderation']['swear_words'].remove(word)
        save_config(config)
    
    return redirect(url_for('index'))

@app.route('/delete_learned_response', methods=['POST'])
@login_required
def delete_learned_response():
    config = load_config()
    question = request.form.get('question')
    
    if 'learning' in config and 'responses' in config['learning'] and question in config['learning']['responses']:
        del config['learning']['responses'][question]
        save_config(config)
    
    return redirect(url_for('index'))

@app.route('/update_image_rules', methods=['POST'])
@login_required
def update_image_rules():
    config = load_config()
    rule = request.form.get('rule')
    response = request.form.get('response')
    
    if not 'image_rules' in config:
        config['image_rules'] = {}
    
    config['image_rules'][rule] = {
        'response': response,
        'uses': 0,
        'last_used': None
    }
    
    save_config(config)
    return redirect(url_for('index'))

@app.route('/update_swear_words', methods=['POST'])
@login_required
def update_swear_words():
    config = load_config()
    word = request.form.get('word')
    
    if not 'moderation' in config:
        config['moderation'] = {}
    if not 'swear_words' in config['moderation']:
        config['moderation']['swear_words'] = []
    
    if word and word not in config['moderation']['swear_words']:
        config['moderation']['swear_words'].append(word)
    
    save_config(config)
    return redirect(url_for('index'))

@app.route('/update_swear_warning', methods=['POST'])
@login_required
def update_swear_warning():
    config = load_config()
    warning = request.form.get('warning')
    
    if not 'moderation' in config:
        config['moderation'] = {}
    
    config['moderation']['warning'] = warning
    
    save_config(config)
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        users_data = load_users()
        user = users_data['users'].get(username)
        
        if user and verify_password(user['password'], password):
            session['logged_in'] = True
            session['username'] = username
            if remember:
                session.permanent = True
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/users')
@admin_required
def users():
    users_data = load_users()
    return render_template('users.html', 
                         users=users_data['users'],
                         current_user=session.get('username'))

@app.route('/add_user', methods=['POST'])
@admin_required
def add_user():
    username = request.form.get('username')
    password = request.form.get('password')
    role = request.form.get('role')
    
    users_data = load_users()
    
    if username in users_data['users']:
        flash('Username already exists', 'danger')
    else:
        users_data['users'][username] = {
            'password': hash_password(password),
            'role': role,
            'created_at': '2025-01-01T09:26:12+01:00'  # Using the provided time
        }
        save_users(users_data)
        flash('User added successfully', 'success')
    
    return redirect(url_for('users'))

@app.route('/delete_user', methods=['POST'])
@admin_required
def delete_user():
    username = request.form.get('username')
    users_data = load_users()
    
    if username == session.get('username'):
        flash('Cannot delete your own account', 'danger')
    elif username in users_data['users']:
        del users_data['users'][username]
        save_users(users_data)
        flash('User deleted successfully', 'success')
    
    return redirect(url_for('users'))

@app.route('/change_user_password', methods=['POST'])
@admin_required
def change_user_password():
    username = request.form.get('username')
    new_password = request.form.get('new_password')
    
    users_data = load_users()
    if username in users_data['users']:
        users_data['users'][username]['password'] = hash_password(new_password)
        save_users(users_data)
        flash('Password changed successfully', 'success')
    else:
        flash('User not found', 'danger')
    
    return redirect(url_for('users'))

@app.route('/change_own_password', methods=['POST'])
@login_required
def change_own_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if new_password != confirm_password:
        flash('New passwords do not match', 'danger')
        return redirect(url_for('users'))
    
    users_data = load_users()
    username = session.get('username')
    user = users_data['users'].get(username)
    
    if user and verify_password(user['password'], current_password):
        users_data['users'][username]['password'] = hash_password(new_password)
        save_users(users_data)
        flash('Your password has been changed successfully', 'success')
    else:
        flash('Current password is incorrect', 'danger')
    
    return redirect(url_for('users'))

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))

def init_admin_user():
    users_data = load_users()
    if not users_data.get('users'):
        users_data['users'] = {}
    
    if 'admin' not in users_data['users']:
        users_data['users']['admin'] = {
            'password': hash_password('admin'),
            'role': 'admin',
            'created_at': '2025-01-01T09:31:25+01:00'
        }
        save_users(users_data)
        print("Admin user initialized successfully")

# Moderation stats storage
moderation_stats = {
    'warnings': 0,
    'deletions': 0,
    'timeouts': 0,
    'active_violations': 0
}

recent_actions = []

@app.route('/moderation')
@login_required
@admin_required
def moderation_dashboard():
    # Load current settings and word lists from the moderator
    moderator = None  # bot.moderator if bot else None
    if not moderator:
        flash('Bot is not initialized', 'error')
        return redirect(url_for('index'))
        
    return render_template('moderation.html',
        stats=moderation_stats,
        settings={},
        word_lists={},
        recent_actions=recent_actions
    )

@app.route('/moderation/word_list/<category>', methods=['POST'])
@login_required
@admin_required
def update_word_list(category):
    moderator = bot.moderator if bot else None
    if not moderator:
        flash('Bot is not initialized', 'error')
        return redirect(url_for('moderation_dashboard'))
        
    words = request.form.get('words', '').split('\n')
    words = [w.strip() for w in words if w.strip()]
    
    # Update word list in moderator
    moderator.word_lists[category] = set(words)
    
    # If updating custom words, also update config
    if category == 'custom':
        config = load_config()
        config['moderation']['swear_words'] = words
        save_config(config)
        # Reload config to apply changes
        moderator.reload_config()
    
    flash(f'{category.title()} word list updated successfully', 'success')
    return redirect(url_for('moderation_dashboard'))

@app.route('/moderation/settings', methods=['POST'])
@login_required
@admin_required
def update_moderation_settings():
    moderator = bot.moderator if bot else None
    if not moderator:
        flash('Bot is not initialized', 'error')
        return redirect(url_for('moderation_dashboard'))
        
    # Update severity levels
    new_settings = {
        'severity_levels': {
            'low': float(request.form.get('low_threshold', 0.3)),
            'medium': float(request.form.get('medium_threshold', 0.6)),
            'high': float(request.form.get('high_threshold', 0.8))
        },
        'auto_moderation': {
            'max_violations': int(request.form.get('max_violations', 3)),
            'timeout_duration': int(request.form.get('timeout_duration', 300)),
            'delete_threshold': float(request.form.get('delete_threshold', 0.7))
        }
    }
    
    # Save to config file
    config = load_config()
    config['moderation']['settings'].update(new_settings)
    save_config(config)
    
    # Reload config to apply changes immediately
    moderator.reload_config()
    
    flash('Moderation settings updated successfully', 'success')
    return redirect(url_for('moderation_dashboard'))

def update_moderation_stats(action):
    """Update moderation statistics"""
    if action == 'warn':
        moderation_stats['warnings'] += 1
    elif action == 'delete':
        moderation_stats['deletions'] += 1
    elif action == 'timeout':
        moderation_stats['timeouts'] += 1
        
    # Update active violations count
    # if bot and bot.moderator:
    #     moderation_stats['active_violations'] = len(bot.moderator.user_violations)

def log_moderation_action(user, action, severity, categories):
    """Log a moderation action"""
    recent_actions.insert(0, {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user': user,
        'action': action,
        'severity': severity,
        'categories': [cat for cat, triggered in categories.items() if triggered]
    })
    
    # Keep only last 100 actions
    if len(recent_actions) > 100:
        recent_actions.pop()
    
    # Update stats
    update_moderation_stats(action)

if __name__ == '__main__':
    init_admin_user()  # Initialize admin user if not exists
    # Use threaded=False to avoid threading issues
    app.run(host='0.0.0.0', port=3001, debug=False, threaded=False)
