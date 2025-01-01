from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import yaml
import os
from functools import wraps
from datetime import datetime, timedelta
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session encryption
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # For "remember me" functionality

# Admin credentials - Change these to secure values
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

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
        if isinstance(data, dict):
            response = data.get('response', '')
        else:
            response = data
            
        if keyword.lower() in message.lower():
            return response
    
    # Check learned responses
    for question, data in config.get('learning', {}).get('responses', {}).items():
        if question.lower() in message.lower():
            return data.get('answer', '')
    
    return "Jag förstår inte frågan. Kan du omformulera den?"

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
    response = request.form.get('response')
    tags = request.form.get('tags', '').split(',')
    tags = [tag.strip() for tag in tags if tag.strip()]
    cooldown = int(request.form.get('cooldown', 5))
    
    if keyword and response:
        if 'keywords' not in config:
            config['keywords'] = {}
        config['keywords'][keyword] = {
            'response': response,
            'tags': tags,
            'cooldown': cooldown,
            'uses': 0,
            'last_used': None
        }
        save_config(config)
    
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
        
        # Check credentials (replace with your actual authentication logic)
        if username == "admin" and password == "admin":
            session['logged_in'] = True
            if remember:
                session.permanent = True  # Uses permanent_session_lifetime from app config
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
