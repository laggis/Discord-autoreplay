# AI Discord Bot with Web Dashboard

A powerful Discord bot with AI capabilities, user management, and a modern web dashboard. Features include chat moderation, user management, and real-time configuration.

## 🌟 Features

### 🤖 AI Integration
- Natural language processing for chat interactions
- Smart response generation
- Learning system for improved responses

### 👥 User Management
- Secure user authentication
- Role-based access control (Admin/User)
- Password management with secure hashing
- Session management with "Remember Me" functionality

### 🛡️ Content Moderation
- Real-time message moderation
- Multiple word list categories:
  - Profanity
  - Hate speech
  - Mild profanity
  - Custom words
- Configurable severity levels
- Progressive punishment system
- Moderation action logging

### 🎛️ Web Dashboard
- Modern, responsive design with Bootstrap
- Real-time statistics
- User management interface
- Moderation control panel
- Configuration management

## 📋 Requirements

```
Python 3.8+
Discord.py
Flask
PyYAML
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/discord-ai-bot.git
cd discord-ai-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the bot**
- Copy `config.yaml.example` to `config.yaml`
- Add your Discord bot token
- Configure other settings as needed

4. **Start the bot and web server**
```bash
python web_config.py
```

5. **Access the dashboard**
- Open http://localhost:5006 in your browser
- Login with default credentials:
  - Username: admin
  - Password: admin
- Change the default password immediately!

## 🔧 Configuration

### Discord Bot Settings
```yaml
discord:
  token: "YOUR_BOT_TOKEN"
  prefix: "!"
```

### Moderation Settings
```yaml
moderation:
  enabled: true
  settings:
    severity_levels:
      low: 0.3    # Mild profanity
      medium: 0.6  # Regular profanity
      high: 0.8    # Hate speech
    auto_moderation:
      max_violations: 3
      timeout_duration: 300
```

## 🛡️ Security Features

- Password hashing with PBKDF2
- Session encryption
- CSRF protection
- Role-based access control
- Secure configuration management

## 📊 Dashboard Features

### User Management
- Add/remove users
- Change passwords
- Assign roles
- View user activity

### Moderation Dashboard
- Real-time statistics
- Word list management
- Configure severity levels
- View moderation logs
- Hot-reload configuration

## 🔍 Monitoring

The bot includes comprehensive logging:
- Moderation actions
- User activities
- System events
- Configuration changes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Discord.py for the excellent Discord API wrapper
- Flask for the web framework
- Bootstrap for the UI components

## 🔮 Future Plans

- [ ] Multi-language support
- [ ] Advanced AI training capabilities
- [ ] Custom plugin system
- [ ] API integration
- [ ] Enhanced analytics

## ⚠️ Important Notes

- Always change the default admin password
- Keep your Discord bot token secure
- Regularly backup your configuration
- Monitor the moderation logs

## 🆘 Support

For support:
1. Check the [Issues](https://github.com/yourusername/discord-ai-bot/issues) page
2. Create a new issue if needed
3. Join our Discord server for community support

## 🔄 Updates

The bot supports hot-reloading of configurations:
- Word list changes apply instantly
- Severity thresholds update in real-time
- Moderation settings can be changed without restart
- User management changes take effect immediately
