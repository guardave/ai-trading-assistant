# AI Trading Assistant

An intelligent, modular trading assistant system with multi-LLM support, multi-channel interfaces, and plugin-based strategy architecture.

## Features

- **Multi-LLM Support**: Pluggable AI providers (Grok, OpenAI, Claude, Ollama)
- **Multi-Channel Interface**: Telegram, Discord, Web UI, REST API
- **Plugin-Based Strategies**: Extensible scanner plugins (VCP, Pivot, AI-powered)
- **Multi-Currency Portfolio**: USD, HKD, JPY with automatic conversion
- **Real-Time Monitoring**: Price alerts, stop/target notifications
- **Flexible Storage**: SQLite (dev) → PostgreSQL (production)

## Quick Start

### Prerequisites

- Python 3.11+
- Telegram bot token
- At least one AI provider API key (Grok, OpenAI, or Claude)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-trading-assistant.git
cd ai-trading-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run
python -m src.main
```

### Docker

```bash
# Build and run
docker compose up -d

# View logs
docker compose logs -f app
```

## Documentation

| Document | Description |
|----------|-------------|
| [Business Requirements](docs/01-business-requirements.md) | Business objectives and functional requirements |
| [System Requirements](docs/02-system-requirements.md) | Technical requirements in TDD format |
| [System Design](docs/03-system-design.md) | Architecture and component design |
| [Configuration Guide](docs/04-configuration-usage-guide.md) | Configuration options and usage |
| [Installation Guide](docs/05-installation-deployment-guide.md) | Installation and deployment instructions |
| [Test Strategy](docs/06-test-strategy.md) | Testing approach and methodology |
| [Test Plan](docs/07-test-plan.md) | Detailed test cases |

## Project Structure

```
ai-trading-assistant/
├── src/                    # Application source code
│   ├── ai/                 # AI engine and providers
│   ├── data/               # Data providers
│   ├── scanner/            # Strategy scanners
│   ├── portfolio/          # Portfolio management
│   ├── bot/                # Bot interfaces
│   ├── api/                # REST API
│   ├── storage/            # Database abstraction
│   └── utils/              # Utilities
├── tests/                  # Test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── docker-compose.yaml     # Docker deployment
```

## Configuration

Key configuration in `config/config.yaml`:

```yaml
ai:
  default_provider: "grok"
  fallback_order: ["grok", "openai", "claude"]

interfaces:
  telegram:
    enabled: true
  rest_api:
    enabled: true
    port: 8000

portfolio:
  base_currency: "USD"
  max_risk_per_trade_pct: 2.0
```

See [Configuration Guide](docs/04-configuration-usage-guide.md) for full options.

## Commands (Telegram)

| Command | Description |
|---------|-------------|
| `/portfolio` | Show all positions |
| `/add SYMBOL SHARES ENTRY STOP TARGET` | Add position |
| `/close SYMBOL` | Close position |
| `/scan` | Run market scan |
| `/watchlist` | Show watchlist |
| `/help` | Show help |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read the documentation and follow the code style.

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading involves risk. Use at your own risk.
