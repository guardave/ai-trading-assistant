# Installation and Deployment Guide

## AI Trading Assistant

**Version:** 1.0.0
**Date:** 2025-11-28
**Status:** Draft

---

## 1. Overview

This guide covers installation and deployment options for the AI Trading Assistant, from local development to production deployment using Docker.

### 1.1 Deployment Options

| Option | Use Case | Complexity | Cost |
|--------|----------|------------|------|
| Local Development | Development, testing | Low | Free |
| Docker (SQLite) | Single user, low volume | Medium | Free |
| Docker Compose (Full) | Production, multi-user | Medium | ~$10/month |

---

## 2. Prerequisites

### 2.1 System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 1 core | 2 cores |
| RAM | 512 MB | 1 GB |
| Disk | 1 GB | 5 GB |
| Python | 3.10+ | 3.11+ |
| Docker | 24.0+ | Latest |

### 2.2 Required Accounts

1. **Telegram Bot** (Required)
   - Create bot via [@BotFather](https://t.me/BotFather)
   - Get bot token and your chat ID

2. **AI Provider** (At least one)
   - [xAI Grok](https://x.ai/) - API key
   - [OpenAI](https://platform.openai.com/) - API key
   - [Anthropic Claude](https://console.anthropic.com/) - API key

3. **Data Providers** (Optional but recommended)
   - [EODHD](https://eodhd.com/) - Paid, reliable
   - [Finnhub](https://finnhub.io/) - Free tier available

---

## 3. Local Development Installation

### 3.1 Clone Repository

```bash
git clone https://github.com/yourusername/ai-trading-assistant.git
cd ai-trading-assistant
```

### 3.2 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.4 Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env  # or use your preferred editor
```

Required variables in `.env`:
```bash
# Telegram (Required)
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id

# AI Provider (At least one)
GROK_API_KEY=your-grok-key
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-claude-key

# Data Providers (Optional)
FINNHUB_API_KEY=your-finnhub-key
# EODHD_API_KEY=your-eodhd-key
```

### 3.5 Initialize Database

```bash
# Create data directory
mkdir -p data

# Initialize database (SQLite)
python -m src.scripts.init_db
```

### 3.6 Run Application

```bash
# Run the application
python -m src.main
```

### 3.7 Verify Installation

1. Open Telegram and message your bot with `/health`
2. Expected response: System health status showing all components healthy

---

## 4. Docker Installation

### 4.1 Quick Start (SQLite)

This is the simplest production deployment using SQLite (no external database required).

**Step 1: Build Image**
```bash
docker build -t ai-trading-assistant .
```

**Step 2: Create .env file**
```bash
# Create .env with your credentials
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
GROK_API_KEY=your-grok-key
FINNHUB_API_KEY=your-finnhub-key
EOF
```

**Step 3: Run Container**
```bash
docker run -d \
  --name trading-assistant \
  --restart unless-stopped \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config:ro \
  -p 8000:8000 \
  ai-trading-assistant
```

**Step 4: Verify**
```bash
# Check logs
docker logs -f trading-assistant

# Test health endpoint
curl http://localhost:8000/health
```

### 4.2 Docker Compose (Full Stack)

For production deployment with PostgreSQL and Redis.

**Step 1: Create docker-compose.yml**
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: trading-assistant
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - GROK_API_KEY=${GROK_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - FINNHUB_API_KEY=${FINNHUB_API_KEY}
      - EODHD_API_KEY=${EODHD_API_KEY}
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/trading
      - REDIS_URL=redis://redis:6379/0
      - API_KEY=${API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  postgres:
    image: postgres:15-alpine
    container_name: trading-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=trading
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: trading-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

**Step 2: Create .env file**
```bash
cat > .env << 'EOF'
# Telegram
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id

# AI Providers
GROK_API_KEY=your-grok-key
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Data Providers
FINNHUB_API_KEY=your-finnhub-key
EODHD_API_KEY=

# Database
POSTGRES_PASSWORD=your-secure-password

# API
API_KEY=your-api-key
EOF
```

**Step 3: Start Services**
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f app
```

**Step 4: Verify**
```bash
# Check all services
docker compose ps

# Test health
curl http://localhost:8000/health
```

---

## 5. Cloud Deployment

### 5.1 VPS Deployment (DigitalOcean, Linode, etc.)

**Recommended Specs:**
- 1 vCPU, 1GB RAM Droplet (~$6/month)
- Ubuntu 22.04 LTS

**Step 1: SSH into server**
```bash
ssh root@your-server-ip
```

**Step 2: Install Docker**
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt install docker-compose-plugin -y

# Add user to docker group (optional)
usermod -aG docker $USER
```

**Step 3: Clone and Deploy**
```bash
# Clone repository
git clone https://github.com/yourusername/ai-trading-assistant.git
cd ai-trading-assistant

# Create .env file
nano .env

# Deploy
docker compose up -d
```

**Step 4: Configure Firewall**
```bash
# Allow SSH
ufw allow 22

# Allow API port (if needed externally)
ufw allow 8000

# Enable firewall
ufw enable
```

### 5.2 Railway Deployment

Railway provides easy container deployment with free tier.

**Step 1: Install Railway CLI**
```bash
npm install -g @railway/cli
```

**Step 2: Login and Initialize**
```bash
railway login
railway init
```

**Step 3: Add Environment Variables**
```bash
railway variables set TELEGRAM_BOT_TOKEN=your-token
railway variables set TELEGRAM_CHAT_ID=your-chat-id
railway variables set GROK_API_KEY=your-key
# ... add other variables
```

**Step 4: Deploy**
```bash
railway up
```

### 5.3 Fly.io Deployment

Fly.io offers simple container deployment with global edge network.

**Step 1: Install Fly CLI**
```bash
curl -L https://fly.io/install.sh | sh
```

**Step 2: Login and Initialize**
```bash
fly auth login
fly launch
```

**Step 3: Set Secrets**
```bash
fly secrets set TELEGRAM_BOT_TOKEN=your-token
fly secrets set TELEGRAM_CHAT_ID=your-chat-id
fly secrets set GROK_API_KEY=your-key
```

**Step 4: Deploy**
```bash
fly deploy
```

---

## 6. Configuration After Deployment

### 6.1 Update Configuration Files

Configuration files should be mounted as read-only volumes. To update:

```bash
# Edit local config
nano config/config.yaml

# Restart container to apply changes
docker compose restart app
```

### 6.2 Database Migrations

When upgrading to a new version with database changes:

```bash
# Run migrations
docker compose exec app python -m src.scripts.migrate

# Or for standalone container
docker exec trading-assistant python -m src.scripts.migrate
```

### 6.3 Backup Data

**SQLite Backup:**
```bash
# Create backup
docker exec trading-assistant cp /app/data/trading.db /app/data/backup_$(date +%Y%m%d).db

# Copy to host
docker cp trading-assistant:/app/data/backup_$(date +%Y%m%d).db ./backups/
```

**PostgreSQL Backup:**
```bash
# Create backup
docker compose exec postgres pg_dump -U postgres trading > backup_$(date +%Y%m%d).sql
```

### 6.4 Restore Data

**SQLite Restore:**
```bash
docker cp ./backups/backup_20251128.db trading-assistant:/app/data/trading.db
docker restart trading-assistant
```

**PostgreSQL Restore:**
```bash
docker compose exec -T postgres psql -U postgres trading < backup_20251128.sql
```

---

## 7. Monitoring and Maintenance

### 7.1 View Logs

```bash
# All logs
docker compose logs -f

# App logs only
docker compose logs -f app

# Last 100 lines
docker compose logs --tail 100 app
```

### 7.2 Health Checks

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check container status
docker compose ps
```

### 7.3 Resource Monitoring

```bash
# Container stats
docker stats

# Disk usage
docker system df
```

### 7.4 Updating

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose down
docker compose build
docker compose up -d

# Clean up old images
docker image prune -f
```

---

## 8. Troubleshooting

### 8.1 Container Won't Start

**Check logs:**
```bash
docker compose logs app
```

**Common issues:**
1. Missing environment variables
2. Invalid API keys
3. Port already in use

**Fix port conflict:**
```bash
# Find what's using port 8000
lsof -i :8000

# Or change port in docker-compose.yml
ports:
  - "8001:8000"
```

### 8.2 Database Connection Failed

**PostgreSQL:**
```bash
# Check if postgres is running
docker compose ps postgres

# Check postgres logs
docker compose logs postgres

# Test connection manually
docker compose exec postgres psql -U postgres -d trading
```

### 8.3 Bot Not Responding

1. **Verify bot token:**
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"
   ```

2. **Verify chat ID:**
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```

3. **Check app logs:**
   ```bash
   docker compose logs -f app | grep -i telegram
   ```

### 8.4 Memory Issues

```bash
# Check memory usage
docker stats --no-stream

# Increase memory limit in docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 1G
```

---

## 9. Security Checklist

### 9.1 Pre-Deployment

- [ ] All API keys stored in environment variables
- [ ] `.env` file not committed to git
- [ ] Strong database password generated
- [ ] API key configured for REST API
- [ ] Firewall configured (if exposed to internet)

### 9.2 Post-Deployment

- [ ] Health endpoint accessible
- [ ] Telegram bot responding to commands
- [ ] Logs not showing sensitive data
- [ ] Backup strategy in place
- [ ] Monitoring configured

### 9.3 Ongoing

- [ ] Regular security updates
- [ ] API key rotation (quarterly)
- [ ] Log review (weekly)
- [ ] Backup verification (monthly)

---

## 10. Quick Reference

### 10.1 Common Commands

| Action | Command |
|--------|---------|
| Start services | `docker compose up -d` |
| Stop services | `docker compose down` |
| Restart app | `docker compose restart app` |
| View logs | `docker compose logs -f app` |
| Check health | `curl localhost:8000/health` |
| Enter container | `docker compose exec app bash` |
| Rebuild | `docker compose build --no-cache` |

### 10.2 File Locations

| File | Purpose |
|------|---------|
| `.env` | Environment variables |
| `config/config.yaml` | Main configuration |
| `data/trading.db` | SQLite database |
| `logs/app.log` | Application logs |

### 10.3 Ports

| Port | Service |
|------|---------|
| 8000 | REST API |
| 5432 | PostgreSQL |
| 6379 | Redis |

---

## 11. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |
