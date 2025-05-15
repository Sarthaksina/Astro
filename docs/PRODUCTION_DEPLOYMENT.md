# Cosmic Market Oracle - Production Deployment Guide

This guide provides detailed instructions for deploying the Cosmic Market Oracle in a production environment. It covers system requirements, security considerations, deployment steps, and monitoring setup.

## System Requirements

### Hardware Requirements
- **API Server**: 
  - Minimum: 2 CPU cores, 4GB RAM
  - Recommended: 4 CPU cores, 8GB RAM
- **Database Server**:
  - Minimum: 2 CPU cores, 8GB RAM
  - Recommended: 4 CPU cores, 16GB RAM
- **Storage**:
  - Minimum: 50GB SSD
  - Recommended: 100GB+ SSD with backup solution

### Software Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- Nginx (for production deployments)
- SSL certificates for secure communication

## Pre-deployment Checklist

1. **Environment Variables**:
   - Create a `.env` file based on the `.env.example` template
   - Set secure passwords for all database users
   - Configure API keys and secrets

2. **Security Considerations**:
   - Restrict CORS origins in production
   - Enable API key authentication
   - Set up proper firewall rules
   - Configure SSL certificates

3. **Database Setup**:
   - Ensure TimescaleDB is properly configured
   - Set up database backups
   - Configure database retention policies

## Deployment Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cosmic-market-oracle.git
cd cosmic-market-oracle
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env file with your production settings
```

Key environment variables to configure:
- `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
- `API_KEY` (set a strong, random value)
- `API_KEY_REQUIRED=true` (enable API key authentication)
- `ALLOWED_ORIGINS` (comma-separated list of allowed origins)
- `ENVIRONMENT=production`

### 3. SSL Certificate Setup

Place your SSL certificates in the `nginx/ssl` directory:
```bash
mkdir -p nginx/ssl
# Copy your SSL certificates to this directory
cp your-cert.crt nginx/ssl/cosmic-market-oracle.crt
cp your-key.key nginx/ssl/cosmic-market-oracle.key
```

### 4. Build and Start the Services

```bash
# Run the deployment script
python scripts/deploy/deploy_production.py
```

Alternatively, you can run the Docker Compose commands manually:
```bash
docker-compose build
docker-compose up -d
```

### 5. Verify Deployment

Check that all services are running:
```bash
docker-compose ps
```

Verify the API is accessible:
```bash
curl -k https://your-domain.com/api/health
```

## Monitoring Setup

The Cosmic Market Oracle includes a comprehensive monitoring stack based on Prometheus, Grafana, and Loki.

### 1. Start the Monitoring Stack

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access the Monitoring Dashboards

- **Grafana**: https://your-domain.com/grafana (default credentials: admin/admin)
- **Prometheus**: https://your-domain.com/prometheus
- **Jaeger UI**: https://your-domain.com/jaeger

### 3. Configure Alerts

1. Navigate to Grafana
2. Go to Alerting > Notification channels
3. Set up email, Slack, or other notification channels
4. Configure alert rules based on your requirements

## Backup and Recovery

### Database Backups

Set up automated backups for the PostgreSQL/TimescaleDB database:

```bash
# Create a backup script
cat > backup.sh << 'EOF'
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=/path/to/backups
mkdir -p $BACKUP_DIR

# Backup the database
docker exec cosmic-market-oracle-db pg_dump -U postgres -d cosmic_market_oracle | gzip > $BACKUP_DIR/cosmic_market_oracle_$TIMESTAMP.sql.gz

# Rotate backups (keep last 7 days)
find $BACKUP_DIR -name "cosmic_market_oracle_*.sql.gz" -mtime +7 -delete
EOF

chmod +x backup.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /path/to/backup.sh") | crontab -
```

### Model Backups

MLflow artifacts are stored in MinIO and should be backed up regularly:

```bash
# Install the MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc alias set minio http://localhost:9000 minioadmin minioadmin

# Backup MLflow artifacts
./mc mirror minio/mlflow /path/to/mlflow-backup
```

## Scaling Considerations

### Horizontal Scaling

To handle increased load, you can scale the API service horizontally:

1. Update the `docker-compose.yml` file to increase the number of API replicas:
   ```yaml
   services:
     api:
       deploy:
         replicas: 3
   ```

2. Ensure Nginx is configured for load balancing:
   ```nginx
   upstream api_servers {
       server api:8000;
       server api:8000;
       server api:8000;
   }
   
   location /api/ {
       proxy_pass http://api_servers/;
       # Other proxy settings...
   }
   ```

### Database Scaling

For high-throughput scenarios, consider:

1. Increasing database resources
2. Setting up TimescaleDB replication
3. Implementing database sharding for historical data

## Troubleshooting

### Common Issues

1. **API Service Not Starting**:
   - Check logs: `docker-compose logs api`
   - Verify environment variables
   - Ensure database is accessible

2. **Database Connection Issues**:
   - Verify database credentials
   - Check network connectivity
   - Ensure TimescaleDB extensions are installed

3. **High CPU/Memory Usage**:
   - Check monitoring dashboards
   - Identify resource-intensive operations
   - Consider scaling resources or optimizing code

### Accessing Logs

```bash
# View API logs
docker-compose logs -f api

# View database logs
docker-compose logs -f db

# View all logs
docker-compose logs -f
```

## Security Best Practices

1. **Regular Updates**:
   - Keep all dependencies updated
   - Apply security patches promptly

2. **Access Control**:
   - Use least privilege principle
   - Implement proper authentication and authorization
   - Rotate API keys regularly

3. **Network Security**:
   - Use HTTPS for all communications
   - Implement proper firewall rules
   - Consider using a VPN for administrative access

4. **Monitoring and Alerting**:
   - Set up alerts for suspicious activities
   - Monitor failed authentication attempts
   - Regularly review logs for security issues

## Performance Tuning

1. **API Performance**:
   - Adjust worker count based on CPU cores
   - Implement caching for frequent requests
   - Optimize database queries

2. **Database Performance**:
   - Configure proper indexes
   - Adjust TimescaleDB chunk intervals
   - Implement query optimization

3. **Monitoring Performance**:
   - Adjust scrape intervals
   - Implement log rotation
   - Configure proper retention policies

## Conclusion

Following this guide will help you deploy a robust, secure, and scalable Cosmic Market Oracle in a production environment. For additional support or questions, please refer to the project documentation or contact the development team.
