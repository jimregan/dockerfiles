#!/bin/bash
# Egress allowlist for the Claude Code container, adapted from Anthropic's
# reference devcontainer. Domains are read from a read-only mounted file so the
# agent cannot widen its own access. IPs are resolved at start-up: if a CDN
# rotates addresses, restart the container.
set -euo pipefail
IFS=$'\n\t'

DOMAINS_FILE=/etc/claude-firewall/allowed-domains.txt

# Keep Docker's embedded DNS NAT rules.
DOCKER_DNS_RULES=$(iptables-save -t nat | grep "127\.0\.0\.11" || true)

iptables -F; iptables -X
iptables -t nat -F; iptables -t nat -X
iptables -t mangle -F; iptables -t mangle -X
ipset destroy allowed-domains 2>/dev/null || true

if [ -n "$DOCKER_DNS_RULES" ]; then
    iptables -t nat -N DOCKER_OUTPUT 2>/dev/null || true
    iptables -t nat -N DOCKER_POSTROUTING 2>/dev/null || true
    echo "$DOCKER_DNS_RULES" | xargs -L 1 iptables -t nat
fi

# DNS, loopback
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A INPUT  -p udp --sport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
iptables -A INPUT  -p tcp --sport 53 -j ACCEPT
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

ipset create allowed-domains hash:net

# GitHub ranges (web, api, git)
gh_meta=$(curl -fsS --max-time 15 https://api.github.com/meta || true)
if [ -n "$gh_meta" ]; then
    echo "$gh_meta" | jq -r '(.web + .api + .git)[]' \
        | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/[0-9]+$' \
        | while read -r cidr; do ipset add -exist allowed-domains "$cidr"; done
else
    echo "WARN: could not fetch GitHub meta; GitHub only via domains file" >&2
fi

# Domains from file (one per line, # comments allowed)
grep -vE '^\s*(#|$)' "$DOMAINS_FILE" | while read -r domain; do
    domain=$(echo "$domain" | xargs)
    ips=$(dig +short A "$domain" | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true)
    if [ -z "$ips" ]; then
        echo "WARN: no A records for $domain" >&2
        continue
    fi
    for ip in $ips; do ipset add -exist allowed-domains "$ip"; done
done

# Every attached Docker network (backend: worker, llm)
for cidr in $(ip -o -f inet addr show | awk '$2 != "lo" {print $4}'); do
    net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
    iptables -A INPUT  -s "$net" -j ACCEPT
    iptables -A OUTPUT -d "$net" -j ACCEPT
done

iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m set --match-set allowed-domains dst -j ACCEPT
iptables -A OUTPUT -j REJECT --reject-with icmp-admin-prohibited

# Verify
if curl -fsS --max-time 5 https://example.com >/dev/null 2>&1; then
    echo "ERROR: firewall check failed (example.com reachable)" >&2; exit 1
fi
if ! curl -sS --max-time 10 -o /dev/null https://api.anthropic.com; then
    echo "ERROR: api.anthropic.com unreachable" >&2; exit 1
fi
echo "firewall: ok"
