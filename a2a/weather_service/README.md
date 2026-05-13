# Agents

The Weather Service Agent is an example of an [A2A](https://a2a-protocol.org/latest/) agent.

This agent depends on the Kagenti [Weather Tool](https://github.com/kagenti/agent-examples/tree/main/mcp/weather_tool).

This agent connects to the weather tool's MCP server.  By default (via the UI or `.env` files), it connects directly to the tool.

To route through the MCP Gateway instead, apply the gateway registration and update the agent's `MCP_URL`:

```bash
kubectl apply -f mcp/weather_tool/deployment/gateway.yaml
# Then set MCP_URL=http://mcp-gateway-istio.gateway-system.svc.cluster.local:8080/mcp on the agent
```

## Run the agent on Kubernetes with Kagenti

You may deploy using Kagenti's UI or through a Kubernetes manifest.

### Deploy using Kagenti's UI

Kagenti's UI is aware of this example agent.  To deploy through the UI

- Browse to http://kagenti-ui.localtest.me:8080/agents/
- Build from source
- Weather service agent
- Expand Environment Variables
  - Import from File/URL, URL, https://raw.githubusercontent.com/kagenti/agent-examples/refs/heads/main/a2a/weather_service/.env.openai
    - If using [Ollama](https://ollama.com/), instead of the default use https://raw.githubusercontent.com/kagenti/agent-examples/refs/heads/main/a2a/weather_service/.env.ollama
  - Fetch and parse
  - Import
- Build and deploy agent
- Chat
  - `What is the weather in New York?`

### Deploy using a Kubernetes deployment manifest

Deploy the sample manifest:

```bash
kubectl apply -f deployment/k8s.yaml
```
