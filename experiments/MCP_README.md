# Crime Prediction MCP Server

This MCP (Model Context Protocol) server provides a tool to make crime predictions using the trained XGBoost model.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r v-xgboost/requirements.txt
   ```

2. **Ensure model files exist:**
   Make sure you have these files in `experiments/v-xgboost/`:
   - `predict.py` (prediction script)
   - `crime_prediction_model.json` (trained model)
   - `label_encoder.pkl` (label encoder)
   
   If missing, train the model first:
   ```bash
   python v-xgboost/train.py
   ```

3. **Configure Azure OpenAI (for LLM summaries):**
   Create a `.env` file in the `experiments/` folder with:
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   AZURE_OPENAI_KEY=your_api_key
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   ```

## Running the Server

```bash
python mcp_server.py
```

The server will start and listen for MCP protocol messages on stdin/stdout.

## Testing the Server

To test the server, you can run the `test_mcp.py` script. This script sends a sample prediction request to the server and prints the response.

```bash
python test_mcp.py
```

## Available Tools

### predict_crime

Makes a crime prediction for a given date, time, and location.

**Parameters:**
- `date` (string): Date in MM-DD format (e.g., "12-01")
- `time` (string): Time in HH:MM format (e.g., "12:00") 
- `latitude` (number): Latitude coordinate (e.g., 46.0)
- `longitude` (number): Longitude coordinate (e.g., 5.5)

**Example Usage:**
```json
{
  "date": "12-01",
  "time": "12:00", 
  "latitude": 46,
  "longitude": 5.5
}
```

**Output:**
The tool returns the predicted crime type and, if Azure OpenAI is configured, an LLM-generated summary with practical safety advice.

## Integration with MCP Clients

This server can be used with any MCP-compatible client like:
- Custom MCP applications
- VS Code extensions that support MCP

Add the server to your MCP client configuration by pointing to the `mcp_server.py` file.

## Troubleshooting

1. **Import errors:** Make sure you've run `pip install -r v-xgboost/requirements.txt` to install dependencies
2. **Model not found:** Ensure the model files exist by running the training script
3. **Prediction fails:** Check that all required Python packages are installed
4. **No LLM summary:** Verify your Azure OpenAI configuration in the `.env` file

## Command Line Equivalent

The MCP tool essentially wraps this command:
```bash
python v-xgboost/predict.py 12-01 12:00 46 5.5
```