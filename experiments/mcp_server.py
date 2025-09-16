#!/usr/bin/env python3
"""
MCP Server for Crime Prediction

This server provides a tool to make crime predictions using the trained XGBoost model.
"""
import asyncio
import os
import sys
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
import importlib.util

# Add the mcp module to the path
sys.path.append(str(Path(__file__).parent))

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolRequest,
        CallToolResult,
        ListToolsRequest,
        TextContent,
        Tool,
    )
except ImportError:
    print("Error: mcp package not found. Install with: pip install mcp")
    sys.exit(1)


class CrimePredictionServer:
    def __init__(self):
        self.server = Server("crime-prediction-server", version="1.0.0")
        self.predict_script_path = Path(__file__).parent / "v-xgboost" / "predict.py"
        self.model_dir = self.predict_script_path.parent
        self.predict_crime_func = self._load_predict_function()
        self.predict_crime_area_func = self._load_predict_area_function()

        # Register handlers using decorators
        self._register_handlers()

    def _load_predict_function(self):
        """Dynamically load the predict function from the script."""
        spec = importlib.util.spec_from_file_location("predict", self.predict_script_path)
        if spec and spec.loader:
            predict_module = importlib.util.module_from_spec(spec)
            # Add the module's directory to the path to resolve its own imports
            sys.path.append(str(self.predict_script_path.parent))
            spec.loader.exec_module(predict_module)
            # Remove it after loading to keep sys.path clean
            sys.path.pop()
            return predict_module.predict
        raise ImportError(f"Could not load predict function from {self.predict_script_path}")
    
    def _load_predict_area_function(self):
        """Dynamically load the predict_area function from the script."""
        spec = importlib.util.spec_from_file_location("predict_area", self.predict_script_path)
        if spec and spec.loader:
            predict_module = importlib.util.module_from_spec(spec)
            # Add the module's directory to the path to resolve its own imports
            sys.path.append(str(self.predict_script_path.parent))
            spec.loader.exec_module(predict_module)
            # Remove it after loading to keep sys.path clean
            sys.path.pop()
            return predict_module.predict_area
        raise ImportError(f"Could not load predict area function from {self.predict_script_path}")
    
    def _register_handlers(self):
        """Register MCP handlers using decorators."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Handle tools/list requests."""
            return [
                Tool(
                    name="predict_crime",
                    description="Make a crime prediction using the trained XGBoost model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in MM-DD format (e.g., '12-01')",
                                "pattern": r"^\d{2}-\d{2}$"
                            },
                            "time": {
                                "type": "string", 
                                "description": "Time in HH:MM format (e.g., '12:00')",
                                "pattern": r"^\d{2}:\d{2}$"
                            },
                            "latitude": {
                                "type": "number",
                                "description": "Latitude coordinate (e.g., 46.0)"
                            },
                            "longitude": {
                                "type": "number", 
                                "description": "Longitude coordinate (e.g., 5.5)"
                            }
                        },
                        "required": ["date", "time", "latitude", "longitude"]
                    }
                ),
                Tool(
                    name="predict_crime_area",
                    description="Make a crime prediction in a specific area using the trained XGBoost model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date in MM-DD format (e.g., '12-01')",
                                "pattern": r"^\d{2}-\d{2}$"
                            },
                            "startTime": {
                                "type": "string", 
                                "description": "Start Time in HH:MM format (e.g., '12:00')",
                                "pattern": r"^\d{2}:\d{2}$"
                            },
                            "endTime": {
                                "type": "string", 
                                "description": "End Time in HH:MM format (e.g., '12:00')",
                                "pattern": r"^\d{2}:\d{2}$"
                            },
                            "southWestLat": {
                                "type": "number",
                                "description": "Southwest latitude coordinate (e.g., 46.0)"
                            },
                            "southWestLng": {
                                "type": "number", 
                                "description": "Southwest longitude coordinate (e.g., 5.5)"
                            },
                            "northEastLat": {
                                "type": "number",
                                "description": "Northeast latitude coordinate (e.g., 46.0)"
                            },
                            "northEastLng": {
                                "type": "number", 
                                "description": "Northeast longitude coordinate (e.g., 5.5)"
                            }
                        },
                        "required": ["date", "startTime", "endTime", "southWestLat", "southWestLng", "northEastLat", "northEastLng"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            """Handle tools/call requests."""
            if name == "predict_crime":
                try:
                    result = await self._predict_crime(arguments)
                    return result
                except Exception as e:
                    return {
                        "content": [{"type": "text", "text": f"Error in tool handler: {str(e)}"}],
                        "isError": True
                    }
            elif name == "predict_crime_area":
                try:
                    result = await self._predict_crime_area(arguments)
                    return result
                except Exception as e:
                    return {
                        "content": [{"type": "text", "text": f"Error in tool handler: {str(e)}"}],
                        "isError": True
                    }
            else:
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True
                }

    async def _predict_crime(self, arguments: Dict[str, Any]):
        """Make a crime prediction using the trained model."""
        try:
            # Validate arguments
            date = arguments.get("date")
            time = arguments.get("time") 
            latitude = arguments.get("latitude")
            longitude = arguments.get("longitude")
            
            print(f"DEBUG: Received arguments: {arguments}", file=sys.stderr)

            if not all([date, time, latitude is not None, longitude is not None]):
                return {
                    "content": [{"type": "text", "text": "Error: Missing required parameters. Need date (MM-DD), time (HH:MM), latitude, and longitude."}],
                    "isError": True
                }

            # Validate date format
            if not isinstance(date, str) or len(date.split('-')) != 2:
                return {
                    "content": [{"type": "text", "text": "Error: Date must be in MM-DD format (e.g., '12-01')"}],
                    "isError": True
                }

            # Validate time format
            if not isinstance(time, str) or len(time.split(':')) != 2:
                return {
                    "content": [{"type": "text", "text": "Error: Time must be in HH:MM format (e.g., '12:00')"}],
                    "isError": True
                }

            # Run the prediction function
            prediction_result = self.predict_crime_func(
                date_input=date,
                time_input=time,
                latitude=latitude,
                longitude=longitude,
                model_dir=str(self.model_dir)
            )

            # Return the prediction result
            return {
                "content": [{"type": "text", "text": f"Crime Prediction Result:\n\n{prediction_result}"}]
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error making prediction: {str(e)}"}],
                "isError": True
            }
        
    async def _predict_crime_area(self, arguments: Dict[str, Any]):
            """Make a crime prediction in an area using the trained model."""
            try:
                # Validate arguments
                date = arguments.get("date")
                startTime = arguments.get("startTime")
                endTime = arguments.get("endTime")
                southWestLat = arguments.get("southWestLat")
                southWestLng = arguments.get("southWestLng")
                northEastLat = arguments.get("northEastLat")
                northEastLng = arguments.get("northEastLng")

                if not all([date, startTime, endTime, southWestLat is not None, southWestLng is not None, northEastLat is not None, northEastLng is not None]):
                    return {
                        "content": [{"type": "text", "text": "Error: Missing required parameters. Need date (MM-DD), startTime (HH:MM), endTime (HH:MM), southWestLat, southWestLng, northEastLat, and northEastLng."}],
                        "isError": True
                    }

                # Validate date format
                if not isinstance(date, str) or len(date.split('-')) != 2:
                    return {
                        "content": [{"type": "text", "text": "Error: Date must be in MM-DD format (e.g., '12-01')"}],
                        "isError": True
                    }

                # Validate time format
                if not isinstance(startTime, str) or len(startTime.split(':')) != 2:
                    return {
                        "content": [{"type": "text", "text": "Error: Time must be in HH:MM format (e.g., '12:00')"}],
                        "isError": True
                    }
                if not isinstance(endTime, str) or len(endTime.split(':')) != 2:
                    return {
                        "content": [{"type": "text", "text": "Error: Time must be in HH:MM format (e.g., '12:00')"}],
                        "isError": True
                    }

                # Run the prediction function
                prediction_result = self.predict_crime_area_func(
                    date=date,
                    startTime=startTime,
                    endTime=endTime,
                    southWestLat=southWestLat,
                    southWestLng=southWestLng,
                    northEastLat=northEastLat,
                    northEastLng=northEastLng,
                    model_dir=str(self.model_dir)
                )

                # Return the prediction result
                return {
                    "content": [{"type": "text", "text": f"Area Crime Prediction Result:\n\n{prediction_result}"}]
                }

            except Exception as e:
                return {
                    "content": [{"type": "text", "text": f"Error making prediction: {str(e)}"}],
                    "isError": True
                }

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = CrimePredictionServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())