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
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

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
        self.azure_maps_key = os.getenv("AZURE_MAPS_KEY")

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
                ),
                Tool(
                    name="reverse_geocode_azure_maps",
                    description="Get the location name from latitude and longitude using Azure Maps API.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "number",
                                "description": "Latitude coordinate (e.g., 48.24)"
                            },
                            "lng": {
                                "type": "number",
                                "description": "Longitude coordinate (e.g., 15.63)"
                            }
                        },
                        "required": ["lat", "lng"]
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
            elif name == "reverse_geocode_azure_maps":
                try:
                    result = await self._reverse_geocode_azure_maps(arguments)
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

    async def _reverse_geocode_azure_maps(self, arguments: Dict[str, Any]):
        """Get detailed location info from latitude/longitude using Azure Maps API."""
        import requests
        lat = arguments.get("lat")
        lng = arguments.get("lng")

        # Validate input
        if lat is None or lng is None:
            return {
                "content": [{"type": "text", "text": "Error: Missing required parameters lat and lng."}],
                "isError": True
            }
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            return {
                "content": [{"type": "text", "text": f"Error: Invalid coordinates ({lat}, {lng})."}],
                "isError": True
            }

        # Ensure API key is available
        if not self.azure_maps_key:
            return {
                "content": [{"type": "text", "text": "Error: AZURE_MAPS_KEY is not set in environment."}],
                "isError": True
            }

        try:
            url = (
                f"https://atlas.microsoft.com/search/address/reverse/json"
                f"?api-version=1.0&subscription-key={self.azure_maps_key}"
                f"&query={lat},{lng}"
            )
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return {
                    "content": [{"type": "text", "text": f"Azure Maps API error: {response.status_code} {response.text}"}],
                    "isError": True
                }

            data = response.json()
            if "addresses" not in data or not data["addresses"]:
                return {
                    "content": [{"type": "text", "text": "No address found for given coordinates."}],
                    "isError": True
                }

            addr = data["addresses"][0].get("address", {})
            freeform = addr.get("freeformAddress", "Unknown location")
            city = addr.get("municipality", "")
            country = addr.get("countrySubdivisionName", "")
            postal = addr.get("postalCode", "")

            text_summary = f"Location: {freeform}\nCity: {city}\nRegion: {country}\nPostal: {postal}"

            return {
                "content": [
                    {"type": "text", "text": text_summary},
                    {"type": "json", "json": {
                        "freeformAddress": freeform,
                        "city": city,
                        "region": country,
                        "postalCode": postal,
                        "latitude": lat,
                        "longitude": lng
                    }}
                ]
            }

        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Azure Maps request failed: {str(e)}"}],
                "isError": True
            }

    async def _predict_crime(self, arguments: Dict[str, Any]):
        """Make a crime prediction using the trained model and include Azure Maps location details in a single human-readable output."""
        try:
            date = arguments.get("date")
            time = arguments.get("time")
            latitude = arguments.get("latitude")
            longitude = arguments.get("longitude")
            print(f"DEBUG: Received arguments: {arguments}", file=sys.stderr)

            # Validate required arguments and formats
            if not (isinstance(date, str) and len(date.split('-')) == 2 and isinstance(time, str) and len(time.split(':')) == 2 and latitude is not None and longitude is not None):
                return {
                    "content": [{"type": "text", "text": "Error: Provide date (MM-DD), time (HH:MM), latitude, and longitude in correct formats."}],
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

            # Get location details from Azure Maps
            location_text = ""
            geo_result = await self._reverse_geocode_azure_maps({"lat": latitude, "lng": longitude})
            if geo_result and "content" in geo_result:
                for item in geo_result["content"]:
                    if item.get("type") == "json":
                        loc = item["json"]
                        location_text = (
                            f"Location Details:\nLocation: {loc.get('freeformAddress', '')}\nCity: {loc.get('city', '')}\nRegion: {loc.get('region', '')}\nPostal: {loc.get('postalCode', '')}\nLatitude: {loc.get('latitude', '')}\nLongitude: {loc.get('longitude', '')}\n"
                        )
                        break

            combined_message = f"{location_text}\nCrime Prediction Result:\n\n{prediction_result}"
            return {
                "content": [{"type": "text", "text": combined_message.strip()}]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error making prediction: {str(e)}"}],
                "isError": True
            }
        
    async def _predict_crime_area(self, arguments: Dict[str, Any]):
        """Make a crime prediction in an area using the trained model and include Azure Maps location details."""
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

            # Try to extract lat/lng pairs from the prediction_result (if possible)
            # This assumes the prediction_result is a string with lat/lng in it, as in your current implementation
            import re
            messages = []
            if "No significant incidents predicted" in prediction_result:
                # For the 'no incidents' case, show the location for the southwest corner
                geo_args = {"lat": southWestLat, "lng": southWestLng}
                geo_result = await self._reverse_geocode_azure_maps(geo_args)
                location_text = ""
                if geo_result and "content" in geo_result:
                    for item in geo_result["content"]:
                        if item.get("type") == "json":
                            loc = item.get("json", {})
                            location_text = (
                                f"Location Details:\nLocation: {loc.get('freeformAddress', '')}\nCity: {loc.get('city', '')}\nRegion: {loc.get('region', '')}\nPostal: {loc.get('postalCode', '')}\nLatitude: {loc.get('latitude', '')}\nLongitude: {loc.get('longitude', '')}\n"
                            )
                            break
                messages.append(f"{location_text}\nArea Crime Prediction Result:\n\n{prediction_result}")
            else:
                # Split by double newlines to get each prediction
                preds = prediction_result.split("\n\n")
                for pred in preds:
                    # Try to extract lat/lng from the message
                    lat_match = re.search(r"lat: ([\d\.-]+)", pred)
                    lng_match = re.search(r"lng: ([\d\.-]+)", pred)
                    location_text = ""
                    if lat_match and lng_match:
                        lat = float(lat_match.group(1))
                        lng = float(lng_match.group(1))
                        geo_args = {"lat": lat, "lng": lng}
                        geo_result = await self._reverse_geocode_azure_maps(geo_args)
                        if geo_result and "content" in geo_result:
                            for item in geo_result["content"]:
                                if item.get("type") == "json":
                                    loc = item.get("json", {})
                                    location_text = (
                                        f"Location Details:\nLocation: {loc.get('freeformAddress', '')}\nCity: {loc.get('city', '')}\nRegion: {loc.get('region', '')}\nPostal: {loc.get('postalCode', '')}\nLatitude: {loc.get('latitude', '')}\nLongitude: {loc.get('longitude', '')}\n"
                                    )
                                    break
                    messages.append(f"{location_text}\n{pred}")
            final_message = "\n\n".join(messages)
            return {
                "content": [{"type": "text", "text": final_message}]
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