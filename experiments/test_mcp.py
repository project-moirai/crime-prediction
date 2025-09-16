#!/usr/bin/env python3
"""
Test script for Crime Prediction MCP Server
"""
import json
import subprocess
import sys
from pathlib import Path
import argparse

def test_mcp_server(date="12-01", time_str="12:00", latitude=46, longitude=5.5):
    """Test the MCP server by sending sample requests."""
    server_path = Path(__file__).parent / "mcp_server.py"
    
    if not server_path.exists():
        print("Error: MCP server not found at", server_path)
        return False
    
    try:
        # Start the MCP server process
        print("Starting MCP server...")
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=server_path.parent
        )
        
        # Give the process a moment to start
        import time
        time.sleep(1)
        
        # Check if process started successfully
        if process.poll() is not None:
            # Process has already terminated
            stdout, stderr = process.communicate()
            print(f"✗ MCP server failed to start:")
            print(f"Return code: {process.returncode}")
            if stdout:
                print(f"STDOUT: {stdout}")
            if stderr:
                print(f"STDERR: {stderr}")
            return False
        
        def send_request_and_get_response(request, request_id, timeout=5):
            """Helper function to send a request and get response."""
            try:
                request_json = json.dumps(request) + "\n"
                
                # Write the request
                try:
                    process.stdin.write(request_json)
                    process.stdin.flush()
                except OSError as e:
                    print(f"✗ Error writing request {request_id}: {e}")
                    return None
                
                # Read response with timeout (Windows-compatible approach)
                import time
                start_time = time.time()
                response_line = ""
                
                while time.time() - start_time < timeout:
                    try:
                        # Check if process is still running
                        if process.poll() is not None:
                            print(f"✗ Server process terminated while waiting for {request_id}")
                            return None
                        
                        # Try to read a line (non-blocking approach for Windows)
                        try:
                            response_line = process.stdout.readline()
                            if response_line:
                                break
                        except OSError:
                            # On Windows, this might fail if no data is ready
                            time.sleep(0.1)
                            continue
                            
                        time.sleep(0.1)  # Small delay to avoid busy waiting
                    except Exception as e:
                        print(f"✗ Error reading response for {request_id}: {e}")
                        return None
                
                if not response_line:
                    print(f"✗ Timeout waiting for response to {request_id}")
                    # Check stderr for any error messages
                    try:
                        # Use communicate with timeout to get any error output
                        process.stdin.close()  # Close stdin to signal end
                        stdout, stderr = process.communicate(timeout=1)
                        if stderr:
                            print(f"Server stderr: {stderr}")
                    except:
                        pass
                    return None
                
                try:
                    return json.loads(response_line.strip())
                except json.JSONDecodeError as e:
                    print(f"✗ Invalid JSON response for request {request_id}: {response_line}")
                    return None
                    
            except Exception as e:
                print(f"✗ Error in send_request_and_get_response for {request_id}: {e}")
                return None
        
        # Initialize the connection
        print("Initializing MCP connection...")
        
        # First try standard MCP initialization
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        init_response = send_request_and_get_response(init_request, "initialization")
        
        # If initialization fails, we'll continue with tools testing anyway
        if init_response:
            print("✓ Initialization response:")
            print(json.dumps(init_response, indent=2))
            
            # Only send initialized notification if initialization succeeded
            if "error" not in init_response:
                print("\nStep 2: Sending initialized notification...")
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                
                notification_json = json.dumps(initialized_notification) + "\n"
                process.stdin.write(notification_json)
                process.stdin.flush()
        else:
            print("⚠ Initialization failed, continuing with direct tool testing...")
        
        # List tools
        print("\nListing available tools...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        tools_response = send_request_and_get_response(list_tools_request, "tools list")
        if not tools_response:
            return False
            
        print("✓ Tools list response:")
        print(json.dumps(tools_response, indent=2))
        
        # Call predict_crime tool
        print("\nMaking a crime prediction...")
        predict_request = {
            "jsonrpc": "2.0", 
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "predict_crime",
                "arguments": {
                    "date": date,
                    "time": time_str,
                    "latitude": latitude,
                    "longitude": longitude
                }
            }
        }
        
        predict_response = send_request_and_get_response(predict_request, "prediction")
        if not predict_response:
            return False
            
        print("✓ Prediction response:")
        # print(json.dumps(predict_response, indent=2))
        
        # Check if we got a valid prediction result
        if "result" in predict_response:
            result = predict_response["result"]
            
            # Try to extract and display the human-readable prediction
            if "structuredContent" in result and "content" in result["structuredContent"]:
                try:
                    human_readable_text = result["structuredContent"]["content"][0]["text"]
                    print("\n--- Human-Readable Prediction ---")
                    print(human_readable_text)
                    print("---------------------------------")
                except (KeyError, IndexError, TypeError) as e:
                    print(f"\nCould not parse human-readable prediction: {e}")
                    print("Raw prediction response:")
                    print(json.dumps(predict_response, indent=2))

            print("\n✓ MCP server Single prediction is working correctly!")
        elif "error" not in predict_response:
            print("✓ MCP server responded successfully (check response format)")
            print(json.dumps(predict_response, indent=2))
            return True
        else:
            print("✗ MCP server returned an error")
            print(json.dumps(predict_response, indent=2))
            return False
        
        # Call predict_crime tool (area version)
        print("\nMaking an area crime prediction...")
        predict_request = {
            "jsonrpc": "2.0", 
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "predict_crime_area",
                "arguments": {
                    "date": date,
                    "startTime": time_str,
                    "endTime": time_str,
                    "southWestLat": latitude,
                    "southWestLng": longitude,
                    "northEastLat": latitude,
                    "northEastLng": longitude
                }
            }
        }
        
        predict_response = send_request_and_get_response(predict_request, "prediction")
        if not predict_response:
            return False
            
        print("✓ Prediction Area response:")
        # print(json.dumps(predict_response, indent=2))
        
        # Check if we got a valid prediction result
        if "result" in predict_response:
            result = predict_response["result"]
            
            # Try to extract and display the human-readable prediction
            if "structuredContent" in result and "content" in result["structuredContent"]:
                try:
                    human_readable_text = result["structuredContent"]["content"][0]["text"]
                    print("\n--- Human-Readable Prediction (area) ---")
                    print(human_readable_text)
                    print("---------------------------------")
                except (KeyError, IndexError, TypeError) as e:
                    print(f"\nCould not parse human-readable prediction: {e}")
                    print("Raw prediction response:")
                    print(json.dumps(predict_response, indent=2))

            print("\n✓ MCP server Area prediction is working correctly!")
            return True
        elif "error" not in predict_response:
            print("✓ MCP server responded successfully (check response format)")
            print(json.dumps(predict_response, indent=2))
            return True
        else:
            print("✗ MCP server returned an error")
            print(json.dumps(predict_response, indent=2))
            return False
            
    except Exception as e:
        print(f"✗ Error testing MCP server: {e}")
        return False
    finally:
        # Clean up the process and show any debug output
        if 'process' in locals():
            try:
                # Close stdin to signal we're done
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                
                # Get any remaining stderr output
                stdout, stderr = process.communicate(timeout=3)
                if stderr and stderr.strip():
                    print("\n🔍 Server debug output:")
                    print(stderr)
                    
                process.terminate()
                process.wait(timeout=2)
            except Exception as e:
                print(f"Note: Error during cleanup: {e}")
                try:
                    process.kill()
                except:
                    pass
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test the Crime Prediction MCP Server.")
    parser.add_argument("--date", default="12-01", help="Date in MM-DD format")
    parser.add_argument("--time", default="12:00", help="Time in HH:MM format")
    parser.add_argument("--latitude", type=float, default=46, help="Latitude")
    parser.add_argument("--longitude", type=float, default=5.5, help="Longitude")
    args = parser.parse_args()

    mcp_ok = test_mcp_server(
        date=args.date,
        time_str=args.time,
        latitude=args.latitude,
        longitude=args.longitude
    )

if __name__ == "__main__":
    main()