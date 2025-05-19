# client_gemini.py (Sales Agent Client - Corrected Orchestration)
import asyncio
import json
import os
import sys
from typing import Optional, Any, Dict, List
from contextlib import AsyncExitStack

from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    Tool,
    FunctionDeclaration,
    HarmCategory,
    HarmBlockThreshold
)
from proto.marshal.collections.maps import MapComposite
from proto.marshal.collections.repeated import RepeatedComposite

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
genai.configure(api_key=GEMINI_API_KEY)

FINANCE_AGENT_SERVER_SCRIPT = "finance_mcp_server.py"
LEGAL_AGENT_SERVER_SCRIPT = "legal_mcp_server.py"

class MCPClient: # Sales Agent's main client
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        self.gemini_model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings
        )
        self.history: List[Dict[str, Any]] = []
        self.tool_specs_cache: Dict[str, Dict[str, Any]] = {}

    async def _cache_tool_specs(self):
        if not self.session:
            print("Sales Client: Error - MCP session not available for caching tool specs.")
            return
        try:
            mcp_tools_response = await self.session.list_tools()
            self.tool_specs_cache = {
                tool.name: {"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema}
                for tool in mcp_tools_response.tools
            }
            print(f"Sales Client: Cached tool specs for Sales Agent: {list(self.tool_specs_cache.keys())}")
        except Exception as e:
            print(f"Sales Client: Failed to fetch or cache tool specs: {e}")

    async def connect_to_server(self, server_script_path: str, agent_name: str = "Sales"):
        if not os.path.exists(server_script_path):
            raise FileNotFoundError(f"Server script not found: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        command = sys.executable if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=os.environ.copy()
        )
        print(f"Sales Client: Attempting to start and connect to {agent_name} MCP server: {server_script_path}")
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        
        if agent_name == "Sales": await self._cache_tool_specs()

        print(f"\nSales Client: Connected to {agent_name} MCP server.")
        if agent_name == "Sales" and self.tool_specs_cache:
             print(f"Sales Client: Tools available for Sales Agent: {list(self.tool_specs_cache.keys())}")


    async def _get_gemini_tools_config(self) -> Optional[List[Tool]]:
        if not self.tool_specs_cache: await self._cache_tool_specs()
        if not self.tool_specs_cache: 
            print("Sales Client: Error - Sales tool specs not available for Gemini config.")
            return None
        return [Tool(function_declarations=[
                FunctionDeclaration(name=name, description=spec.get("description",""), parameters=spec.get("inputSchema") or {})
            ]) for name, spec in self.tool_specs_cache.items()]

    async def _call_external_agent_tool(self, agent_server_script: str, agent_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        print(f"Sales Client: Preparing to call {agent_name} Agent's tool '{tool_name}'.")
        tool_response_text = None
        async with AsyncExitStack() as temp_stack:
            try:
                if not os.path.exists(agent_server_script):
                    msg = f"Error: {agent_name} server script not found at {agent_server_script}."
                    print(f"Sales Client: {msg}")
                    return msg
                is_python = agent_server_script.endswith('.py')
                command = sys.executable if is_python else "node"
                server_params = StdioServerParameters(command=command, args=[agent_server_script], env=os.environ.copy())
                
                stdio_transport = await temp_stack.enter_async_context(stdio_client(server_params))
                # temp_stdio, temp_write = stdio_transport # Not directly used after this
                temp_session = await temp_stack.enter_async_context(ClientSession(stdio_transport[0], stdio_transport[1]))
                await temp_session.initialize()
                print(f"Sales Client: Dynamically connected to {agent_name} MCP server.")

                result = await temp_session.call_tool(tool_name, arguments)
                if result.content and isinstance(result.content, list) and result.content:
                    tool_response_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                else: tool_response_text = f"{agent_name} Agent tool '{tool_name}' returned no parsable content."
                print(f"Sales Client: Response from {agent_name}'s tool '{tool_name}': {tool_response_text[:150]}...")
            except Exception as e:
                print(f"Sales Client: Error during call to {agent_name} Agent's tool '{tool_name}': {e}")
                tool_response_text = f"Error communicating with {agent_name} Agent ({tool_name}): {e}"
            finally: print(f"Sales Client: Finished dynamic call to {agent_name} Agent.")
        return tool_response_text

    def _extract_json_from_tool_output(self, tool_output_str: str) -> Optional[dict]:
        """Extracts JSON part from a tool output string that might have a prefix."""
        try:
            # Find the first occurrence of '{' or '[' which usually marks the start of JSON
            json_start_index = -1
            if '{' in tool_output_str:
                json_start_index = tool_output_str.find('{')
            if '[' in tool_output_str:
                alt_start = tool_output_str.find('[')
                if json_start_index == -1 or alt_start < json_start_index :
                    json_start_index = alt_start
            
            if json_start_index != -1:
                json_string = tool_output_str[json_start_index:]
                # Attempt to find the matching closing bracket for robustness
                # This is a simplified way; a proper parser would be more robust for nested structures.
                # For now, json.loads will tell us if it's valid.
                return json.loads(json_string)
            else:
                return None
        except json.JSONDecodeError as e:
            print(f"Sales Client: JSONDecodeError parsing tool output: {e}. Output was: {tool_output_str}")
            return None

    async def process_query(self, query: str) -> str:
        if not self.session:
            raise ConnectionError("Sales Client: Not connected to Sales MCP server.")

        gemini_tool_config = await self._get_gemini_tools_config()
        
        if self.history and self.history[-1]["role"] == "user":
            self.history[-1] = {"role": "user", "parts": [{"text": query}]}
        else:
            self.history.append({"role": "user", "parts": [{"text": query}]})
            
        final_response_text_parts = []

        for i in range(2): # Max 2 turns: 1 for initial tool call & orchestration, 1 for Gemini's summary
            print(f"\nSales Client: Turn {i+1}/2 for query processing.")
            if not self.history: break

            try:
                print(f"Sales Client: Sending to Gemini. History length: {len(self.history)}, Last role: {self.history[-1]['role']}")
                response = await self.gemini_model.generate_content_async(
                    self.history, tools=gemini_tool_config, generation_config=GenerationConfig(candidate_count=1)
                )
            except Exception as e:
                error_msg = f"Sales Client: Error calling Gemini API: {str(e)}"; print(error_msg); return error_msg

            if not response.candidates:
                no_candidate_msg = "Sales Client: [No response candidates from Gemini.]"
                if response.prompt_feedback: no_candidate_msg += f" Prompt Feedback: {response.prompt_feedback}"
                print(no_candidate_msg); final_response_text_parts.append(no_candidate_msg)
                self.history.append({"role": "model", "parts": [{"text": no_candidate_msg}]}); break 

            candidate = response.candidates[0]
            model_response_content_dict = {"role": "model", "parts": []}
            if candidate.content and candidate.content.parts:
                 for part_obj in candidate.content.parts:
                    if part_obj.text: model_response_content_dict["parts"].append({"text": part_obj.text})
                    elif hasattr(part_obj, 'function_call'):
                        fc = part_obj.function_call
                        model_response_content_dict["parts"].append({"function_call": {"name": fc.name, "args": dict(fc.args)}})
            self.history.append(model_response_content_dict)
            
            latest_model_turn = self.history[-1]
            text_from_this_model_turn = []
            actionable_function_call = None 
            
            for part_data in latest_model_turn.get("parts", []):
                if "text" in part_data: text_from_this_model_turn.append(part_data["text"])
                if "function_call" in part_data:
                    fc_data = part_data["function_call"]
                    if self.tool_specs_cache.get(fc_data["name"]):
                        actionable_function_call = (fc_data["name"], fc_data["args"])
                    else:
                        print(f"Sales Client: Gemini proposed unknown tool '{fc_data['name']}'.")
                        self.history.append({"role": "function", "parts": [{"function_response": {"name": fc_data["name"], "response": {"error": "Proposed tool not recognized by Sales Agent."}}}]})
                    break 
            
            if text_from_this_model_turn:
                final_response_text_parts.extend(text_from_this_model_turn)

            if actionable_function_call:
                tool_name, tool_args_from_gemini = actionable_function_call
                final_tool_args = {k: dict(v) if isinstance(v, MapComposite) else ([dict(i) if isinstance(i, MapComposite) else i for i in v] if isinstance(v, RepeatedComposite) else v) for k,v in tool_args_from_gemini.items()}
                
                current_object_name_for_tool = final_tool_args.get("object_name", "").lower()
                if tool_name == "create_record" and current_object_name_for_tool == "lead":
                    lead_data = final_tool_args.get("data", {})
                    if isinstance(lead_data, dict):
                        original_name_field_value = lead_data.get("Name") 
                        has_explicit_firstname = "FirstName" in lead_data and lead_data.get("FirstName")
                        has_explicit_lastname = "LastName" in lead_data and lead_data.get("LastName")
                        if original_name_field_value and not has_explicit_lastname:
                            if not has_explicit_firstname:
                                name_parts = original_name_field_value.strip().split(" ", 1)
                                if len(name_parts) > 1: lead_data["FirstName"], lead_data["LastName"] = name_parts[0], name_parts[1]
                                else: lead_data["LastName"] = original_name_field_value
                            else: lead_data["LastName"] = original_name_field_value
                            if "Name" in lead_data: del lead_data["Name"]
                        final_tool_args["data"] = lead_data

                print(f"Sales Client: Executing Sales Agent tool '{tool_name}' with args: {final_tool_args}")
                orchestrated_tool_output_summary = f"Tool '{tool_name}' executed with no specific output summary generated." # Default
                try:
                    tool_call_mcp_result = await self.session.call_tool(tool_name, final_tool_args)
                    tool_output_str = (tool_call_mcp_result.content[0].text if tool_call_mcp_result.content and 
                                       isinstance(tool_call_mcp_result.content, list) and tool_call_mcp_result.content and
                                       hasattr(tool_call_mcp_result.content[0], 'text') else str(tool_call_mcp_result.content))
                    
                    print(f"Sales Client: Tool '{tool_name}' executed by Sales MCP. Raw Result: {tool_output_str[:300]}...") # Increased log length
                    orchestrated_tool_output_summary = tool_output_str 

                    if tool_name == "create_record" and current_object_name_for_tool == "lead":
                        new_lead_id = None
                        lead_description_for_verification = final_tool_args.get("data", {}).get("Description", "No description provided.")
                        
                        # CORRECTED JSON PARSING
                        sf_response_data = self._extract_json_from_tool_output(tool_output_str)

                        if sf_response_data and isinstance(sf_response_data, dict) and sf_response_data.get("success") and sf_response_data.get("id"):
                            new_lead_id = sf_response_data["id"]
                            print(f"Sales Client: Lead created (ID: {new_lead_id}). Initiating verification workflow.")
                            
                            verification_notes = [f"Lead {new_lead_id} created in Salesforce."]
                            finance_approved = False; legal_approved = False

                            finance_response = await self._call_external_agent_tool(
                                FINANCE_AGENT_SERVER_SCRIPT, "Finance", "assess_financial_viability", 
                                {"lead_id": new_lead_id, "lead_description": lead_description_for_verification}
                            )
                            if finance_response:
                                verification_notes.append(f"Finance Check: {finance_response}")
                                if "financially viable" in finance_response.lower(): finance_approved = True
                            
                            if finance_approved:
                                legal_response = await self._call_external_agent_tool(
                                    LEGAL_AGENT_SERVER_SCRIPT, "Legal", "assess_legal_acceptability", 
                                    {"lead_id": new_lead_id, "lead_description": lead_description_for_verification}
                                )
                                if legal_response:
                                    verification_notes.append(f"Legal Check: {legal_response}")
                                    if "legally acceptable" in legal_response.lower(): legal_approved = True
                            else: verification_notes.append("Legal check skipped (Finance not approved).")

                            final_sf_status = "Verification Pending"
                            if finance_approved and legal_approved: final_sf_status = "Fully Verified"
                            elif finance_approved: final_sf_status = "Finance Approved"
                            elif not finance_approved and finance_response: final_sf_status = "Finance Rejected"
                            
                            print(f"Sales Client: Updating lead {new_lead_id} to '{final_sf_status}'.")
                            update_res = await self.session.call_tool("update_record", {
                                "object_name": "Lead", "record_id": new_lead_id, "data": {"Status": final_sf_status}
                            })
                            update_out_raw = (update_res.content[0].text if update_res.content and isinstance(update_res.content,list) and update_res.content and hasattr(update_res.content[0],'text') else str(update_res.content))
                            update_out_json = self._extract_json_from_tool_output(update_out_raw) # Parse update output too
                            update_status_msg = f"Update to '{final_sf_status}' "
                            if update_out_json and update_out_json.get("success"): update_status_msg += "succeeded."
                            elif update_out_json: update_status_msg += f"failed or had issues: {update_out_json.get('errors', [])}"
                            else: update_status_msg += f"response unclear: {update_out_raw}"
                            verification_notes.append(f"Salesforce status {update_status_msg}")
                            
                            orchestrated_tool_output_summary = (f"Lead creation and verification for ID {new_lead_id} complete. "
                                                               f"Final Status in Salesforce: {final_sf_status}. "
                                                               f"Details: {' '.join(verification_notes)}")
                        else: 
                            orchestrated_tool_output_summary = f"Lead creation by Sales Agent: JSON could not be parsed or ID/success missing. Verification skipped. SF Response: {tool_output_str}"
                            print(f"Sales Client: {orchestrated_tool_output_summary}")
                    
                    self.history.append({"role": "function", "parts": [{"function_response": {"name": tool_name, "response": {"result": orchestrated_tool_output_summary}}}]})
                
                except Exception as e: 
                    raw_error_message = str(e); print(f"Sales Client: Error executing Sales tool '{tool_name}': {raw_error_message}")
                    self.history.append({"role": "function", "parts": [{"function_response": {"name": tool_name, "response": {"error": f"Error in Sales tool {tool_name}: {raw_error_message}"}}}]})
            
            if not actionable_function_call:
                print("Sales Client: No actionable function call from Gemini in this turn. Assuming final text response.")
                break 
            
            print(f"Sales Client: End of turn {i+1}. History length: {len(self.history)}")

        return "\n".join(filter(None, final_response_text_parts))

    async def chat_loop(self):
        print("\nSales Client: MCP Client Started (Orchestration v4 Mode)!")
        print("Sales Client: Type your queries or 'quit' to exit.")
        self.history = [] 
        while True:
            try:
                query_text = input("\nQuery: ").strip()
                if query_text.lower() == 'quit': break
                if not query_text: continue
                response_str = await self.process_query(query_text)
                print("\nSales Gemini: " + response_str)
            except KeyboardInterrupt: print("\nSales Client: Exiting chat loop..."); break
            except Exception as e: print(f"\nSales Client: Error in chat loop: {e}"); import traceback; traceback.print_exc()

    async def cleanup(self):
        print("Sales Client: Cleaning up resources...")
        if hasattr(self, 'exit_stack') and self.exit_stack: await self.exit_stack.aclose()
        print("Sales Client: Cleanup complete.")

async def main():
    sales_server_script = "salesforce_mcp_server.py" 
    if len(sys.argv) >= 2: sales_server_script = sys.argv[1]
    
    for script_path in [sales_server_script, FINANCE_AGENT_SERVER_SCRIPT, LEGAL_AGENT_SERVER_SCRIPT]:
        if not os.path.exists(script_path):
            print(f"CRITICAL WARNING: Required server script '{script_path}' not found. The system may not function correctly.")
            if script_path == sales_server_script: sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sales_server_script, agent_name="Sales")
        await client.chat_loop()
    except Exception as e:
        print(f"Sales Client: An unhandled error occurred in main: {e}")
        import traceback; traceback.print_exc()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
