# finance_mcp_server.py
import asyncio
import json
import os
from dotenv import load_dotenv

# Assuming mcp.server and mcp.types are in yourPYTHONPATH or installed
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio as mcp_stdio
import mcp.types as types

import google.generativeai as genai

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
    # For critical operations, you might want to raise an error or exit
genai.configure(api_key=GEMINI_API_KEY)

# Create a server instance for the Finance Agent
finance_server = Server("finance-mcp-agent")

# Initialize Gemini Model (can be shared)
try:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    gemini_model = None # Handle cases where model init might fail

@finance_server.list_tools()
async def handle_list_finance_tools() -> list[types.Tool]:
    """Lists available tools for the Finance Agent."""
    return [
        types.Tool(
            name="assess_financial_viability",
            description="Assesses the financial viability of a lead based on its description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "lead_id": {
                        "type": "string",
                        "description": "The unique identifier of the lead.",
                    },
                    "lead_description": {
                        "type": "string",
                        "description": "A description of the lead and their request/interest.",
                    },
                },
                "required": ["lead_id", "lead_description"],
            },
        )
    ]

@finance_server.call_tool()
async def handle_call_finance_tool(name: str, arguments: dict[str, str]) -> list[types.TextContent]:
    """Handles tool calls for the Finance Agent."""
    if name == "assess_financial_viability":
        lead_id = arguments.get("lead_id")
        lead_description = arguments.get("lead_description")

        if not lead_id or not lead_description:
            raise ValueError("Missing 'lead_id' or 'lead_description' argument.")

        if not gemini_model:
            return [types.TextContent(type="text", text=f"Finance Agent Error: Gemini model not available for lead {lead_id}.")]

        print(f"Finance Agent: Assessing financial viability for lead ID: {lead_id}")
        print(f"Finance Agent: Lead Description: \"{lead_description[:100]}...\"")

        try:
            # Simple prompt for Gemini
            prompt = (
                f"Based on the following lead description, provide a brief, positive assessment "
                f"of its general financial viability for a standard B2B product/service. "
                f"Assume standard positive conditions unless the description strongly implies otherwise. "
                f"Lead ID: {lead_id}\nDescription: \"{lead_description}\"\n\n"
                f"Example positive assessment: \"The lead's interest in [topic] suggests potential for a valuable engagement. Appears financially viable.\""
                f"Respond with a concise affirmation focused on financial viability."
            )
            
            response = await gemini_model.generate_content_async(prompt)
            assessment_text = response.text.strip()
            
            # Ensure a positive framing for this simplified agent
            final_assessment = f"Lead ID {lead_id}: Financial assessment based on description - '{assessment_text}'. Considered financially viable for next steps."
            print(f"Finance Agent: Assessment for {lead_id}: {final_assessment}")

        except Exception as e:
            print(f"Finance Agent: Error during Gemini call for lead {lead_id}: {str(e)}")
            final_assessment = f"Finance Agent: Could not complete financial viability assessment for lead {lead_id} due to an internal error: {str(e)}"

        return [types.TextContent(type="text", text=final_assessment)]

    raise ValueError(f"Finance Agent: Unknown tool: {name}")

async def run_finance_server():
    """Runs the Finance MCP server."""
    async with mcp_stdio.stdio_server() as (read, write):
        await finance_server.run(
            read,
            write,
            InitializationOptions(
                server_name="finance-mcp-agent",
                server_version="0.1.0",
                capabilities=finance_server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    print("Starting Finance MCP Agent Server...")
    if not GEMINI_API_KEY:
        print("CRITICAL: GOOGLE_API_KEY is not set. The agent's core functionality will fail.")
    asyncio.run(run_finance_server())
