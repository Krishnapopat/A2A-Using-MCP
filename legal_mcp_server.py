# legal_mcp_server.py
import asyncio
import json
import os
from dotenv import load_dotenv

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio as mcp_stdio
import mcp.types as types

import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

legal_server = Server("legal-mcp-agent")

try:
    gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    gemini_model = None

@legal_server.list_tools()
async def handle_list_legal_tools() -> list[types.Tool]:
    """Lists available tools for the Legal Agent."""
    return [
        types.Tool(
            name="assess_legal_acceptability",
            description="Assesses the legal acceptability of a lead based on its description.",
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

@legal_server.call_tool()
async def handle_call_legal_tool(name: str, arguments: dict[str, str]) -> list[types.TextContent]:
    """Handles tool calls for the Legal Agent."""
    if name == "assess_legal_acceptability":
        lead_id = arguments.get("lead_id")
        lead_description = arguments.get("lead_description")

        if not lead_id or not lead_description:
            raise ValueError("Missing 'lead_id' or 'lead_description' argument.")

        if not gemini_model:
            return [types.TextContent(type="text", text=f"Legal Agent Error: Gemini model not available for lead {lead_id}.")]

        print(f"Legal Agent: Assessing legal acceptability for lead ID: {lead_id}")
        print(f"Legal Agent: Lead Description: \"{lead_description[:100]}...\"")
        
        try:
            prompt = (
                f"Based on the following lead description, provide a brief, positive assessment "
                f"of its general legal acceptability for a standard B2B product/service. "
                f"Assume standard positive conditions and compliance unless the description strongly implies otherwise. "
                f"Lead ID: {lead_id}\nDescription: \"{lead_description}\"\n\n"
                f"Example positive assessment: \"The lead's request aligns with common business practices. Appears legally acceptable.\""
                f"Respond with a concise affirmation focused on legal acceptability."
            )
            
            response = await gemini_model.generate_content_async(prompt)
            assessment_text = response.text.strip()
            
            final_assessment = f"Lead ID {lead_id}: Legal assessment based on description - '{assessment_text}'. Considered legally acceptable for next steps."
            print(f"Legal Agent: Assessment for {lead_id}: {final_assessment}")

        except Exception as e:
            print(f"Legal Agent: Error during Gemini call for lead {lead_id}: {str(e)}")
            final_assessment = f"Legal Agent: Could not complete legal acceptability assessment for lead {lead_id} due to an internal error: {str(e)}"
            
        return [types.TextContent(type="text", text=final_assessment)]

    raise ValueError(f"Legal Agent: Unknown tool: {name}")

async def run_legal_server():
    """Runs the Legal MCP server."""
    async with mcp_stdio.stdio_server() as (read, write):
        await legal_server.run(
            read,
            write,
            InitializationOptions(
                server_name="legal-mcp-agent",
                server_version="0.1.0",
                capabilities=legal_server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    print("Starting Legal MCP Agent Server...")
    if not GEMINI_API_KEY:
        print("CRITICAL: GOOGLE_API_KEY is not set. The agent's core functionality will fail.")
    asyncio.run(run_legal_server())
