# A2A-Using-MCP

This project demonstrates a multi-agent system using Google's Gemini LLM and the Model Contex Protocol (MCP) to automate lead creation and verification. It features specialized Sales, Finance, and Legal agents.

## Overview

The system automates a lead processing workflow:
1.  A **Sales Agent** creates a new lead in a CRM (e.g., Salesforce) based on user input.
2.  This triggers an automated verification process:
    * The **Finance Agent** assesses financial viability.
    * If approved, the **Legal Agent** checks for legal/compliance acceptability.
3.  The Sales Agent updates the lead's status in the CRM based on these verifications and informs the user.

The Sales Agent's client-side logic orchestrates this Agent-to-Agent (A2A) communication.

## Features

* **Specialized AI Agents:** Dedicated agents for Sales (CRM), Finance (financial assessment), and Legal (legal assessment).
* **LLM-Powered Tools:** Agents use Gemini-powered tools for their tasks.
* **MCP-Based Communication:** Agents interact via the Model Contex Protocol.
* **Orchestrated Workflow:** The Sales Agent's client automatically manages the multi-step verification process after lead creation.
* **Salesforce Integration:** Configurable integration with Salesforce for CRM operations.

## Core Components

* **`salesforce_mcp_server.py`**: Sales Agent's server (Salesforce interaction).
* **`finance_mcp_server.py`**: Finance Agent's server (financial assessment).
* **`legal_mcp_server.py`**: Legal Agent's server (legal assessment).
* **`client_gemini.py`**: Sales Agent's client & A2A orchestrator for lead verification.
* **`.env`**: For environment variables (API keys, credentials).
* **`requirements.txt`**: Python package dependencies.

## Setup and Installation

### Prerequisites

* Python 3.9+
* Git
* Google Gemini API Key
* Salesforce Developer Org/Sandbox (optional, for full functionality)

### Steps

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/Krishnapopat/A2A-Using-MCP.git
    cd A2A-Using-MCP
    ```
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Environment Variables (`.env` file):**
    ```env
    # For Google Gemini API (used by all agents)
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    # For Salesforce Integration (used by salesforce_mcp_server.py)
    SALESFORCE_USERNAME="your_salesforce_username"
    SALESFORCE_PASSWORD="your_salesforce_password"
    SALESFORCE_SECURITY_TOKEN="your_salesforce_security_token"
    ```

## Running the System

The Sales Agent client (`client_gemini.py`) orchestrates the process. It will attempt to start the Finance and Legal agent MCP server scripts (`finance_mcp_server.py`, `legal_mcp_server.py`) as subprocesses when needed.

1.  **Ensure Server Scripts Are Present:**
    Verify `salesforce_mcp_server.py`, `finance_mcp_server.py`, and `legal_mcp_server.py` are in the same directory as `client_gemini.py` (or update paths in `client_gemini.py`).
2.  **Run the Sales Agent Client:**
    ```bash
    python client_gemini.py salesforce_mcp_server.py
    ```
    *(The `salesforce_mcp_server.py` argument can be omitted if it's the default in `client_gemini.py`'s `main` function).*

The client will start, connect to (and start) `salesforce_mcp_server.py`. When you request a lead creation, the client will then dynamically start and interact with the Finance and Legal agent servers.

## Workflow Example (Simplified)

1.  **User (to Sales Agent):** "Create a lead for Innovate Corp..."
2.  **Sales LLM:** Decides to use `create_record`.
3.  **`client_gemini.py` (Orchestrator):**
    * Executes `create_record` via `salesforce_mcp_server.py`.
    * If successful, starts `finance_mcp_server.py` & calls its assessment tool.
    * If Finance approves, starts `legal_mcp_server.py` & calls its assessment tool.
    * Calls `salesforce_mcp_server.py` again to update the lead status (e.g., to "Fully Verified").
    * Compiles a summary.
4.  **Sales LLM:** Receives the summary.
5.  **Sales LLM (to User):** "OK. Lead for Innovate Corp created and verified (ID 00Q...). Status: Fully Verified."

## Contributing

Contributions are welcome! Please fork, make changes, and submit pull requests. Open an issue for bugs or suggestions.

## License

This project is licensed under the [MIT License](LICENSE).
