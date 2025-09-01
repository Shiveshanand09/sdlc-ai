// Updated SDLC AI VS Code Extension with main editor display for JSON files
// and improved validation workflow

import * as vscode from 'vscode';
import * as path from 'path';

interface WebviewMessage {
  command: 'startAgents' | 'continueAgent' | 'validateTask' | 'saveData' | 'log';
  data: any;
}

type AgentResponse = Record<string, any>;

export function activate(context: vscode.ExtensionContext) {
  const provider = new SdlcAiViewProvider(context);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      'sdlcai.configView',
      provider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('extension.showInputForm', () => {
      vscode.commands.executeCommand('sdlcai.configView.focus');
      vscode.commands.executeCommand('workbench.view.extension.sdlcai-sidebar');
    })
  );
}

class SdlcAiViewProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private agentOutputs: Record<string, AgentResponse> = {};
  private agentSequence: string[] = [];
  private currentAgentIndex = -1;
  private initialInputs: { requirements: string; context: string; title: string; agents: string[] } | null = null;
  private tempFilePaths: Record<string, vscode.Uri> = {}; // Track temporary files

  constructor(private context: vscode.ExtensionContext) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this.context.extensionUri]
    };

    webviewView.webview.html = this.getWebviewContent();

    webviewView.webview.onDidReceiveMessage(
      async (message: WebviewMessage) => {
        try {
          switch (message.command) {
            case 'startAgents':
              // Save inputs and reset state
              this.initialInputs = message.data;
              this.agentSequence = message.data.agents;
              this.agentOutputs = {};
              this.currentAgentIndex = -1;
              this.tempFilePaths = {};
              this.postMessage({ command: 'setAgentSequence', agents: this.agentSequence });
              // Kick off first agent
              await this.processNextAgent();
              break;

            case 'continueAgent':
              // Continue to next agent after validation
              await this.processNextAgent();
              break;

            case 'validateTask':
              await this.validateTask(
                message.data.agentName,
                message.data.taskName,
                message.data.output
              );
              break;

            case 'saveData':
              await this.saveData(message.data);
              break;

            case 'log':
              console.log(message.data);
              break;
          }
        } catch (error) {
          console.error('Error handling message:', error);
          this.postMessage({ command: 'error', message: `Failed to process request: ${error}` });
        }
      },
      undefined,
      this.context.subscriptions
    );
  }

  private async callApiEndpoint<T extends object>(endpoint: string, data: any): Promise<T> {
    const res = await fetch(`http://localhost:8000${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      throw new Error(`API call failed with status ${res.status}`);
    }
    return (await res.json()) as T;
  }

  private async processNextAgent() {
    this.currentAgentIndex++;
    if (!this.initialInputs) {
      throw new Error('Initial inputs not set');
    }

    if (this.currentAgentIndex >= this.agentSequence.length) {
      // All agents done
      this.postMessage({ command: 'agentsCompleted' });
      vscode.window.showInformationMessage('SDLC Process completed successfully');
      return;
    }

    const agent = this.agentSequence[this.currentAgentIndex];
    this.postMessage({ command: 'agentStarted', agent });

    try {
      // Destructure saved inputs
      const { requirements, context: projContext, title } = this.initialInputs;
      let output: AgentResponse;

      switch (agent) {
        case 'KnowledgeBase':
          output = await this.callApiEndpoint<AgentResponse>('/agent/knowledge', {
            user_requirements: requirements,
            project_context: projContext,
            title
          });
          break;

        case 'Requirements':
          output = await this.callApiEndpoint<AgentResponse>('/agent/requirements', {
            user_requirements: requirements,
            project_context: projContext,
            title
          });
          break;

        case 'Architecture':
          output = await this.callApiEndpoint<AgentResponse>('/agent/architecture', {
            user_requirements: requirements,
            project_context: projContext,
            title,
            requirement_output: this.agentOutputs['Requirements'],
            knowledge_output: this.agentOutputs['KnowledgeBase']
          });
          break;

        case 'Skeletons':
          output = await this.callApiEndpoint<AgentResponse>('/agent/skeleton', {
            project_context: projContext,
            title,
            architecture_output: this.agentOutputs['Architecture']
          });
          break;

        case 'Generator':
          output = await this.callApiEndpoint<AgentResponse>('/agent/codegen', {
            project_context: projContext,
            title,
            architecture_output: this.agentOutputs['Architecture'],
            skeleton_output: this.agentOutputs['Skeletons']
          });
          break;

        default:
          throw new Error(`Unknown agent: ${agent}`);
      }

      // Store the output
      this.agentOutputs[agent] = output;
      
      // Open the output in the main editor and wait for validation
      await this.openOutputInEditor(agent, output);
      
      // Notify the webview
      this.postMessage({ 
        command: 'validateAgent', 
        agent, 
        tasks: Object.keys(output), 
        outputs: output 
      });

    } catch (error) {
      console.error(`Error in agent ${agent}:`, error);
      this.postMessage({ command: 'error', message: `Agent ${agent} failed: ${error}` });
    }
  }

  private async openOutputInEditor(agent: string, output: Record<string, any>): Promise<void> {
    try {
      const jsonContent = JSON.stringify(output, null, 2);
      
      // Create a temporary file for each task output
      const tempFilePath = vscode.Uri.joinPath(
        this.context.globalStorageUri, 
        `${agent}_${Date.now()}.json`
      );
      
      // Ensure the directory exists
      await vscode.workspace.fs.createDirectory(this.context.globalStorageUri);
      
      // Write the JSON content to the temporary file
      await vscode.workspace.fs.writeFile(tempFilePath, Buffer.from(jsonContent));
      
      // Store the file path for later reference
      this.tempFilePaths[agent] = tempFilePath;
      
      // Open the file in the editor
      const document = await vscode.workspace.openTextDocument(tempFilePath);
      await vscode.window.showTextDocument(document, { preview: false });
      
      // Show information message to guide the user
      vscode.window.showInformationMessage(
        `${agent} results are now open in the editor. Review and validate when ready.`,
        'Validate'
      ).then(selection => {
        if (selection === 'Validate') {
          this.validateCurrentAgentOutput(agent);
        }
      });
    } catch (error) {
      console.error(`Error opening output in editor:`, error);
      throw error;
    }
  }

  private async validateCurrentAgentOutput(agent: string) {
    try {
      // Try to find the document that corresponds to this agent
      const tempFilePath = this.tempFilePaths[agent];
      if (!tempFilePath) {
        throw new Error(`No file found for agent ${agent}`);
      }
      
      // Get the document
      const document = await vscode.workspace.openTextDocument(tempFilePath);
      const content = document.getText();
      
      try {
        // Parse the JSON content (catches syntax errors)
        const updatedOutput = JSON.parse(content);
        
        // Update the stored output
        this.agentOutputs[agent] = updatedOutput;
        
        // Send all tasks for validation to the backend
        for (const taskName of Object.keys(updatedOutput)) {
          await this.validateTask(agent, taskName, updatedOutput[taskName]);
        }
        
        // Mark all tasks as validated in the UI
        this.postMessage({ 
          command: 'agentValidated', 
          agent,
          success: true
        });
        
        vscode.window.showInformationMessage(`${agent} output validated successfully.`, 'Continue').then(selection => {
          if (selection === 'Continue') {
            this.processNextAgent();
          }
        });
      } catch (error) {
        if (error instanceof Error) {
          vscode.window.showErrorMessage(`Invalid JSON format: ${error.message}. Please fix the errors and try again.`);
        } else {
          vscode.window.showErrorMessage('Invalid JSON format: An unknown error occurred. Please fix the errors and try again.');
        }
      }
    } catch (error) {
      console.error(`Error validating agent output:`, error);
      if (error instanceof Error) {
        vscode.window.showErrorMessage(`Error validating output: ${error.message}`);
      } else {
        vscode.window.showErrorMessage('Error validating output: An unknown error occurred.');
      }
    }
  }

  private async validateTask(agentName: string, taskName: string, output: any) {
    // Send modified output for validation
    await this.callApiEndpoint<{}>('/agent/validate', { task_name: taskName, modified_output: output });
    // Update stored output
    this.agentOutputs[agentName][taskName] = output;
    this.postMessage({ command: 'taskValidated', agentName, taskName, success: true });
  }

  private async saveData(data: any) {
    const jsonContent = JSON.stringify(data, null, 2);
    const filePath = vscode.Uri.joinPath(this.context.globalStorageUri, 'user_inputs.json');

    await vscode.workspace.fs.createDirectory(this.context.globalStorageUri);
    await vscode.workspace.fs.writeFile(filePath, Buffer.from(jsonContent));

    vscode.window.showInformationMessage('Input and results saved successfully');
  }

  private postMessage(message: any) {
    this._view?.webview.postMessage(message);
  }

  private getWebviewContent(): string {
    return `
      <!DOCTYPE html>
      <html lang="en">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SDLC AI Assistant</title>
        <style>
          body {
            font-family: var(--vscode-font-family);
            padding: 10px;
            color: var(--vscode-foreground);
          }
          textarea, input[type="text"] {
            width: 100%;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 5px;
            margin-bottom: 10px;
          }
          button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            cursor: pointer;
            margin-top: 10px;
          }
          button:hover {
            background-color: var(--vscode-button-hoverBackground);
          }
          .hidden {
            display: none;
          }
          .agent-panel {
            border: 1px solid var(--vscode-panel-border);
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
          }
          .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border-left-color: var(--vscode-button-background);
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
            vertical-align: middle;
          }
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          .alert {
            padding: 8px;
            margin: 8px 0;
            border-radius: 4px;
            background-color: var(--vscode-inputValidation-infoBackground);
            border: 1px solid var(--vscode-inputValidation-infoBorder);
          }
          .error {
            background-color: var(--vscode-inputValidation-errorBackground);
            border: 1px solid var(--vscode-inputValidation-errorBorder);
          }
          .success {
            background-color: var(--vscode-terminal-ansiGreen);
            color: var(--vscode-editor-background);
            padding: 4px 8px;
            border-radius: 4px;
            margin-left: 8px;
          }
        </style>
      </head>
      <body>
        <div id="input-form">
          <h2>SDLC AI Project Generator</h2>
          
          <label>User Requirements:</label>
          <textarea id="requirements" rows="4" placeholder="Describe your project requirements here..."></textarea>
          
          <label>Project Context:</label>
          <textarea id="context" rows="4" placeholder="Provide additional context about your project..."></textarea>
          
          <label>One Word Title:</label>
          <input id="title" type="text" placeholder="ProjectName" />
          
          <label>Select Agents:</label><br>
          <div style="display: flex; flex-direction: column; margin: 8px 0;">
            <label>
              <input type="checkbox" id="kb" value="KnowledgeBase" checked>
              KnowledgeBase - Gathers domain knowledge and tech stack
            </label>
            <label>
              <input type="checkbox" id="req" value="Requirements" checked>
              Requirements - Analyzes requirements and extracts tasks
            </label>
            <label>
              <input type="checkbox" id="arch" value="Architecture" checked>
              Architecture - Designs system architecture
            </label>
            <label>
              <input type="checkbox" id="skel" value="Skeletons" checked>
              Skeletons - Creates code skeletons and boilerplate
            </label>
            <label>
              <input type="checkbox" id="gen" value="Generator" checked>
              Generator - Implements full code files
            </label>
          </div>
          
          <button id="startButton">Start SDLC Process</button>
        </div>
        
        <div id="progress-panel" class="hidden">
          <h2>SDLC Process</h2>
          <div id="agent-progress"></div>
          <div id="current-agent" class="agent-panel hidden">
            <h3 id="current-agent-name"></h3>
            <div id="current-agent-status">
              <div class="spinner"></div> Processing...
            </div>
            <div class="alert">
              <p>The output has been opened in the main editor. Review the content, make any necessary changes, and then click "Validate" to continue.</p>
            </div>
          </div>
          <div id="completion-panel" class="hidden">
            <h3>SDLC Process Completed</h3>
            <p>All selected agents have completed their work successfully.</p>
            <button id="reset-button">Start New SDLC Process</button>
          </div>
        </div>
        
        <script>
          // Get the VS Code API
          const vscode = acquireVsCodeApi();
          
          // DOM elements
          const inputForm = document.getElementById('input-form');
          const progressPanel = document.getElementById('progress-panel');
          const currentAgentPanel = document.getElementById('current-agent');
          const currentAgentName = document.getElementById('current-agent-name');
          const currentAgentStatus = document.getElementById('current-agent-status');
          const completionPanel = document.getElementById('completion-panel');
          const resetButton = document.getElementById('reset-button');
          const agentProgress = document.getElementById('agent-progress');
          
          // Store data between webview refreshes
          let state = {
            selectedAgents: []
          };
          
          // Add event listeners
          document.getElementById('startButton').addEventListener('click', startSdlcProcess);
          resetButton.addEventListener('click', resetProcess);
          
          // Start the SDLC process
          function startSdlcProcess() {
            const requirements = document.getElementById('requirements').value;
            const context = document.getElementById('context').value;
            const title = document.getElementById('title').value;
            
            if (!requirements.trim() || !context.trim() || !title.trim()) {
              alert('Please fill in all fields');
              return;
            }
            
            const agents = [];
            document.querySelectorAll('input[type="checkbox"]:checked')
              .forEach(cb => agents.push(cb.value));
            
            if (agents.length === 0) {
              alert('Please select at least one agent');
              return;
            }
            
            // Store input data
            const inputData = { requirements, context, title, agents };
            state.selectedAgents = agents;
            
            // Show progress panel and hide input form
            inputForm.classList.add('hidden');
            progressPanel.classList.remove('hidden');
            
            // Initialize progress display
            updateAgentProgressDisplay(agents);
            
            // Start the agent sequence
            vscode.postMessage({ 
              command: 'startAgents', 
              data: inputData 
            });
          }
          
          function updateAgentProgressDisplay(agents) {
            agentProgress.innerHTML = '';
            agents.forEach((agent, index) => {
              const agentElem = document.createElement('div');
              agentElem.className = 'agent-panel';
              agentElem.innerHTML = \`
                <h4>\${agent} \${index === 0 ? '<span class="spinner"></span>' : ''}</h4>
                <div class="status">Waiting...</div>
              \`;
              agentElem.id = \`agent-progress-\${agent}\`;
              agentProgress.appendChild(agentElem);
            });
          }
          
          function updateAgentStatus(agent, status) {
            const agentElem = document.getElementById(\`agent-progress-\${agent}\`);
            if (agentElem) {
              const statusElem = agentElem.querySelector('.status');
              statusElem.textContent = status;
              
              // Remove spinner from all agents
              document.querySelectorAll('#agent-progress .spinner').forEach(spinner => {
                spinner.remove();
              });
              
              // Add spinner to current agent
              if (status === 'Processing') {
                const header = agentElem.querySelector('h4');
                const spinner = document.createElement('span');
                spinner.className = 'spinner';
                header.appendChild(spinner);
              }
              
              if (status === 'Completed') {
                agentElem.querySelector('h4').innerHTML += ' <span class="success">✓</span>';
              }
            }
          }
          
          function resetProcess() {
            state = {
              selectedAgents: []
            };
            
            // Reset UI
            progressPanel.classList.add('hidden');
            completionPanel.classList.add('hidden');
            inputForm.classList.remove('hidden');
            currentAgentPanel.classList.add('hidden');
          }
          
          // Handle messages from the extension
          window.addEventListener('message', event => {
            const message = event.data;
            
            switch(message.command) {
              case 'setAgentSequence':
                state.agentSequence = message.agents;
                break;
                
              case 'agentStarted':
                updateAgentStatus(message.agent, 'Processing');
                currentAgentPanel.classList.remove('hidden');
                currentAgentName.textContent = message.agent;
                break;
                
              case 'validateAgent':
                updateAgentStatus(message.agent, 'Validating');
                currentAgentStatus.textContent = 'Review output in editor and validate';
                break;
                
              case 'agentValidated':
                updateAgentStatus(message.agent, 'Completed');
                break;
                
              case 'agentsCompleted':
                currentAgentPanel.classList.add('hidden');
                completionPanel.classList.remove('hidden');
                break;
                
              case 'error':
                const errorElem = document.createElement('div');
                errorElem.className = 'alert error';
                errorElem.textContent = message.message;
                document.body.prepend(errorElem);
                
                // Auto-remove after 10 seconds
                setTimeout(() => {
                  errorElem.remove();
                }, 10000);
                break;
            }
          });
        </script>
      </body>
      </html>
    `;
  }
}