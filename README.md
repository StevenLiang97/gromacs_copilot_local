
## The local LLM (Qwen3:32B) is now supported.

Now you can use Ollama(qwen3:32b) to control GROMACS!

This project is modified from the original source code, with assistance from Claude-3.7-Sonnet. 

A new "--time out" option has been added, which can be adjusted based on the response time of the local Ollama model (in seconds).

!!! Please make sure that Ollama is running. 

!!! The "exit" command will also stop the local Ollama model, and you will need to restart Ollama before starting a new project.

```bash
ollama run qwen3:32b
```

### Using Ollama
```bash
gmx_copilot --workspace md_workspace/
--prompt "setup simulation system for 1pga_protein.pdb in the workspace" \
--api-key ~ \
--model qwen3:32b \
--url http://localhost:11434/v1/chat/completions \
--timeout 120
```

