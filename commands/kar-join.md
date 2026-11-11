---
description: Join a kar run as one more agent, working one lane by hand in this session
argument-hint: <run folder>
---
Join the kar run at $ARGUMENTS as one agent of its team.

`kar` is this plugin's command, on the shell's PATH (else `${CLAUDE_PLUGIN_ROOT}/bin/kar`).

1. If no harness process serves the run (`$ARGUMENTS/ws/queue/server.json` missing or older than 15 seconds), start
   one in the background: `kar serve $ARGUMENTS --detach`.
2. `cd $ARGUMENTS/ws`, read `AGENTS.md` and `METHOD.md`, then run `./kar next`. It gives you a lane and its command
   (`./lanes/<id>/kar`). If it says the run is driven by `kar run`, tell the user and stop: that run hands out its
   own lanes.
3. Do your lane's step exactly as `./lanes/<id>/kar next` describes. Write only in your lane folder, use the GPU only
   through your lane's command, and hand back with your role's command. Subagents are welcome for reading.
4. When your lane is done, run `./kar next` for another lane if the user wants you to keep going.
