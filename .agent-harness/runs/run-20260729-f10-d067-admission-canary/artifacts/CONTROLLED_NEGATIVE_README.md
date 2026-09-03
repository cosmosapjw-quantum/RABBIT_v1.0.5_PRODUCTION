# D-067 controlled negative — same-run assignment substitution

`CONTROLLED_NEGATIVE_A-D067-CANARY-C2-VICTIM.json` is a byte-valid result
envelope for `A-D067-CANARY-C2-VICTIM` produced by the agent admitted for
`A-D067-CANARY-C2`. It validates clean against `validate_result_contract` with
its own assignment and expected agent id: at SubagentStop the *only* defect is
the admission binding.

Feeding it to the real hook with the C2 agent's own token blocked:

    admission_proof does not match the receipt for 'A-D067-CANARY-C2-VICTIM'.
    An agent may only submit the assignment it was admitted for.

Event and hook output: `../raw_logs/stop_event_d067-canary-c2-substitution.json`,
`../raw_logs/stop_out_d067-canary-c2-substitution.json`.

It is kept here rather than under `results/` because it was never admitted, and
`validate_harness.py` now requires every file in `results/` to carry a consumed
admission receipt. The `A-D067-CANARY-C2-VICTIM` receipt remains `open`.
