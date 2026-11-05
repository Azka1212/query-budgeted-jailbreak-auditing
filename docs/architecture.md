# Framework architecture

This Mermaid diagram summarizes the manuscript interface. It is a repository-native redraw, not the paper's original artwork.

```mermaid
flowchart TD
    X["Seed objective and category"] --> S["Numerical state: category, operator context, seed length, Success input"]
    S --> A["SAC controller"]
    A --> H["Discrete operator and length, temperature, persona controls"]
    X --> W["Frozen prompt rewriter"]
    H --> W
    W --> U["Realized test prompt"]
    U --> T["Frozen black-box target"]
    T --> Y["Target response"]
    U --> J["Frozen structured judge"]
    Y --> J
    J --> M["Success, Stealth, Novelty, Efficiency, Impact"]
    M --> R["Success-gated reward"]
    R --> B["Replay and actor/critic updates during training"]
    B -.-> A
    M --> F["Success input for next evaluation attempt"]
    F --> S
    M --> Q["Evaluation stops at success or target-query cap"]
```

During evaluation, all components are frozen. Rewriter and judge calls are auxiliary; submissions to the target count toward the target-query budget. The two inspected controller-feedback checkpoints have one-step terminal training records; the evaluation interface supports repeated attempts.

The manuscript references `Figure-Page-4.drawio.pdf`. That original asset was not present in the available workspace and has not been substituted with an unrelated figure.
