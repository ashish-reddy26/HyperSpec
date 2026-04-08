graph TD
    A[Start: Global Tracks & Current Frame Pods] --> B[KD-Tree: Find centroid neighbors within radius]
    B --> C{Are there neighbors?}
    C -- No --> L[Mark as Unmatched New Pod]
    C -- Yes --> D[Compute Symmetric Overlap Score]
    D --> E[Hungarian Algorithm: Resolve conflicts & assign optimal pairs]
    
    E --> F{Overlap Score >= 50%?}
    F -- Yes --> G[MATCH ACCEPTED]
    F -- No --> H[Fallback: Compute Chamfer Distance]
    
    H --> I{Chamfer < Threshold?}
    I -- Yes --> G
    I -- No --> J[MATCH REJECTED]
    
    J --> L
    
    G --> K[Merge point clouds, unify color, downsample]
    K --> M[Update Global Point Cloud]
    L --> N[Assign New Track ID & Color]
    N --> M
