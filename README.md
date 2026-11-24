# Scalizer

**A what-if simulator for cloud autoscaling.** Scalizer lets IT managers, cloud architects and DevOps teams test autoscaling policies on real traffic before applying them in production. It compares the current policy (the baseline) with alternative policies under different traffic scenarios, showing cost, latency, availability and SLA violations for each. It can also recommend the best policy, which a person must approve.

Project for the **Complex Systems Design** course, puzzle piece **P21 – Intelligent “What if?” scenario simulator**.

**Team:** Francesco Palozzi · Luca Magrini · Victoria Coacci

📊 **Presentation (6 November 2026):** [Figma](https://www.figma.com/design/W2I3AlcYRpVWjARFoM6Acj/Scalizer?node-id=16-33&t=INQkUcJtrWhNMaiA-1)

## Repository structure

```
Scalizer/
├── README.md
├── docs/                         Project documentation, in reading order
│   ├── 01_Project_Idea.md
│   ├── 02_Project_Description.md
│   ├── 03_Application_Domain.md    domain, actors, Alibaba dataset
│   ├── 04_Architecture.md          two modes, data flow
│   ├── 05_Components.md            the seven components
│   └── 06_Final_Decisions.md       decisions, success criteria
├── diagrams/
│   ├── architecture/             High-level architecture (draw.io source + PNG)
│   └── uml/                      Component diagram, sequence diagrams (PNG)
├── planning/
│   ├── Scalizer_Gantt.html       Editable Gantt: open it in a browser
│   ├── Scalizer_Gantt.png
│   └── TODO.md
└── presentation/
    └── Scalizer_Presentation.pptx   Offline copy of the slides
```

## Where to start

| I want to… | Open |
| --- | --- |
| Understand the project in 2 minutes | [`docs/02_Project_Description.md`](docs/02_Project_Description.md) |
| See the architecture | [`docs/04_Architecture.md`](docs/04_Architecture.md) and [`diagrams/uml/Component_Diagram.png`](diagrams/uml/Component_Diagram.png) |
| See how the system behaves | [`diagrams/uml/Sequence_WhatIf.png`](diagrams/uml/Sequence_WhatIf.png), [`diagrams/uml/Sequence_Optimization.png`](diagrams/uml/Sequence_Optimization.png) |
| Check or change the plan | [`planning/Scalizer_Gantt.html`](planning/Scalizer_Gantt.html) (download it and open it in a browser) |

## Editing the diagrams

- **Gantt:** open `planning/Scalizer_Gantt.html` in a browser. Drag the bars or use the table, then click **Save as HTML file** and replace the file in the repository. **Download PNG** gives the image for the slides.
- **Architecture overview:** open `diagrams/architecture/Architecture_Overview.drawio` in [diagrams.net](https://app.diagrams.net).

## Data

[Alibaba Cluster Trace](https://github.com/alibaba/clusterdata), microservices v2022: 13 days of per-minute call rates, response times and CPU/memory usage of production microservices. Published for research and study; see Luo et al., *The Power of Prediction: Microservice Auto Scaling via Workload Learning*, ACM SoCC 2022.

## Workload API

The FastAPI adapter accepts the agreed `services`, `start`, `end` and `version`
inputs, processes workloads in a background job, and serves paginated series and
calibration JSON for the React frontend. It also accepts canonical CSV uploads.

```powershell
py -m pip install -r requirements-api.txt
$env:PYTHONPATH = "$PWD\src"
py -m uvicorn scalizer.api.main:create_app --factory --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/docs>. See [the API contract and local demo guide](docs/API.md)
for Keycloak configuration, request/response examples and inspection of existing
datasets. Simulation endpoints will be added with the Simulation Engine.

