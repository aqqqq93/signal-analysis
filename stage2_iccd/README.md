# Stage 2: ICCD Parameter Estimation

This folder is reserved for the second stage of the project.

Planned inputs from stage 1:

- IF curves;
- ridge heatmaps or confidence maps;
- top-2 candidate IF curves;
- route probabilities and selected expert labels;
- uncertainty metrics for deciding whether an IF estimate is suitable for ICCD initialization.

Planned work:

- build an ICCD parameter-estimation interface initialized by IF-Net output;
- compare convergence from neural IF initialization versus traditional ridge initialization;
- define acceptance criteria for entering full reconstruction.

