# Print-size check (2026-10-21)

Assumed placement: single-column panels at 3.4 in (244.8 pt) width; Fig 4a-c three across in a 7.16-in double column (171.8 pt each); legends at 1:1. Scale = min(1, intended width / PDF width); printed font = smallest font in the PDF x scale (7 pt = tick/legend/annotation size; 6.5 pt = body text of the concept figure).
Changes made on 2026-10-21 (data, colours, markers, file names unchanged): Fig 1 point labels 5.5 -> 7 pt (PDF size unchanged); Fig 2a-c figsize 3.4 -> 2.1 in with the x-label shortened and Fig 5 3.4 -> 2.8 in so that the PDF including tick labels is <= 244.8 pt; Fig 4a-c figsize 2.6 -> 2.15 in with x-labels wrapped so that three panels fit a double column at 1:1.
For an 8-pt rule (e.g. IEEE) every panel must be placed at >= 1.15x its PDF width (fig0: 1.23x), or the tick/legend fonts raised from 7 to 8 pt in scripts/80 (not done: journal not yet chosen).

| file | PDF size (pt) | intended width (pt) | scale | min font in PDF (pt) | printed (pt) | >= 7 pt | >= 8 pt |
|---|---|---|---|---|---|---|---|
| fig0_concept.pdf | 246.0 x 180.0 | 245 | 0.995 | 6.5 | 6.47 | NO | place at >= 1.24x |
| fig1_legend.pdf | 135.0 x 37.2 | 1:1 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig1_reliability.pdf | 246.0 x 233.4 | 245 | 0.995 | 7.0 | 6.97 | yes | place at >= 1.15x |
| fig2_legend.pdf | 283.4 x 25.6 | 1:1 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig2a_transfer_chirkov.pdf | 241.5 x 166.9 | 245 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig2b_transfer_wu.pdf | 236.0 x 166.9 | 245 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig2c_transfer_hawkins.pdf | 228.0 x 166.9 | 245 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig3_roles_decomposition.pdf | 241.3 x 189.3 | 245 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig4a_group_chirkov.pdf | 153.9 x 160.1 | 172 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig4b_group_wu.pdf | 153.9 x 160.1 | 172 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig4c_group_hawkins.pdf | 153.9 x 160.3 | 172 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig5_composition.pdf | 244.6 x 144.7 | 245 | 1.000 | 7.0 | 7.00 | yes | place at >= 1.14x |
| fig5_composition_pooled.pdf | 246.3 x 166.9 | 245 | 0.994 | 7.0 | 6.96 | yes | place at >= 1.15x |
