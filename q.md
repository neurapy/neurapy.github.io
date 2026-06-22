
How one training point changes a PINN prediction

A physics-informed neural network learns a solution by minimizing physics, initial-condition, and boundary-condition losses. PINNfluence estimates how a trained model would react if one training point were perturbed, without running a full retraining experiment.

****
****

PINN Plot:
> Make the points right
> Add time/space axis?

****
****

Core estimate ->
Influence as local sensitivity -> The core formula

****
Read the Formula Right to left and see how a Loss at a point gets transformed through Parameter Space ->
****

1. Training-Point Gradient
How does the selected training point pull on the model parameters through its loss contribution.
-> Parameter Space

2. Local Training Geometry
 Given the tiny loss push from x, what parameter shift would the trained landscape allow? H⁻¹ applies that reverse map; steep directions shrink, flat directions carry more.
->

****
****
--- Wait: In Indicators: woher kommen die Influence Indicator zahlen her
****
****

Letzter Satz hinzufügen:
It also applies only to the equilibrium the Model found.
Retraining without the Training-Point in question could push the Model into an entierly different equilibrium.
