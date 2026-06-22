export const EXPLAINER_TOKENS = [
  {
    id: "score",
    title: "Influence Score",
    body:
      "The score estimates how much a training point x changes the quantity f at location z when that point is perturbed. Large absolute values mean the trained PINN is locally sensitive to that training point.",
  },
  {
    id: "loss",
    title: "Training Loss L",
    body:
      "L is the training objective. In a PINN it is a composite loss made from the PDE residual, initial-condition terms, boundary-condition terms, and any extra measurement terms.",
  },
  {
    id: "quantity",
    title: "Quantity f",
    body:
      "f is what we inspect after training. In the demo this is usually a prediction field or a loss value at a candidate location.",
  },
  {
    id: "train-point",
    title: "Training Point x",
    body:
      "x is a collocation, initial-condition, or boundary-condition point that participated in training. PINNfluence asks how the model would respond if this point were changed.",
  },
  {
    id: "test-point",
    title: "Evaluation Point z",
    body:
      "z is the candidate location where we observe the trained model. One fixed z produces a map over influential training points.",
  },
  {
    id: "sign",
    title: "Direction of Change",
    body:
      "The sign tells whether upweighting the training point tends to increase or decrease f at z. The app often uses absolute influence because signed effects can be noisy and can cancel.",
  },
  {
    id: "grad-f",
    title: "Sensitivity of f",
    body:
      "After H⁻¹ turns the loss push into a parameter shift, this left term projects that shift into the final change of f at z.",
  },
  {
    id: "hessian",
    title: "Local Training Geometry",
    body:
      "The Hessian maps parameter-space movement to loss-gradient change. PINNfluence needs the reverse: given the tiny loss push from x, what parameter shift would the trained landscape allow? H⁻¹ applies that reverse map; steep directions shrink, flat directions carry more.",
  },
  {
    id: "grad-loss",
    title: "Training-Point Gradient",
    body:
      "This gradient describes how the selected training point pulls on the model parameters through its loss contribution.",
  },
  {
    id: "loss-fraction",
    title: "Loss-Term Fraction",
    body:
      "A term fraction measures how much of the absolute influence comes from one loss component, such as the PDE, IC, or BC term.",
  },
  {
    id: "loss-normalizer",
    title: "All Loss Terms",
    body:
      "The denominator sums absolute influence over all loss components. This makes the fractions comparable and keeps them between 0 and 1.",
  },
  {
    id: "cancellation",
    title: "Cancellation κ",
    body:
      "Cancellation rises when signed loss-term influences oppose each other. High cancellation means the fractions should be read together with the signed effects.",
  },
] as const;

export type ExplainerToken = (typeof EXPLAINER_TOKENS)[number];
export type ExplainerTokenId = ExplainerToken["id"];

const DEFAULT_TOKEN_ID: ExplainerTokenId = "score";
const TOKEN_BY_ID: ReadonlyMap<string, ExplainerToken> = new Map(
  EXPLAINER_TOKENS.map((token) => [token.id, token]),
);

export class ExplainerView {
  private activeTokenId: ExplainerTokenId = DEFAULT_TOKEN_ID;

  constructor(private readonly root: HTMLElement) {
    this.render();
    this.root.addEventListener("click", this.handleTokenEvent);
    this.root.addEventListener("focusin", this.handleTokenEvent);
    this.root.addEventListener("pointerover", this.handleTokenEvent);
  }

  activate(): void {
    this.setActiveToken(this.activeTokenId);
  }

  private readonly handleTokenEvent = (event: Event): void => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest<HTMLButtonElement>("[data-explainer-token]");
    if (!button || !this.root.contains(button)) return;
    const id = button.dataset.explainerToken;
    if (id) this.setActiveToken(id);
  };

  private render(): void {
    this.root.innerHTML = `
      <div class="explainer-shell">
        <section class="explainer-panel explainer-hero" aria-labelledby="explainer-title">
          <div class="explainer-hero-copy">
            <p class="explainer-eyebrow">PINNfluence in one pass</p>
            <h2 id="explainer-title">How one training point shapes a PINN prediction</h2>
            <p>
              A physics-informed neural network learns a solution by minimizing physics, initial-condition,
              and boundary-condition losses. PINNfluence estimates how a trained model would react if one
              training point were perturbed, without running a full retraining experiment.
            </p>
          </div>
          <div class="explainer-domain" role="img" aria-label="PINN domain with PDE, initial-condition, and boundary-condition training points">
            <div class="explainer-domain-stage">
              <span class="explainer-domain-band explainer-domain-band-ic">IC</span>
              <span class="explainer-domain-band explainer-domain-band-bc-left">BC</span>
              <span class="explainer-domain-band explainer-domain-band-bc-right">BC</span>
              ${domainPoints()}
              <span class="explainer-domain-query">z</span>
            </div>
            <div class="explainer-domain-legend">
              <span><i class="explainer-dot explainer-dot-pde"></i>PDE residual</span>
              <span><i class="explainer-dot explainer-dot-ic"></i>Initial condition</span>
              <span><i class="explainer-dot explainer-dot-bc"></i>Boundary condition</span>
            </div>
          </div>
        </section>

        <section class="explainer-flow-panel" aria-label="PINNfluence workflow">
          <ol class="explainer-flow">
            ${flowStep("1", "Train", "Fit the PINN on PDE, IC, and BC losses.")}
            ${flowStep("2", "Perturb", "Ask what changes if one training point is upweighted or removed.")}
            ${flowStep("3", "Diagnose", "Use local sensitivities to turn scores into maps, fractions, regions, and indicators.")}
          </ol>
        </section>

        <section class="explainer-panel explainer-formula-panel" aria-labelledby="explainer-formula-title">
          <div class="explainer-panel-header">
            <div>
              <p class="explainer-eyebrow">Core estimate</p>
              <h2 id="explainer-formula-title">Influence as local sensitivity</h2>
            </div>
            <div class="explainer-read-direction" aria-label="Formula reading order">
              <span class="explainer-read-direction-badge">Read the product from right to left!</span>
              <span class="explainer-read-direction-chain" aria-hidden="true">
                <span>change in f</span>
                <i>&lt;-</i>
                <span>landscape response</span>
                <i>&lt;-</i>
                <span>training-point push</span>
              </span>
              <span class="sr-only">change in f from landscape response from training-point push</span>
            </div>
          </div>
          <div class="explainer-formula-workbench">
            <div class="explainer-formula-scroll" aria-label="PINNfluence influence function">
              <div class="explainer-formula" aria-describedby="explainer-token-body">
                <span class="explainer-formula-lhs">
                  <span class="explainer-symbol-stack">
                    ${formulaToken("score", "I", "symbol")}
                    <span class="explainer-symbol-sup">
                      ${formulaToken("loss", "L", "script")}
                      <span class="explainer-script-arrow">→</span>
                      ${formulaToken("quantity", "f", "script")}
                    </span>
                    <span class="explainer-symbol-sub">θ₀</span>
                  </span>
                  <span class="explainer-arguments">
                    <span class="explainer-formula-paren">(</span>
                    ${formulaToken("train-point", "x", "variable")}
                    <span class="explainer-formula-comma">,</span>
                    ${formulaToken("test-point", "z", "variable")}
                    <span class="explainer-formula-paren">)</span>
                  </span>
                </span>
                <span class="explainer-formula-equals">=</span>
                <span class="explainer-formula-rhs">
                  ${formulaToken("sign", "−", "operator")}
                  <span class="explainer-formula-factor">
                    ${formulaTokenMarkup("grad-f", "∇<sub>θ</sub> f(z; θ₀)", "wide")}
                    <sup class="explainer-transpose">T</sup>
                  </span>
                  ${formulaTokenMarkup("hessian", '<span class="explainer-math-atom"><span class="explainer-atom-base">H</span><span class="explainer-atom-sup">−1</span><span class="explainer-atom-sub">θ₀</span></span>', "wide atom")}
                  ${formulaTokenMarkup("grad-loss", "∇<sub>θ</sub> L(x; θ₀)", "wide")}
                </span>
              </div>
            </div>
            <aside class="explainer-token-detail" aria-live="polite">
              <p class="explainer-detail-kicker">Formula component</p>
              <h3 id="explainer-token-title"></h3>
              <p id="explainer-token-body"></p>
            </aside>
          </div>
        </section>

        <section class="explainer-panel explainer-split-panel" aria-labelledby="explainer-split-title">
          <div class="explainer-panel-header">
            <div>
              <p class="explainer-eyebrow">Composite PINN losses</p>
              <h2 id="explainer-split-title">Split influence by PDE, IC, and BC terms</h2>
            </div>
            <p>
              Because influence is linear in the loss term, the same score can be decomposed into
              the constraints that shaped it.
            </p>
          </div>
          <div class="explainer-split-content">
            <div class="explainer-mini-formula">
              <span>r<sub>Li</sub></span>
              <span>=</span>
              <span class="explainer-fraction">
                <span>${formulaToken("loss-fraction", "|Iᵢ|")}</span>
                <span>${formulaToken("loss-normalizer", "Σⱼ |Iⱼ|")}</span>
              </span>
            </div>
            <div class="explainer-loss-bars" aria-label="Example loss-term influence fractions">
              ${lossBar("PDE", 58, "pde")}
              ${lossBar("IC", 24, "ic")}
              ${lossBar("BC", 18, "bc")}
              <button type="button" class="explainer-cancellation" data-explainer-token="cancellation">
                κ cancellation
              </button>
            </div>
          </div>
        </section>

        <section class="explainer-diagnostics" aria-label="PINNfluence diagnostics">
          ${diagnosticTile("Point Map", "Fix z and color training points by |I(x,z)| to locate influential samples.")}
          ${diagnosticTile("Loss Split", "Compare PDE, IC, and BC shares to see which constraints dominate.")}
          ${diagnosticTile("Region Aggregate", "Sum over sets of points to compare how one domain region shapes another.")}
          ${diagnosticTile("Indicator η", "Compress temporal or spatial directionality into a comparable diagnostic number.")}
        </section>

        <section class="explainer-caveat" aria-label="Interpretation caveats">
          <strong>Read as sensitivity.</strong>
          <span>
            PINNfluence is a first-order, local approximation around the trained parameters. Large influence
            identifies sensitivity to a training point or constraint, not a formal causal guarantee. It also
            applies only to the equilibrium the model found. Retraining without the training point in question
            could push the model into an entirely different equilibrium.
          </span>
        </section>
      </div>
    `;
    this.setActiveToken(this.activeTokenId);
  }

  private setActiveToken(id: string): void {
    const token = TOKEN_BY_ID.get(id);
    if (!token) return;
    this.activeTokenId = token.id;
    this.root.querySelectorAll<HTMLButtonElement>("[data-explainer-token]").forEach((button) => {
      const active = button.dataset.explainerToken === token.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const title = this.root.querySelector<HTMLElement>("#explainer-token-title");
    const body = this.root.querySelector<HTMLElement>("#explainer-token-body");
    if (title) title.textContent = token.title;
    if (body) body.textContent = token.body;
  }
}

function formulaToken(id: ExplainerTokenId, label: string, variant = ""): string {
  return formulaTokenMarkup(id, escapeHtml(label), variant, label);
}

function formulaTokenMarkup(id: ExplainerTokenId, html: string, variant = "", fallbackLabel = ""): string {
  const token = TOKEN_BY_ID.get(id);
  const classes = [
    "explainer-formula-token",
    ...variant
      .split(/\s+/)
      .filter(Boolean)
      .map((name) => `explainer-formula-token-${name}`),
  ].join(" ");
  return `
    <button
      type="button"
      class="${classes}"
      data-explainer-token="${escapeHtml(id)}"
      aria-label="${escapeHtml(token?.title ?? fallbackLabel)}"
      aria-pressed="false"
    >${html}</button>
  `;
}

function flowStep(index: string, title: string, body: string): string {
  return `
    <li>
      <span class="explainer-flow-index">${escapeHtml(index)}</span>
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
    </li>
  `;
}

function diagnosticTile(title: string, body: string): string {
  return `
    <article class="explainer-diagnostic">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(body)}</p>
    </article>
  `;
}

function lossBar(label: string, value: number, kind: string): string {
  return `
    <div class="explainer-loss-row">
      <span>${escapeHtml(label)}</span>
      <span class="explainer-loss-track">
        <span class="explainer-loss-fill explainer-loss-fill-${escapeHtml(kind)}" style="--value: ${value}%"></span>
      </span>
      <span>${value}%</span>
    </div>
  `;
}

function domainPoints(): string {
  const points = [
    ["pde", 20, 28],
    ["pde", 38, 38],
    ["pde", 58, 26],
    ["pde", 74, 48],
    ["pde", 30, 62],
    ["pde", 52, 70],
    ["pde", 68, 64],
    ["ic", 18, 86],
    ["ic", 38, 86],
    ["ic", 58, 86],
    ["ic", 78, 86],
    ["bc", 8, 30],
    ["bc", 8, 58],
    ["bc", 92, 34],
    ["bc", 92, 66],
  ] as const;
  return points
    .map(
      ([kind, x, y]) =>
        `<span class="explainer-domain-point explainer-domain-point-${kind}" style="--x: ${x}%; --y: ${y}%"></span>`,
    )
    .join("");
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
