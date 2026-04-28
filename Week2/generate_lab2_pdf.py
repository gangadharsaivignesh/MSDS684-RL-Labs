from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.colors import HexColor, black, grey, white
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Gangadhar_Saivignesh_Lab2.pdf")

doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=letter,
    topMargin=0.75*inch,
    bottomMargin=0.75*inch,
    leftMargin=1*inch,
    rightMargin=1*inch,
)

styles = getSampleStyleSheet()

# Custom styles
styles.add(ParagraphStyle(
    name='MainTitle',
    parent=styles['Title'],
    fontSize=18,
    spaceAfter=6,
    textColor=HexColor('#1a1a2e'),
))
styles.add(ParagraphStyle(
    name='SubTitle',
    parent=styles['Normal'],
    fontSize=11,
    alignment=TA_CENTER,
    spaceAfter=20,
    textColor=HexColor('#444444'),
))
styles.add(ParagraphStyle(
    name='SectionHead',
    parent=styles['Heading1'],
    fontSize=14,
    spaceBefore=18,
    spaceAfter=8,
    textColor=HexColor('#16213e'),
    borderWidth=1,
    borderColor=HexColor('#16213e'),
    borderPadding=4,
))
styles.add(ParagraphStyle(
    name='SubHead',
    parent=styles['Heading2'],
    fontSize=12,
    spaceBefore=12,
    spaceAfter=6,
    textColor=HexColor('#0f3460'),
))
styles.add(ParagraphStyle(
    name='BodyText2',
    parent=styles['Normal'],
    fontSize=10.5,
    leading=15,
    alignment=TA_JUSTIFY,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name='BulletText',
    parent=styles['Normal'],
    fontSize=10.5,
    leading=15,
    leftIndent=20,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name='Caption',
    parent=styles['Normal'],
    fontSize=9.5,
    leading=13,
    textColor=HexColor('#333333'),
    alignment=TA_JUSTIFY,
    spaceBefore=4,
    spaceAfter=12,
    leftIndent=12,
    rightIndent=12,
))

story = []

# ── Title ──
story.append(Paragraph("Lab 2: Markov Decision Processes &amp; Dynamic Programming", styles['MainTitle']))
story.append(Paragraph("MSDS 684 — Reinforcement Learning | Regis University", styles['SubTitle']))
story.append(Paragraph("Student: Saivignesh Gangadhar &nbsp;&nbsp;|&nbsp;&nbsp; Date: April 17, 2026", styles['SubTitle']))
story.append(Spacer(1, 12))

# ═══════════════════════════════════════════
# SECTION 1: PROJECT OVERVIEW
# ═══════════════════════════════════════════
story.append(Paragraph("Section 1: Project Overview", styles['SectionHead']))

s1_paragraphs = [
    "This lab investigates how an agent can compute optimal behavior in fully known environments "
    "using Dynamic Programming (DP), one of the foundational approaches in reinforcement learning. "
    "The central problem is the <i>planning problem</i>: given a complete model of the environment as "
    "a Markov Decision Process (MDP), how can we efficiently compute the value function V(s) and "
    "derive an optimal policy \u03c0* without ever interacting with the environment through trial and "
    "error? This connects directly to Sutton &amp; Barto Chapter 4, which presents DP as the bridge "
    "between the theoretical Bellman equations of Chapter 3 and the model-free methods introduced in "
    "later chapters.",

    "The environment used is a custom 4\u00d74 GridWorld built to follow the Gymnasium API. The state "
    "space is discrete with 16 states (one per grid cell), and the action space consists of four "
    "discrete actions: UP, RIGHT, DOWN, and LEFT. The reward structure includes a goal state (state 15) "
    "with reward +1.0, a trap state (state 11) with reward \u22121.0, and a step penalty of \u22120.04 for all "
    "non-terminal transitions, which incentivizes the agent to reach the goal quickly. An obstacle at "
    "state 5 blocks movement entirely. Episodes terminate when the agent reaches the goal or falls "
    "into the trap.",

    "Two configurations of this environment are tested. The first is deterministic, where each action "
    "moves the agent in the intended direction with probability 1.0. The second is stochastic, where "
    "the agent moves in the intended direction with probability 0.8 and slips to each perpendicular "
    "direction with probability 0.1. This stochastic setting mirrors the classic \"slippery\" dynamics "
    "discussed in Sutton &amp; Barto Section 4.1, where the environment's uncertainty forces the agent "
    "to adopt more cautious policies.",

    "Two DP algorithms are implemented: Policy Iteration (PI) and Value Iteration (VI). Policy "
    "Iteration alternates between full policy evaluation\u2014computing V(s) for a fixed policy using the "
    "Bellman expectation equation\u2014and policy improvement, which greedily updates the policy based on "
    "the computed values. The Policy Improvement Theorem (Sutton &amp; Barto, Section 4.2) guarantees "
    "that each greedy step produces a policy at least as good as the previous one. Value Iteration "
    "combines both steps into a single Bellman optimality backup per state, effectively truncating the "
    "evaluation step to a single sweep. Both algorithms are implemented in synchronous (two-array) and "
    "in-place (single-array) variants, and both are additionally applied to Gymnasium's FrozenLake-v1 "
    "environment to demonstrate generalization.",

    "I hypothesize that Value Iteration will converge in fewer total sweeps than Policy Iteration on "
    "these small grids, that in-place updates will converge faster than synchronous due to immediate "
    "propagation of updated values, and that stochastic dynamics will require more iterations because "
    "uncertainty slows value propagation.",
]
for p in s1_paragraphs:
    story.append(Paragraph(p, styles['BodyText2']))

story.append(PageBreak())

# ═══════════════════════════════════════════
# SECTION 2: DELIVERABLES
# ═══════════════════════════════════════════
story.append(Paragraph("Section 2: Deliverables", styles['SectionHead']))
story.append(Spacer(1, 4))
story.append(Paragraph(
    '<b>GitHub Repository:</b> '
    '<a href="https://github.com/gangadharsaivignesh/MSDS684-RL-Labs/tree/main" color="blue">'
    'https://github.com/gangadharsaivignesh/MSDS684-RL-Labs/tree/main</a>',
    styles['BodyText2']
))

story.append(Paragraph("Implementation Summary", styles['SubHead']))
story.append(Paragraph(
    "I implemented a custom GridWorld environment class following the Gymnasium API, with configurable "
    "grid size, start state, goal and trap states, obstacles, step reward, and slip probability. The "
    "full transition model P(s\u2032,r|s,a) is built at initialization, enabling direct use by DP algorithms. "
    "Policy Evaluation, Policy Iteration, and Value Iteration were each coded in both synchronous and "
    "in-place variants using NumPy arrays for V(s) and \u03c0(s). Experiments used \u03b3=0.99 and convergence "
    "threshold \u03b8=1e-8. All four algorithm variants were tested on both the deterministic and stochastic "
    "GridWorld configurations, and Policy Iteration and Value Iteration were additionally applied to "
    "Gymnasium's FrozenLake-v1 (4\u00d74, slippery) via a lightweight adapter class.",
    styles['BodyText2']
))

story.append(Paragraph("Key Results &amp; Analysis", styles['SubHead']))

# Figure 1
story.append(Paragraph("<b>Figure 1: Value Function Heatmaps Over Iterations (Deterministic GridWorld, Value Iteration)</b>", styles['BodyText2']))
story.append(Paragraph(
    "This sequence of heatmaps shows V(s) evolving from all zeros to the converged optimal values "
    "across 7 iterations of synchronous Value Iteration. The value signal originates at the goal state "
    "(bottom-right, V=1.0) and propagates backward through the grid, with each iteration extending the "
    "reach of the goal's influence by one step. The obstacle at state 5 creates a visible gap in value "
    "propagation, forcing the agent to route around it. By iteration 7, convergence is complete, and "
    "the value gradient clearly points from start (top-left, V=0.755) toward the goal. This backward "
    "propagation of value is precisely the mechanism described in Sutton &amp; Barto Section 4.4, where "
    "the Bellman optimality backup at each state pulls in the best available value from neighboring states.",
    styles['Caption']
))

# Figure 2
story.append(Paragraph("<b>Figure 2: Optimal Policy Comparison (Deterministic vs. Stochastic GridWorld)</b>", styles['BodyText2']))
story.append(Paragraph(
    "These quiver plots overlay policy arrows on the value function heatmap. In the deterministic case, "
    "the optimal policy takes the shortest path from start to goal, routing around the obstacle and "
    "passing directly adjacent to the trap because there is zero risk of slipping into it. In the "
    "stochastic case, the policy changes: the agent avoids states adjacent to the trap because a 10% "
    "slip probability creates meaningful risk. This illustrates a key insight from Sutton &amp; Barto "
    "Section 4.1: when transitions are stochastic, the optimal policy must account for the "
    "probability-weighted outcomes of all possible next states, not just the intended one. The stochastic "
    "policy is safer but may take longer paths, reflecting a trade-off between speed and risk.",
    styles['Caption']
))

# Figure 3
story.append(Paragraph("<b>Figure 3: Convergence Curves (Max Bellman Error per Iteration)</b>", styles['BodyText2']))
story.append(Paragraph(
    "These log-scale plots show the maximum absolute change in V(s) at each iteration for both VI "
    "variants across all three environments. In the deterministic GridWorld, both synchronous and "
    "in-place VI converge in exactly 7 iterations. On the stochastic GridWorld, in-place VI converges "
    "in 35 iterations versus 60 for synchronous\u2014a 42% reduction. This confirms the theoretical "
    "expectation from Sutton &amp; Barto Section 4.5: in-place updates propagate information faster "
    "because each state immediately benefits from its neighbors' updated values within the same sweep. "
    "FrozenLake's strong stochasticity (uniform 1/3 split) requires 438 iterations, reflecting how "
    "high transition uncertainty slows convergence.",
    styles['Caption']
))

# Figure 4
story.append(Paragraph("<b>Figure 4: Convergence Comparison Bar Chart (All Algorithms)</b>", styles['BodyText2']))
story.append(Paragraph(
    "This bar chart compares total sweeps and wall-clock time. Value Iteration (7 sweeps, ~0.35ms) "
    "dramatically outperforms Policy Iteration (7,577 sweeps, ~53ms) on total computation. While PI "
    "converges in only 6 policy improvement steps, each step requires a full policy evaluation loop "
    "that runs for hundreds of iterations. This aligns with Sutton &amp; Barto Section 4.6 on "
    "Generalized Policy Iteration: both PI and VI are instances of GPI, alternating between evaluation "
    "and improvement, but VI's strategy of truncating evaluation to a single sweep proves far more "
    "efficient on these small grids.",
    styles['Caption']
))

# Summary Table
story.append(Paragraph("Summary Table", styles['SubHead']))
table_data = [
    ['Environment', 'Algorithm', 'Total Sweeps', 'Time (ms)'],
    ['Det. GridWorld', 'PI (sync)', '7,577', '52.80'],
    ['Det. GridWorld', 'PI (in-place)', '6,070', '40.63'],
    ['Det. GridWorld', 'VI (sync)', '7', '0.36'],
    ['Det. GridWorld', 'VI (in-place)', '7', '0.34'],
    ['Stoch. GridWorld', 'PI (sync)', '2,442', '26.63'],
    ['Stoch. GridWorld', 'PI (in-place)', '2,894', '36.33'],
    ['Stoch. GridWorld', 'VI (sync)', '60', '4.46'],
    ['Stoch. GridWorld', 'VI (in-place)', '35', '2.60'],
    ['FrozenLake', 'PI (sync)', '1,529', '16.42'],
    ['FrozenLake', 'VI (sync)', '438', '25.87'],
]
t = Table(table_data, colWidths=[1.6*inch, 1.3*inch, 1.2*inch, 1.0*inch])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f5f5f5'), white]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Spacer(1, 8))

story.append(Paragraph(
    "An unexpected result was that in-place Policy Iteration on the stochastic grid was slightly slower "
    "(2,894 sweeps) than synchronous PI (2,442 sweeps). This occurred because in-place evaluation can "
    "converge differently depending on state ordering, and the particular sweep order in this grid caused "
    "slightly longer stabilization at certain policy evaluation steps.",
    styles['BodyText2']
))

story.append(PageBreak())

# ═══════════════════════════════════════════
# SECTION 3: AI USE REFLECTION
# ═══════════════════════════════════════════
story.append(Paragraph("Section 3: AI Use Reflection", styles['SectionHead']))

story.append(Paragraph("Initial Interaction", styles['SubHead']))
story.append(Paragraph(
    "I began by asking Claude to explain the foundational concepts I needed before implementing the lab: "
    "MDPs, the Bellman equations, Policy Evaluation, Policy Iteration, and Value Iteration. Claude provided "
    "detailed explanations with two to three worked examples for each concept, walking through how the "
    "Bellman expectation equation computes V(s) for a fixed policy and how the Bellman optimality equation "
    "underlies Value Iteration. This conceptual grounding helped me connect the mathematical formulations "
    "in Sutton &amp; Barto Chapters 3\u20134 to concrete implementation steps before writing any code.",
    styles['BodyText2']
))

story.append(Paragraph("Iteration Cycles", styles['SubHead']))

story.append(Paragraph(
    "<b>Iteration 1 \u2014 Transition Model Design:</b> When building the GridWorld class, my initial "
    "implementation only stored the grid layout without precomputing transition probabilities. I asked "
    "Claude how DP algorithms access the environment model, and it explained that DP requires the full "
    "transition model P(s\u2032,r|s,a) as a nested dictionary structure: P[s][a] returns a list of "
    "(probability, next_state, reward, done) tuples. I restructured the class to build this model at "
    "initialization via _build_transition_model(), which made the DP functions clean and efficient. "
    "This taught me that the distinction between model-based and model-free RL is not just theoretical\u2014"
    "it directly shapes how you design the environment interface.",
    styles['BodyText2']
))

story.append(Paragraph(
    "<b>Iteration 2 \u2014 FrozenLake Action Mapping Mismatch:</b> When visualizing the FrozenLake optimal "
    "policy, the arrows pointed in wrong directions. I compared the output with what I expected from the "
    "value function and realized something was off. After investigating, I discovered that FrozenLake uses "
    "a different action encoding (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP) than our GridWorld (0=UP, 1=RIGHT, "
    "2=DOWN, 3=LEFT). I asked Claude to confirm, and it provided a remapping dictionary to translate "
    "between the two action spaces. After applying the remap, the policy arrows correctly matched the "
    "value landscape. This highlighted that RL environments do not share a universal action encoding, "
    "and adapting between them requires careful attention to documentation.",
    styles['BodyText2']
))

story.append(Paragraph(
    "<b>Iteration 3 \u2014 Stochastic Transition Probabilities:</b> My first attempt at implementing "
    "stochastic transitions double-counted outcomes when the intended direction and a perpendicular "
    "direction led to the same next state (e.g., when adjacent to a wall). The value function showed "
    "unexpected asymmetries near boundaries. I asked Claude to review the logic, and it suggested "
    "aggregating probabilities by outcome using a dictionary keyed on (next_state, reward, done) "
    "before appending to the transitions list. This fixed the probability normalization issue and "
    "produced correct symmetric values near walls, matching the expected behavior from Sutton &amp; "
    "Barto Section 4.1.",
    styles['BodyText2']
))

story.append(Paragraph("Critical Evaluation", styles['SubHead']))
story.append(Paragraph(
    "I independently verified that the optimal value functions from PI and VI matched to numerical "
    "precision, confirming both algorithms converged to the same solution. I also cross-referenced "
    "the FrozenLake action mapping with Gymnasium's official documentation rather than relying solely "
    "on Claude's suggestion. Claude's initial approach of discussing design before coding significantly "
    "reduced the number of bugs, but I learned to always test edge cases myself, particularly around "
    "boundary states and transition probability normalization.",
    styles['BodyText2']
))

story.append(Paragraph("Learning Reflection", styles['SubHead']))
story.append(Paragraph(
    "The transition model debugging taught me that DP's requirement for a complete model is not just "
    "a theoretical constraint\u2014building and verifying that model is a significant implementation "
    "challenge. The action mapping issue reinforced the importance of checking assumptions when adapting "
    "code across environments. Most importantly, I learned that discussing design thoroughly with AI "
    "before coding is far more effective than generating code first and debugging after.",
    styles['BodyText2']
))

story.append(PageBreak())

# ═══════════════════════════════════════════
# SECTION 4: SPEAKER NOTES
# ═══════════════════════════════════════════
story.append(Paragraph("Section 4: Speaker Notes (~5 Minutes)", styles['SectionHead']))
story.append(Spacer(1, 8))

bullets = [
    "<b>Problem:</b> Solve a 4\u00d74 GridWorld MDP using Dynamic Programming. Given a complete "
    "transition model P(s\u2032,r|s,a), compute optimal value functions and policies without any "
    "environment interaction, applying the Bellman equations from Sutton &amp; Barto Chapter 4.",

    "<b>Method:</b> Implemented Policy Iteration (alternating full evaluation and greedy improvement) "
    "and Value Iteration (single Bellman optimality backup per sweep), each in synchronous and in-place "
    "variants. Tested on deterministic GridWorld, stochastic GridWorld (slip_prob=0.1), and Gymnasium's "
    "FrozenLake-v1.",

    "<b>Key Design Choice:</b> Built the GridWorld as a full Gymnasium-compatible environment with a "
    "precomputed transition model, enabling direct use by DP algorithms. Created a lightweight adapter "
    "for FrozenLake to reuse the same DP functions without code duplication.",

    "<b>Main Result:</b> Value Iteration converged in 7 sweeps versus Policy Iteration's 7,577 on the "
    "deterministic grid. Stochastic dynamics increased VI iterations from 7 to 60 (synchronous) and "
    "changed the optimal policy to avoid trap-adjacent states due to slip risk.",

    "<b>Key Insight:</b> In-place updates reduced stochastic VI iterations by 42% (60 to 35) because "
    "updated values propagate immediately within a sweep. However, on deterministic grids both variants "
    "were identical, showing the advantage is environment-dependent.",

    "<b>Challenge:</b> Debugging the FrozenLake action mapping mismatch and stochastic transition "
    "probability aggregation required careful cross-referencing of documentation and independent "
    "verification of results.",

    "<b>Connection to Future Work:</b> DP requires a known model, which is rarely available in practice. "
    "Weeks 3\u20134 introduce model-free methods (Monte Carlo and TD learning) that learn from experience "
    "instead, but the Bellman equations underlying DP remain the theoretical foundation for those algorithms.",
]

for b in bullets:
    story.append(Paragraph("\u2022 " + b, styles['BulletText']))
    story.append(Spacer(1, 6))

# Build PDF
doc.build(story)
print(f"PDF generated: {OUTPUT_PATH}")
