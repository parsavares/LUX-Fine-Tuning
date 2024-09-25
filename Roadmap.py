import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Define phases and the corresponding paper titles for each phase
phases = [
    "Phase 1: Fine-Tuning Whisper for Luxembourgish (2 Papers)",
    "Phase 2: Master's Thesis – Voting System with Democratic Agents",
    "Phase 3: PhD Research – Democratic Multi-Agent Systems for Real-Time Processing"
]

papers_phase1 = [
    "Paper 1: Fine-Tuning Whisper for Luxembourgish",
    "Paper 2: Integrating Punctuation and Spell-Checker Agents"
]

papers_phase2 = [
    "Thesis: Multi-Agent Voting System for Luxembourgish ASR",
    "Extra Paper: Democracy in Spelling & Punctuation Correction"
]

papers_phase3 = [
    "Paper 3: Democratic Voting in Real-Time ASR",
    "Paper 4: Real-Time Democratic Multi-Agent Systems for Luxembourgish"
]

# Colors for each phase
colors = ['#ff9999', '#66b3ff', '#99ff99']

# Coordinates for drawing
y_start = 8
box_height = 0.7
gap = 1.5
paper_gap = 0.8

# Function to draw a phase and its papers
def draw_phase(ax, phase_title, papers, color, y_pos):
    # Draw the phase box
    ax.add_patch(mpatches.FancyBboxPatch((0.1, y_pos), 0.8, box_height, 
                                         boxstyle="round,pad=0.3", edgecolor='black', facecolor=color, lw=2))
    ax.text(0.5, y_pos + box_height / 2, phase_title, ha="center", va="center", fontsize=12, weight='bold')
    
    # Draw the papers related to the phase
    for i, paper in enumerate(papers):
        paper_y = y_pos - (i + 1) * paper_gap
        ax.add_patch(mpatches.FancyBboxPatch((0.15, paper_y), 0.7, box_height, 
                                             boxstyle="round,pad=0.3", edgecolor='black', facecolor='#f2f2f2', lw=2))
        ax.text(0.5, paper_y + box_height / 2, paper, ha="center", va="center", fontsize=10)

# Draw phases and papers
y_pos = y_start
draw_phase(ax, phases[0], papers_phase1, colors[0], y_pos)

y_pos -= gap + len(papers_phase1) * paper_gap
draw_phase(ax, phases[1], papers_phase2, colors[1], y_pos)

y_pos -= gap + len(papers_phase2) * paper_gap
draw_phase(ax, phases[2], papers_phase3, colors[2], y_pos)

# Set limits and remove axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
plt.title("10/10 Research Roadmap", fontsize=16, weight='bold')

# Show the plot
plt.tight_layout()
plt.show()
