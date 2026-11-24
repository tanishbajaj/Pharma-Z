import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('#0F1117')
ax.set_facecolor('#0F1117')

# ── helpers ─────────────────────────────────────────────────────
def box(x, y, w, h, bg, title, subtitle=''):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0.12",
                       facecolor=bg, edgecolor='#718096', lw=1.8, zorder=3)
    ax.add_patch(p)
    dy = 0.22 if subtitle else 0
    ax.text(x, y+dy, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=4)
    if subtitle:
        ax.text(x, y-0.28, subtitle, ha='center', va='center',
                fontsize=8.5, color='#A0AEC0', zorder=4)

def arr(x1, x2, y=3.5, color='#718096', lbl=''):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.2), zorder=2)
    if lbl:
        ax.text((x1+x2)/2, y+0.28, lbl, ha='center', va='center',
                fontsize=8, color=color, style='italic')

# ── Title ────────────────────────────────────────────────────────
ax.text(8, 6.55, 'Graph Variational Autoencoder — High-Level Architecture',
        ha='center', fontsize=15, fontweight='bold', color='white')
ax.text(8, 6.1, 'Pharma-Z  |  Molecular Generation Pipeline',
        ha='center', fontsize=10, color='#718096')
ax.plot([0.4, 15.6], [5.82, 5.82], color='#4A5568', lw=0.8)

# ── 5 main blocks ────────────────────────────────────────────────
#          x     y    w    h   bg         title           subtitle
blocks = [
    (1.5,  3.5, 2.2, 1.6, '#1E3A5F', '🧪 Input\nMolecule',   'Atoms + Bonds\nas Graph'),
    (4.5,  3.5, 2.2, 1.6, '#1B4332', '📡 Encoder',            '3× R-GIN Layers\nMessage Passing'),
    (8.0,  3.5, 2.6, 1.6, '#3D1A6E', '🌀 Latent Space',       'z = μ + σ·ε\nz ~ N(μ, σ²)'),
    (11.5, 3.5, 2.2, 1.6, '#7B2D00', '🔓 Decoder',            'Predict Nodes\nEdges & Bonds'),
    (14.5, 3.5, 2.2, 1.6, '#1A3A4A', '✅ Generated\nMolecule', 'Valid SMILES\nvia RDKit'),
]
for b in blocks:
    box(*b)

# ── Arrows between blocks ────────────────────────────────────────
arr(2.62, 3.38, lbl='Graph\n(X, A)')
arr(5.62, 6.68, lbl='μ, σ')
arr(9.32, 10.38, lbl='z')
arr(12.62, 13.38, lbl='Â, X̂')

# ── KL + Recon loss arrows (bottom) ─────────────────────────────
# reconstruction loss arc
ax.annotate('', xy=(5.0, 2.55), xytext=(11.5, 2.55),
            arrowprops=dict(arrowstyle='<->', color='#F6AD55', lw=1.6,
                            connectionstyle='arc3,rad=0.0'), zorder=2)
ax.text(8.0, 2.2, 'Reconstruction Loss  =  BCE(A, Â) + CE(X, X̂)',
        ha='center', fontsize=9, color='#F6AD55')

# KL loss
ax.annotate('', xy=(6.9, 1.55), xytext=(9.1, 1.55),
            arrowprops=dict(arrowstyle='<->', color='#F687B3', lw=1.6), zorder=2)
ax.text(8.0, 1.2, 'KL Loss  =  D_KL[ q(z|G) || N(0,I) ]',
        ha='center', fontsize=9, color='#F687B3')

ax.text(8.0, 0.65, 'Total Loss  =  L_recon  +  β · L_KL',
        ha='center', fontsize=10, fontweight='bold', color='#68D391')

# ── Vertical dashed lines to loss section ────────────────────────
for x in [4.5, 8.0, 11.5]:
    ax.plot([x, x], [2.68, 2.72], color='#4A5568', lw=1, ls='--')

plt.tight_layout(pad=0.4)
plt.savefig('graphvae_highlevel.png', dpi=180,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: graphvae_highlevel.png")
