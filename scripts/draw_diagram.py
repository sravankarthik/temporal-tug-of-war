import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 12))
ax.axis('off')

def draw_box(ax, x, y, width, height, text, facecolor='#EAEAF2', edgecolor='black', fontsize=12):
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                 edgecolor=edgecolor, facecolor=facecolor, lw=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, text=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    if text:
        ax.text((x1+x2)/2 + 1.2, (y1+y2)/2, text, ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# Define box dimensions
bw = 5
bh = 1.2

# Coordinates
x_center = 5

y_ds = 10
y_filter = 8
y_noise = 6
y_extract = 4
y_tvs = 2
y_lstm = 0
y_out = -2

draw_box(ax, x_center-bw/2, y_ds, bw, bh, "1. Dataset Ingestion\n(Synthetic, SciQ, PopQA, CounterFact)", facecolor='#e1f5fe')
draw_arrow(ax, x_center, y_ds, x_center, y_filter+bh, "Extract Base Facts")

draw_box(ax, x_center-bw/2, y_filter, bw, bh, "2. Parametric Memory Filter", facecolor='#ffebee')
draw_arrow(ax, x_center, y_filter, x_center, y_noise+bh, "4/5 Seeds Correct\nInject Clean/Fake Context")

draw_box(ax, x_center-bw/2, y_noise, bw, bh, "3. Multi-Seed Denoising\n(N=5 parallel runs, T=50 steps)", facecolor='#e8f5e9')
draw_arrow(ax, x_center, y_noise, x_center, y_extract+bh, "Extract step-by-step history")

draw_box(ax, x_center-bw/2, y_extract, bw, bh, "4. Semantic Extraction\n(SentenceTransformer Embeddings)", facecolor='#f3e5f5')
draw_arrow(ax, x_center, y_extract, x_center, y_tvs+bh, "Pairwise cosine distance")

draw_box(ax, x_center-bw/2, y_tvs, bw, bh, "5. Compute TVS & Velocity\n(2-Channel Temporal Tensor)", facecolor='#fff3e0')
draw_arrow(ax, x_center, y_tvs, x_center, y_lstm+bh, "Input Vector [50 x 2]")

draw_box(ax, x_center-bw/2, y_lstm, bw, bh, "6. Attention-Augmented\nBidirectional LSTM", facecolor='#e1f5fe')
draw_arrow(ax, x_center, y_lstm, x_center, y_out+bh, "Binary Classification")

draw_box(ax, x_center-bw/2, y_out, bw, bh, "7. Conflict Detected (1) / Clean (0)", facecolor='#ffebee')

plt.title("End-to-End Evaluation Pipeline for Knowledge Friction", fontsize=16, fontweight='bold', y=1.02)
plt.xlim(0, 10)
plt.ylim(-3, 12)
plt.savefig('pipeline.png', dpi=300, bbox_inches='tight')
print("Successfully generated pipeline.png")
