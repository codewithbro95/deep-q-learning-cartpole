import matplotlib.pyplot as plt
import numpy as np

class LiveVisualizer:
    def __init__(self, model):
        self.model = model
        
        # Setup the figure (3 plots side-by-side)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.ion() # Interactive mode on
        
        self.titles = ["Input -> Hidden 1", "Hidden 1 -> Hidden 2", "Hidden 2 -> Output"]
        self.images = []
        
        # Initial draw to set up the heatmaps
        layers = [model.layer1, model.layer2, model.layer3]
        for i, ax in enumerate(self.axes):
            # We detach the weights from PyTorch to plot them as numpy arrays
            weights = layers[i].weight.detach().cpu().numpy()
            
            # Use a Red-Yellow-Green colormap (RdYlGn)
            # Red = Negative, Green = Positive
            img = ax.imshow(weights, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
            
            ax.set_title(self.titles[i])
            self.images.append(img)
            
        self.fig.colorbar(self.images[0], ax=self.axes[0], fraction=0.046, pad=0.04)
        self.fig.tight_layout()

    def update(self):
        layers = [self.model.layer1, self.model.layer2, self.model.layer3]
        
        for i, img in enumerate(self.images):
            weights = layers[i].weight.detach().cpu().numpy()
            img.set_data(weights)
            # Optional: Dynamic scaling to see contrast better
            # img.set_clim(vmin=weights.min(), vmax=weights.max())
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()