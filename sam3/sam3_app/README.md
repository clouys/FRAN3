# SAM3 Segmentation Studio — Flask Web App

Replaces the Jupyter widget with a proper web UI that supports native right-click
context menus, beautiful dark-theme design, and all the original functionality.

## Setup

```bash
# 1. Activate your SAM3 conda environment
conda activate sam3

# 2. Install the extra web dependency (all others should already be present)
pip install flask

# 3. Run the server
cd sam3_app
python app.py
```

Then open **http://localhost:5000** in your browser.

## Features

| Feature | How it works |
|---------|-------------|
| Upload image | Drag-and-drop or file picker |
| Load from URL | Paste any image URL |
| Text prompt | Type and press Enter or click Segment |
| Box prompts | Draw positive (green) or negative (red) boxes directly on the canvas |
| Confidence slider | Real-time threshold adjustment |
| **Right-click context menu** | Right-click any mask on the canvas OR in the Masks panel — pick a species from a native browser menu |
| Species management | Add species in the right panel; they appear instantly in the context menu |
| Export PNG + GPKG | Choose a folder or leave blank for auto-timestamped folder |

## Architecture

```
browser  ──AJAX──►  Flask (app.py)  ──►  SAM3 processor (in-process)
   ▲                     │
   └── HTML/CSS/JS ◄─────┘  (rendered overlay images returned as base64 PNG)
```

All SAM3 processing happens server-side. The browser only handles:
- Canvas drawing (image, overlay, box preview)
- Native right-click context menu
- Species assignment UI
