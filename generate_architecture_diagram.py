"""
RAiCare System Architecture Diagram Generator
Generates a professional system architecture diagram for the project
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

def create_architecture_diagram():
    # Create figure with larger size for clarity
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    color_frontend = '#61DAFB'  # React blue
    color_backend = '#009688'   # Teal
    color_database = '#4CAF50'  # Green
    color_external = '#FF9800'  # Orange
    color_ml = '#9C27B0'        # Purple
    color_user = '#2196F3'      # Blue
    
    # Title
    title = ax.text(9, 11.5, 'RAiCare System Architecture', 
                    ha='center', va='top', fontsize=24, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='black', linewidth=2))
    
    # ========== USER LAYER ==========
    user_box = FancyBboxPatch((7.5, 10.2), 3, 0.8, boxstyle="round,pad=0.08", 
                              facecolor=color_user, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(user_box)
    ax.text(9, 10.6, '👤 User / Patient', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # ========== FRONTEND LAYER ==========
    # Frontend box
    frontend_box = FancyBboxPatch((6, 7.8), 6, 1.8, boxstyle="round,pad=0.12", 
                                  facecolor=color_frontend, edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(frontend_box)
    ax.text(9, 9.3, 'Frontend (React) - Port: 3000', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#000')
    
    # Frontend components (simplified)
    components = [
        ('Upload', 6.8, 8.5),
        ('Dashboard', 8, 8.5),
        ('Chat', 9.2, 8.5),
        ('History', 10.4, 8.5)
    ]
    for comp_name, x, y in components:
        comp_box = FancyBboxPatch((x-0.35, y-0.25), 0.7, 0.5, boxstyle="round,pad=0.04",
                                 facecolor='white', edgecolor='#333', linewidth=1)
        ax.add_patch(comp_box)
        ax.text(x, y, comp_name, ha='center', va='center', fontsize=7, fontweight='bold')
    
    # ========== BACKEND LAYER ==========
    # Backend box
    backend_box = FancyBboxPatch((4.5, 3.5), 9, 3.5, boxstyle="round,pad=0.12", 
                                 facecolor=color_backend, edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(backend_box)
    ax.text(9, 6.8, 'Backend (FastAPI) - Port: 8000', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    # API Endpoints
    endpoints = [
        ('🔐 Auth', 5.8, 6.0),
        ('📤 Prediction', 7.2, 6.0),
        ('💬 Chat', 8.6, 6.0),
        ('🔍 Health', 10.0, 6.0)
    ]
    for ep_name, x, y in endpoints:
        ep_box = FancyBboxPatch((x-0.45, y-0.3), 0.9, 0.6, boxstyle="round,pad=0.04",
                               facecolor='#004D40', edgecolor='white', linewidth=1.5)
        ax.add_patch(ep_box)
        ax.text(x, y, ep_name, ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white')
    
    # Services Layer
    ax.text(9, 5.3, 'Services', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    services = [
        ('Auth', 5.8, 4.7),
        ('Cloudinary', 7.2, 4.7),
        ('Prediction', 8.6, 4.7),
        ('Chatbot', 10.0, 4.7)
    ]
    for svc_name, x, y in services:
        svc_box = FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5, boxstyle="round,pad=0.03",
                                facecolor='#00695C', edgecolor='#E0F2F1', linewidth=1)
        ax.add_patch(svc_box)
        ax.text(x, y, svc_name, ha='center', va='center', fontsize=6, color='white')
    
    # ========== ML MODEL ==========
    ml_box = FancyBboxPatch((0.5, 4), 3, 2.2, boxstyle="round,pad=0.1", 
                           facecolor=color_ml, edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(ml_box)
    ax.text(2, 6, '🧠 ML Model', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    ax.text(2, 5.6, 'PyTorch Ensemble', ha='center', va='center', 
            fontsize=8, color='#E1BEE7')
    
    ml_components = [
        ('ResNet50', 1.2, 5.0),
        ('DenseNet', 2, 5.0),
        ('EfficientNet', 2.8, 5.0)
    ]
    for model, x, y in ml_components:
        model_box = FancyBboxPatch((x-0.3, y-0.18), 0.6, 0.36, boxstyle="round,pad=0.02",
                                  facecolor='#6A1B9A', edgecolor='white', linewidth=0.8)
        ax.add_patch(model_box)
        ax.text(x, y, model, ha='center', va='center', fontsize=5.5, color='white', fontweight='bold')
    
    ax.text(2, 4.5, 'RA Detection', ha='center', va='center', 
            fontsize=7, color='white', fontweight='bold')
    
    # ========== DATABASE ==========
    db_box = FancyBboxPatch((6, 0.8), 6, 2, boxstyle="round,pad=0.1", 
                           facecolor=color_database, edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(db_box)
    ax.text(9, 2.6, '🗄️ MongoDB Database', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    collections = [
        ('Users', 7.2, 1.8),
        ('Predictions', 8.4, 1.8),
        ('Chat', 9.6, 1.8),
        ('Sessions', 10.8, 1.8)
    ]
    for coll, x, y in collections:
        coll_box = FancyBboxPatch((x-0.35, y-0.2), 0.7, 0.4, boxstyle="round,pad=0.02",
                                 facecolor='#2E7D32', edgecolor='white', linewidth=1)
        ax.add_patch(coll_box)
        ax.text(x, y, coll, ha='center', va='center', fontsize=6, color='white', fontweight='bold')
    
    # ========== EXTERNAL SERVICES ==========
    # Cloudinary
    cloud_box = FancyBboxPatch((0.5, 1.8), 3, 1, boxstyle="round,pad=0.08", 
                              facecolor=color_external, edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(cloud_box)
    ax.text(2, 2.6, '☁️ Cloudinary', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(2, 2.2, 'Image Storage', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Gemini AI
    gemini_box = FancyBboxPatch((14, 4.5), 3.5, 1, boxstyle="round,pad=0.08", 
                               facecolor=color_external, edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(gemini_box)
    ax.text(15.75, 5.3, '🤖 Google Gemini', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(15.75, 4.9, 'AI Chatbot', ha='center', va='center', 
            fontsize=7, color='white')
    
    # ========== ARROWS / DATA FLOW ==========
    arrow_style = dict(arrowstyle='->', lw=2, color='#333')
    arrow_style_thick = dict(arrowstyle='->', lw=2.5, color='#1976D2')
    
    # User to Frontend
    ax.annotate('', xy=(9, 9.6), xytext=(9, 10.2), 
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#2196F3'))
    
    # Frontend to Backend (bidirectional)
    ax.annotate('', xy=(9, 7.8), xytext=(9, 7.0), 
                arrowprops=arrow_style_thick)
    ax.annotate('', xy=(9.2, 7.0), xytext=(9.2, 7.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#1976D2'))
    
    # Backend to ML Model
    ax.annotate('', xy=(3.5, 5.5), xytext=(4.5, 5.5), 
                arrowprops=arrow_style)
    ax.annotate('', xy=(4.5, 4.8), xytext=(3.5, 4.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#666', linestyle='dashed'))
    
    # Backend to Database
    ax.annotate('', xy=(9, 2.8), xytext=(9, 3.5), 
                arrowprops=arrow_style_thick)
    ax.annotate('', xy=(8.8, 3.5), xytext=(8.8, 2.8), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#1976D2'))
    
    # Backend to Cloudinary
    ax.annotate('', xy=(3, 2.5), xytext=(4.5, 4.2), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#FF9800'))
    
    # Backend to Gemini
    ax.annotate('', xy=(14, 5), xytext=(13.5, 5), 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#FF9800'))
    
    # ========== KEY FEATURES ==========
    features_box = FancyBboxPatch((14, 1.2), 3.5, 3, boxstyle="round,pad=0.08", 
                                 facecolor='#E8F5E9', edgecolor='black', linewidth=2, alpha=0.95)
    ax.add_patch(features_box)
    ax.text(15.75, 4, '✨ Key Features', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#000')
    
    features = [
        '✅ JWT Auth',
        '✅ X-ray Analysis',
        '✅ RA Detection',
        '✅ Classification',
        '✅ AI Chatbot',
        '✅ History Tracking'
    ]
    
    y_start = 3.7
    for i, feature in enumerate(features):
        ax.text(14.2, y_start - (i * 0.35), feature, ha='left', va='center', 
                fontsize=7, color='#000')
    
    # ========== TECH STACK ==========
    tech_box = FancyBboxPatch((0.5, 0.2), 3.5, 1.3, boxstyle="round,pad=0.08", 
                             facecolor='#E3F2FD', edgecolor='black', linewidth=2, alpha=0.95)
    ax.add_patch(tech_box)
    ax.text(2.25, 1.3, '💻 Tech Stack', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#000')
    
    tech = [
        '• React • FastAPI',
        '• PyTorch • MongoDB',
        '• Cloudinary • Gemini'
    ]
    
    y_tech = 1.05
    for i, t in enumerate(tech):
        ax.text(0.7, y_tech - (i * 0.25), t, ha='left', va='center', 
                fontsize=6, color='#000')
    
    # ========== FOOTER ==========
    ax.text(9, 0.1, 'RAiCare - Medical AI System for Rheumatoid Arthritis Detection', 
            ha='center', va='center', fontsize=8, color='#666', style='italic')
    
    # Save the diagram
    plt.tight_layout()
    output_path = 'e:/RA-Project/ra-detection/RAiCare_System_Architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"System Architecture Diagram saved successfully!")
    print(f"Location: {output_path}")
    print(f"Resolution: 300 DPI (High Quality)")
    
    plt.show()

if __name__ == "__main__":
    print("Generating RAiCare System Architecture Diagram...")
    create_architecture_diagram()
