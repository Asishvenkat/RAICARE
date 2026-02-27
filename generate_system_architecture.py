"""
Generate System Architecture Diagram for RAiCare Project
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

def create_system_architecture():
    """Create and display the RAiCare system architecture diagram"""
    
    # Vertical layout with white background
    fig, ax = plt.subplots(1, 1, figsize=(8, 14))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Black and white color scheme
    box_color = 'white'
    edge_color = 'black'
    text_color = 'black'
    
    # Title
    ax.text(4, 13.5, 'RAiCare System Architecture', 
            fontsize=18, fontweight='bold', ha='center', color=text_color)
    
    # ===== USER INPUT =====
    input_box = Rectangle((1, 12.2), 6, 0.8, 
                          edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(input_box)
    ax.text(4, 12.6, 'User Input (X-ray Image)', fontsize=11, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # ===== FRONTEND LAYER =====
    # Container box for frontend
    frontend_container = Rectangle((0.5, 10), 7, 1.8, 
                                   edgecolor=edge_color, facecolor='none', 
                                   linewidth=2, linestyle='--')
    ax.add_patch(frontend_container)
    ax.text(0.7, 11.5, 'Frontend', fontsize=9, ha='left', va='center', 
            fontweight='bold', color=text_color, style='italic')
    
    # React Component
    react_box = Rectangle((1.5, 10.8), 2, 0.7, 
                         edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(react_box)
    ax.text(2.5, 11.15, 'React UI', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # Tailwind CSS
    tailwind_box = Rectangle((4.5, 10.8), 2, 0.7,
                            edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(tailwind_box)
    ax.text(5.5, 11.15, 'Tailwind CSS', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # Axios Client
    axios_box = Rectangle((1.5, 10), 2, 0.5,
                         edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(axios_box)
    ax.text(2.5, 10.25, 'Axios Client', fontsize=9, ha='center', va='center', color=text_color)
    
    # Router
    router_box = Rectangle((4.5, 10), 2, 0.5,
                          edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(router_box)
    ax.text(5.5, 10.25, 'React Router', fontsize=9, ha='center', va='center', color=text_color)
    
    # ===== BACKEND LAYER =====
    # Container box for backend
    backend_container = Rectangle((0.5, 6.5), 7, 3, 
                                  edgecolor=edge_color, facecolor='none', 
                                  linewidth=2, linestyle='--')
    ax.add_patch(backend_container)
    ax.text(0.7, 9.2, 'Backend', fontsize=9, ha='left', va='center', 
            fontweight='bold', color=text_color, style='italic')
    
    # FastAPI Server
    fastapi_box = Rectangle((2.5, 8.5), 3, 0.8,
                           edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(fastapi_box)
    ax.text(4, 8.9, 'FastAPI Server\n(Uvicorn)', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # Auth Service
    auth_box = Rectangle((1, 7.4), 2, 0.6,
                        edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(auth_box)
    ax.text(2, 7.7, 'Auth (JWT)', fontsize=9, ha='center', va='center', color=text_color)
    
    # Prediction Service
    pred_service_box = Rectangle((3.5, 7.4), 2, 0.6,
                                edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(pred_service_box)
    ax.text(4.5, 7.7, 'Prediction\nService', fontsize=9, ha='center', va='center', color=text_color)
    
    # Chat Service
    chat_service_box = Rectangle((6, 7.4), 1.5, 0.6,
                                edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(chat_service_box)
    ax.text(6.75, 7.7, 'Chat\nService', fontsize=9, ha='center', va='center', color=text_color)
    
    # Cloudinary Service
    cloudinary_box = Rectangle((1, 6.6), 2, 0.5,
                              edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(cloudinary_box)
    ax.text(2, 6.85, 'Cloudinary', fontsize=9, ha='center', va='center', color=text_color)
    
    # ===== ML MODEL LAYER =====
    # Container box for ML
    ml_container = Rectangle((0.5, 3.8), 7, 2.2,
                            edgecolor=edge_color, facecolor='none', 
                            linewidth=2, linestyle='--')
    ax.add_patch(ml_container)
    ax.text(0.7, 5.7, 'ML Pipeline', fontsize=9, ha='left', va='center', 
            fontweight='bold', color=text_color, style='italic')
    
    # Image Preprocessing
    preprocess_box = Rectangle((1.5, 5.2), 2.2, 0.6,
                              edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(preprocess_box)
    ax.text(2.6, 5.5, 'Image\nPreprocessing', fontsize=9, ha='center', va='center', color=text_color)
    
    # PyTorch
    pytorch_box = Rectangle((4.3, 5.2), 2.2, 0.6,
                           edgecolor=edge_color, facecolor=box_color, linewidth=1.5)
    ax.add_patch(pytorch_box)
    ax.text(5.4, 5.5, 'PyTorch\nEngine', fontsize=9, ha='center', va='center', color=text_color)
    
    # ResNet50 Model
    resnet_box = Rectangle((2, 4.2), 4, 0.7,
                          edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(resnet_box)
    ax.text(4, 4.55, 'ResNet50 Model\n(Ensemble)', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # ===== DATABASE =====
    db_box = Rectangle((2.5, 2.8), 3, 0.7,
                      edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(db_box)
    ax.text(4, 3.15, 'MongoDB\n(Beanie ODM)', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # ===== EXTERNAL SERVICES =====
    # Gemini AI
    gemini_box = Rectangle((5.8, 2.8), 1.7, 0.7,
                          edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(gemini_box)
    ax.text(6.65, 3.15, 'Google\nGemini AI', fontsize=9, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # ===== PREDICTION OUTPUT =====
    output_box = Rectangle((2, 1.5), 4, 0.7,
                          edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(output_box)
    ax.text(4, 1.85, 'Prediction Results\n& Recommendations', fontsize=10, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # ===== USER OUTPUT =====
    final_output = Rectangle((1, 0.5), 6, 0.7,
                            edgecolor=edge_color, facecolor=box_color, linewidth=2)
    ax.add_patch(final_output)
    ax.text(4, 0.85, 'User Dashboard (Results Display)', fontsize=11, ha='center', va='center', 
            fontweight='bold', color=text_color)
    
    # ===== ARROWS (Data Flow) =====
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to Frontend
    ax.annotate('', xy=(4, 11.8), xytext=(4, 12.2),
                arrowprops=arrow_props)
    
    # Frontend to Backend
    ax.annotate('', xy=(4, 9.3), xytext=(4, 10),
                arrowprops=arrow_props)
    
    # Backend to Auth
    ax.annotate('', xy=(2, 8), xytext=(3.5, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Backend to Prediction Service
    ax.annotate('', xy=(4.5, 8), xytext=(4, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Backend to Chat Service
    ax.annotate('', xy=(6.75, 8), xytext=(4.5, 8.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Prediction Service to ML Pipeline
    ax.annotate('', xy=(4, 6), xytext=(4.5, 7.4),
                arrowprops=arrow_props)
    
    # Preprocessing to Model
    ax.annotate('', xy=(4, 4.9), xytext=(3.7, 5.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Model to Database
    ax.annotate('', xy=(4, 3.5), xytext=(4, 4.2),
                arrowprops=arrow_props)
    
    # Chat Service to Gemini
    ax.annotate('', xy=(6.65, 3.5), xytext=(6.75, 7.4),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    
    # Database to Output
    ax.annotate('', xy=(4, 2.2), xytext=(4, 2.8),
                arrowprops=arrow_props)
    
    # Output to User Dashboard
    ax.annotate('', xy=(4, 1.2), xytext=(4, 1.5),
                arrowprops=arrow_props)
    
    # Cloudinary integration
    ax.annotate('', xy=(2, 6.6), xytext=(2, 7.4),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'models/system_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ System architecture diagram saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    create_system_architecture()
