import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data preparation
data = {
    'ResNet-101': {
        'cardboard3.jpg': [
            ('cardboard32', 'cardboard', 0.9257), ('cardboard329', 'cardboard', 0.9113),
            ('cardboard62', 'cardboard', 0.9045), ('cardboard45', 'cardboard', 0.9023),
            ('cardboard139', 'cardboard', 0.9008), ('cardboard241', 'cardboard', 0.9006),
            ('cardboard192', 'cardboard', 0.9004), ('cardboard71', 'cardboard', 0.8999),
            ('cardboard319', 'cardboard', 0.8998), ('cardboard372', 'cardboard', 0.8951)
        ],
        'glass23.jpg': [
            ('glass412', 'glass', 0.9247), ('glass308', 'glass', 0.8756),
            ('glass318', 'glass', 0.8634), ('plastic292', 'plastic', 0.8568),
            ('plastic371', 'plastic', 0.8561), ('plastic297', 'plastic', 0.8561),
            ('plastic277', 'plastic', 0.8557), ('glass171', 'glass', 0.8554),
            ('glass352', 'glass', 0.8527), ('plastic32', 'plastic', 0.8526)
        ],
        'metal38.jpg': [
            ('glass409', 'glass', 0.8665), ('metal188', 'metal', 0.8549),
            ('metal174', 'metal', 0.8529), ('glass190', 'glass', 0.8499),
            ('glass448', 'glass', 0.8393), ('metal269', 'metal', 0.8371),
            ('metal330', 'metal', 0.8363), ('metal45', 'metal', 0.8322),
            ('glass163', 'glass', 0.8321), ('metal49', 'metal', 0.8286)
        ],
        'paper29.jpg': [
            ('paper465', 'paper', 0.9269), ('paper425', 'paper', 0.8480),
            ('paper448', 'paper', 0.8418), ('paper426', 'paper', 0.8320),
            ('paper77', 'paper', 0.8305), ('paper129', 'paper', 0.8296),
            ('paper543', 'paper', 0.8240), ('cardboard385', 'cardboard', 0.8137),
            ('paper257', 'paper', 0.8091), ('paper569', 'paper', 0.7996)
        ],
        'plastic21.jpg': [
            ('plastic208', 'plastic', 0.9533), ('plastic106', 'plastic', 0.9291),
            ('plastic395', 'plastic', 0.9192), ('plastic298', 'plastic', 0.9172),
            ('plastic379', 'plastic', 0.9098), ('plastic386', 'plastic', 0.9085),
            ('plastic353', 'plastic', 0.9078), ('plastic64', 'plastic', 0.9054),
            ('plastic202', 'plastic', 0.8965), ('plastic331', 'plastic', 0.8948)
        ],
        'trash40.jpg': [
            ('glass86', 'glass', 0.8328), ('plastic392', 'plastic', 0.8214),
            ('plastic315', 'plastic', 0.8088), ('metal125', 'metal', 0.7926),
            ('metal101', 'metal', 0.7925), ('trash41', 'trash', 0.7898),
            ('plastic443', 'plastic', 0.7898), ('plastic213', 'plastic', 0.7882),
            ('plastic385', 'plastic', 0.7874), ('plastic129', 'plastic', 0.7824)
        ]
    },
    'ZFNet': {
        'cardboard3.jpg': [
            ('glass352', 'glass', 0.9982), ('cardboard48', 'cardboard', 0.9982),
            ('glass488', 'glass', 0.9982), ('glass470', 'glass', 0.9982),
            ('glass383', 'glass', 0.9981), ('glass282', 'glass', 0.9981),
            ('glass160', 'glass', 0.9981), ('glass202', 'glass', 0.9980),
            ('glass31', 'glass', 0.9980), ('paper177', 'paper', 0.9979)
        ],
        'glass23.jpg': [
            ('glass108', 'glass', 0.9988), ('glass86', 'glass', 0.9988),
            ('paper225', 'paper', 0.9988), ('trash28', 'trash', 0.9987),
            ('metal318', 'metal', 0.9987), ('glass485', 'glass', 0.9986),
            ('glass308', 'glass', 0.9986), ('glass65', 'glass', 0.9986),
            ('glass321', 'glass', 0.9986), ('trash118', 'trash', 0.9985)
        ],
        'metal38.jpg': [
            ('metal1', 'metal', 0.9984), ('paper304', 'paper', 0.9984),
            ('metal45', 'metal', 0.9982), ('cardboard201', 'cardboard', 0.9982),
            ('metal267', 'metal', 0.9981), ('metal314', 'metal', 0.9981),
            ('glass294', 'glass', 0.9981), ('paper536', 'paper', 0.9980),
            ('glass460', 'glass', 0.9980), ('glass360', 'glass', 0.9979)
        ],
        'paper29.jpg': [
            ('paper446', 'paper', 0.9981), ('metal398', 'metal', 0.9979),
            ('glass86', 'glass', 0.9977), ('paper197', 'paper', 0.9975),
            ('trash108', 'trash', 0.9975), ('metal41', 'metal', 0.9974),
            ('paper122', 'paper', 0.9974), ('glass443', 'glass', 0.9974),
            ('paper205', 'paper', 0.9974), ('metal135', 'metal', 0.9974)
        ],
        'plastic21.jpg': [
            ('plastic208', 'plastic', 0.9996), ('glass408', 'glass', 0.9992),
            ('plastic358', 'plastic', 0.9991), ('plastic202', 'plastic', 0.9991),
            ('metal18', 'metal', 0.9990), ('plastic409', 'plastic', 0.9990),
            ('plastic296', 'plastic', 0.9990), ('plastic106', 'plastic', 0.9990),
            ('glass166', 'glass', 0.9990), ('plastic38', 'plastic', 0.9990)
        ],
        'trash40.jpg': [
            ('metal39', 'metal', 0.9983), ('trash34', 'trash', 0.9982),
            ('trash10', 'trash', 0.9981), ('metal29', 'metal', 0.9980),
            ('metal92', 'metal', 0.9980), ('trash60', 'trash', 0.9980),
            ('trash41', 'trash', 0.9978), ('trash87', 'trash', 0.9978),
            ('cardboard192', 'cardboard', 0.9978), ('metal318', 'metal', 0.9978)
        ]
    },
    'GoogleNet': {
        'cardboard3.jpg': [
            ('cardboard269', 'cardboard', 0.8856), ('cardboard192', 'cardboard', 0.8835),
            ('cardboard71', 'cardboard', 0.8768), ('cardboard339', 'cardboard', 0.8681),
            ('cardboard252', 'cardboard', 0.8668), ('cardboard32', 'cardboard', 0.8614),
            ('cardboard45', 'cardboard', 0.8610), ('cardboard241', 'cardboard', 0.8581),
            ('cardboard393', 'cardboard', 0.8579), ('cardboard36', 'cardboard', 0.8578)
        ],
        'glass23.jpg': [
            ('glass232', 'glass', 0.8611), ('glass318', 'glass', 0.8500),
            ('glass135', 'glass', 0.8476), ('glass123', 'glass', 0.8440),
            ('glass412', 'glass', 0.8433), ('glass35', 'glass', 0.8356),
            ('glass34', 'glass', 0.8301), ('plastic411', 'plastic', 0.8295),
            ('plastic32', 'plastic', 0.8266), ('glass375', 'glass', 0.8256)
        ],
        'metal38.jpg': [
            ('metal64', 'metal', 0.8539), ('metal188', 'metal', 0.8430),
            ('metal194', 'metal', 0.8355), ('metal174', 'metal', 0.8345),
            ('glass409', 'glass', 0.8262), ('glass141', 'glass', 0.8232),
            ('glass117', 'glass', 0.8207), ('glass163', 'glass', 0.8178),
            ('metal330', 'metal', 0.8160), ('plastic469', 'plastic', 0.8126)
        ],
        'paper29.jpg': [
            ('paper465', 'paper', 0.8883), ('paper257', 'paper', 0.8326),
            ('paper259', 'paper', 0.8140), ('paper129', 'paper', 0.7928),
            ('paper448', 'paper', 0.7918), ('cardboard385', 'cardboard', 0.7798),
            ('paper425', 'paper', 0.7796), ('paper7', 'paper', 0.7795),
            ('paper311', 'paper', 0.7679), ('paper584', 'paper', 0.7637)
        ],
        'plastic21.jpg': [
            ('plastic208', 'plastic', 0.9160), ('plastic393', 'plastic', 0.8630),
            ('glass75', 'glass', 0.8519), ('plastic64', 'plastic', 0.8494),
            ('glass376', 'glass', 0.8484), ('glass255', 'glass', 0.8471),
            ('plastic106', 'plastic', 0.8436), ('glass446', 'glass', 0.8430),
            ('plastic298', 'plastic', 0.8428), ('plastic202', 'plastic', 0.8419)
        ],
        'trash40.jpg': [
            ('trash41', 'trash', 0.7422), ('trash57', 'trash', 0.7253),
            ('trash122', 'trash', 0.7236), ('trash53', 'trash', 0.7179),
            ('glass466', 'glass', 0.7073), ('trash108', 'trash', 0.7068),
            ('metal245', 'metal', 0.7059), ('glass102', 'glass', 0.7042),
            ('paper589', 'paper', 0.7033), ('glass259', 'glass', 0.6967)
        ]
    }
}

# Define class labels and colors
class_colors = {
    'cardboard': '#8B4513',  # Brown
    'glass': '#00CED1',      # Cyan
    'metal': '#C0C0C0',      # Silver
    'paper': '#F5F5DC',      # Beige
    'plastic': '#FF4500',    # Orange
    'trash': '#696969'       # Gray
}

# Plotting function
def plot_nearest_neighbors(model, image, neighbors, ax):
    # Extract data
    labels = [f"{name} ({cls})" for name, cls, _ in neighbors]
    similarities = [sim for _, _, sim in neighbors]
    classes = [cls for _, cls, _ in neighbors]
    
    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Neighbor': labels,
        'Similarity': similarities,
        'Class': classes
    })
    
    # Plot
    sns.barplot(
        x='Similarity', y='Neighbor', hue='Class', palette=class_colors,
        data=df, ax=ax, dodge=False
    )
    
    # Customize plot
    ax.set_title(f'{model}: Nearest Neighbors for {image}', fontsize=12, pad=10)
    ax.set_xlabel('Similarity Score', fontsize=10)
    ax.set_ylabel('Neighbor (Class)', fontsize=10)
    ax.set_xlim(0, 1)  # Adjust based on model similarity range
    ax.tick_params(axis='both', labelsize=8)
    
    # Adjust legend
    ax.legend(title='Class', fontsize=8, title_fontsize=9, loc='lower right')
    
    # Add grid for readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Create subplots for each class
classes = ['cardboard3.jpg', 'glass23.jpg', 'metal38.jpg', 'paper29.jpg', 'plastic21.jpg', 'trash40.jpg']
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

for class_idx, (image, class_name) in enumerate(zip(classes, class_names)):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    
    for model_idx, model in enumerate(['ResNet-101', 'ZFNet', 'GoogleNet']):
        plot_nearest_neighbors(model, image, data[model][image], axes[model_idx])
        
        # Adjust x-axis for ZFNet due to high similarity scores
        if model == 'ZFNet':
            axes[model_idx].set_xlim(0.997, 1)
    
    plt.suptitle(f'Top 10 Nearest Neighbors for {class_name} Class', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()