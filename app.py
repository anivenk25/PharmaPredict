import streamlit as st
import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Draw
import pubchempy as pcp
import matplotlib.pyplot as plt

# Define a function to compute the Tanimoto similarity between two molecules
def compute_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0.0  # Handle invalid input

    fp1 = FingerprintMols.FingerprintMol(mol1)
    fp2 = FingerprintMols.FingerprintMol(mol2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Define a function to get SMILES notation for a drug name
def get_smiles_for_drug(drug_name):
    try:
        # Search PubChem for the drug by name
        compounds = pcp.get_compounds(drug_name, 'name')

        if compounds:
            # Take the first result (you may want to handle multiple results differently)
            compound = compounds[0]

            # Check if SMILES notation is available
            if compound.isomeric_smiles:
                return compound.isomeric_smiles, compound.molecular_formula
            else:
                return "SMILES notation not found for this drug.", None
        else:
            return "Drug not found in PubChem database.", None

    except Exception as e:
        return str(e), None

# Function to predict DDI for a pair of drugs
def predict_ddi_for_drugs(drug1_name, drug2_name, model_path):
    # Get SMILES notations and molecular formulas for the drugs
    drug1_smiles, drug1_formula = get_smiles_for_drug(drug1_name)
    drug2_smiles, drug2_formula = get_smiles_for_drug(drug2_name)

    if "SMILES notation not found" in (drug1_smiles, drug2_smiles):
        return "SMILES notation not found for one or both drugs.", None, None

    # Compute the structural similarity score between the drug pair
    similarity_score = compute_similarity(drug1_smiles, drug2_smiles)

    # Create an SSP feature vector (100 components as per your description)
    ssp_feature_vector = np.array([similarity_score] * 100)

    # Load the saved model
    loaded_model = keras.models.load_model(model_path)

    # Prepare input data for prediction (reshape the feature vector)
    input_data = ssp_feature_vector.reshape(1, -1)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(input_data)

    # Find the class label with the highest probability
    predicted_class = np.argmax(predictions)

    # Dictionary mapping predicted class labels to descriptions
    class_description_mapping = {
        1: "Drug1 can cause an increase in the absorption of Drug2 resulting in an increased serum concentration and potentially a worsening of adverse effects.",
        2: "Drug1 can cause a decrease in the absorption of Drug2 resulting in a reduced serum concentration and potentially a decrease in efficacy.",
        3: "The risk or severity of adverse effects can be increased when Drug1 is combined with Drug2.",
        4: "The bioavailability of Drug2 can be decreased when combined with Drug1.",
        5: "Drug1 may decrease the vasoconstricting activities of Drug2.",
        6: "Drug1 may increase the anticoagulant activities of Drug2.",
        7: "Drug1 may increase the ototoxic activities of Drug2.",
        8: "The therapeutic efficacy of Drug2 can be increased when used in combination with Drug1.",
        9: "The serum concentration of Drug2 can be decreased when it is combined with Drug1.",
        10: "The risk or severity of hypertension can be increased when Drug2 is combined with Drug1.",
        11: "The serum concentration of the active metabolites of Drug2 can be reduced when Drug2 is used in combination with Drug1 resulting in a loss in efficacy.",
        12: "Drug1 may decrease the anticoagulant activities of Drug2.",
        13: "The absorption of Drug2 can be decreased when combined with Drug1.",
        14: "Drug1 may decrease the bronchodilatory activities of Drug2.",
        15: "Drug1 may decrease the bioavailability of Drug2.",
        16: "Drug1 may increase the central neurotoxic activities of Drug2.",
        17: "Drug1 may increase the cardiotoxic activities of Drug2.",
        18: "Drug1 may increase the cardiotoxic activities of Drug2.",
        19: "Drug1 may increase the vasoconstricting activities of Drug2.",
        20: "Drug1 may increase the QTc-prolonging activities of Drug2.",
        21: "Drug1 may increase the neuromuscular blocking activities of Drug2.",
        22: "Drug1 may increase the adverse neuromuscular activities of Drug2.",
        23: "Drug1 may increase the stimulatory activities of Drug2.",
        24: "Drug1 may decrease effectiveness of Drug2 as a diagnostic agent.",
        25: "Drug1 may increase the atrioventricular blocking (AV block) activities of Drug2.",
        26: "Drug1 may decrease the antiplatelet activities of Drug2.",
        27: "Drug1 may increase the neuroexcitatory activities of Drug2.",
        28: "Drug1 may increase the dermatologic adverse activities of Drug2.",
        29: "Drug1 may decrease the diuretic activities of Drug2.",
        30: "Drug1 may increase the orthostatic hypotensive activities of Drug2.",
        31: "The risk or severity of hypertension can be increased when Drug1 is combined with Drug2.",
        32: "The risk or severity of QTc prolongation can be increased when Drug1 is combined with Drug2.",
        33: "Drug1 may decrease the analgesic activities of Drug2.",
        34: "Drug1 may decrease the anticoagulant activities of Drug2.",
        35: "Drug1 may decrease the antihypertensive activities of Drug2.",
        36: "Drug1 may decrease the antiplatelet activities of Drug2.",
        37: "Drug1 may decrease the antihypertensive activities of Drug2.",
        38: "Drug1 may increase the vasodilatory activities of Drug2.",
        39: "Drug1 may increase the constipating activities of Drug2.",
        40: "Drug1 may increase the respiratory depressant activities of Drug2.",
        41: "Drug1 may increase the hypotensive and central nervous system depressant (CNS depressant) activities of Drug2.",
        42: "The risk or severity of hyperkalemia can be increased when Drug1 is combined with Drug2.",
        43: "The protein binding of Drug2 can be decreased when combined with Drug1.",
        44: "Drug1 may increase the central nervous system depressant (CNS depressant) and hypertensive activities of Drug2.",
        45: "Drug1 may decrease the effectiveness of Drug2 as a diagnostic agent.",
        46: "Drug1 may increase the bronchoconstrictory activities of Drug2.",
        47: "The metabolism of Drug2 can be decreased when combined with Drug1.",
        48: "Drug1 may increase the myopathic rhabdomyolysis activities of Drug2.",
        49: "The risk or severity of adverse effects can be increased when Drug1 is combined with Drug2.",
        50: "The risk or severity of heart failure can be increased when Drug2 is combined with Drug1.",
        51: "Drug1 may increase the hypercalcemic activities of Drug2.",
        52: "Drug1 may decrease the analgesic activities of Drug2.",
        53: "Drug1 may decrease the effectiveness of Drug2 as a diagnostic agent.",
        54: "Drug1 may decrease the effectiveness of Drug2 as a diagnostic agent.",
        55: "Drug1 may decrease the effectiveness of Drug2 as a diagnostic agent.",
        56: "The risk or severity of hypotension can be increased when Drug1 is combined with Drug2.",
        57: "The risk or severity of hypotension can be increased when Drug1 is combined with Drug2.",
        58: "Drug1 may decrease the cardiotoxic activities of Drug2.",
        59: "Drug1 may increase the ulcerogenic activities of Drug2.",
        60: "Drug1 may increase the hyponatremic activities of Drug2.",
        61: "Drug1 may decrease the sedative activities of Drug2.",
        62: "Drug1 may decrease the excretion rate of Drug2 which could result in a higher serum level.",
        63: "Drug1 may increase the myelosuppressive activities of Drug2.",
        64: "Drug1 may increase the hypoglycemic activities of Drug2.",
        65: "Drug1 may increase the excretion rate of Drug2 which could result in a lower serum level and potentially a reduction in efficacy.",
        66: "The risk or severity of bleeding can be increased when Drug1 is combined with Drug2.",
        67: "The risk or severity of hypotension can be increased when Drug1 is combined with Drug2.",
        68: "Drug1 may increase the analgesic activities of Drug2.",
        69: "Drug1 may increase the analgesic activities of Drug2.",
        70: "The therapeutic efficacy of Drug2 can be decreased when used in combination with Drug1.",
        71: "Drug1 may increase the hypertensive activities of Drug2.",
        72: "Drug1 may decrease the excretion rate of Drug2 which could result in a higher serum level.",
        73: "The serum concentration of Drug2 can be increased when it is combined with Drug1.",
        74: "Drug1 may increase the fluid retaining activities of Drug2.",
        75: "The serum concentration of Drug2 can be decreased when it is combined with Drug1.",
        76: "Drug1 may decrease the sedative activities of Drug2.",
        77: "The serum concentration of the active metabolites of Drug2 can be increased when Drug2 is used in combination with Drug1.",
        78: "Drug1 may increase the hyperglycemic activities of Drug2.",
        79: "Drug1 may increase the central nervous system depressant (CNS depressant) and hypertensive activities of Drug2.",
        80: "Drug1 may increase the hepatotoxic activities of Drug2.",
        81: "Drug1 may increase the thrombogenic activities of Drug2.",
        82: "Drug1 may increase the arrhythmogenic activities of Drug2.",
        83: "Drug1 may increase the vasopressor activities of Drug2.",
        84: "Drug1 may increase the vasodilatory activities of Drug2.",
        85: "Drug1 may increase the tachycardic activities of Drug2.",
        86: "The risk of a hypersensitivity reaction to Drug2 is increased when it is combined with Drug1."
}

    # Check if the predicted class label exists in the mapping dictionary
    if predicted_class in class_description_mapping:
        description = class_description_mapping[predicted_class]
    else:
        description = "Predicted Class Label not found in the mapping."

    # Generate molecular diagrams (if molecular formulas are available)
    drug1_molecule = None
    drug2_molecule = None
    if drug1_smiles:
        drug1_molecule = Chem.MolFromSmiles(drug1_smiles)
    if drug2_smiles:
        drug2_molecule = Chem.MolFromSmiles(drug2_smiles)

    # Save molecular diagrams as images
    drug1_image_path = "drug1_molecule.png"
    drug2_image_path = "drug2_molecule.png"
    if drug1_molecule:
        Draw.MolToImage(drug1_molecule, size=(300, 300)).save(drug1_image_path)
    if drug2_molecule:
        Draw.MolToImage(drug2_molecule, size=(300, 300)).save(drug2_image_path)

    return f"Drug 1: {drug1_name} (SMILES: {drug1_smiles}, Molecular Formula: {drug1_formula})\n" \
           f"Drug 2: {drug2_name} (SMILES: {drug2_smiles}, Molecular Formula: {drug2_formula})\n" \
           f"Predicted Class Label: {predicted_class}\n" \
           f"Description: {description}", drug1_image_path, drug2_image_path

# Create a Streamlit web app
st.set_page_config(
    page_title="Drug-Drug Interaction Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set the app title and description
st.title("Drug-Drug Interaction Prediction")
st.write(
    "This app predicts drug-drug interactions (DDIs) and provides information about the drugs involved, "
    "their SMILES notation, and a brief description of the interaction."
)

# Create input fields for drug names
st.sidebar.header("Input Drugs")
drug1_name = st.sidebar.text_input("Enter Drug 1 Name", "Aspirin")
drug2_name = st.sidebar.text_input("Enter Drug 2 Name", "Ibuprofen")

# Create a button to trigger prediction
if st.sidebar.button("Predict DDI"):
    # Path to the saved model
    model_path = "ddi_model.h5"

    # Make predictions for Drug-Drug Interaction
    result, drug1_image_path, drug2_image_path = predict_ddi_for_drugs(drug1_name, drug2_name, model_path)

    # Output the result
    st.write(result)

    # Display molecular diagrams (if available)
    if drug1_image_path:
        drug1_image = plt.imread(drug1_image_path)
        st.image(drug1_image, caption=f"Molecular Diagram of {drug1_name}", use_column_width=True)

    if drug2_image_path:
        drug2_image = plt.imread(drug2_image_path)
        st.image(drug2_image, caption=f"Molecular Diagram of {drug2_name}", use_column_width=True)
