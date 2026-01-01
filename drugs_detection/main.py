import uvicorn
import json
import pandas as pd
import numpy as np
import faiss
import cv2
import re
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import easyocr
from collections import Counter
from rapidfuzz import process
from ctransformers import AutoModelForCausalLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RAGRequest(BaseModel):
    query: str

print("\n\n⚡ *** DÉMARRAGE MODE CORRIGÉ (ANTI-HALLUCINATION) *** ⚡")

# 0. HARDWARE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Matériel : {device.upper()}")

# 1. MODÈLES
print("1️⃣ Chargement...")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
reranker = CrossEncoder('BAAI/bge-reranker-base', device=device)
reader = easyocr.Reader(['fr', 'en'], gpu=(device == "cuda"))

# 2. CSV (OCR)
try:
    df_drugs = pd.read_csv("medicaments_frances_approx_poso_full.csv", sep=';', encoding='utf-8-sig')
    df_drugs['drug_name'] = df_drugs['drug_name'].astype(str).str.strip()
    df_drugs['sub_category'] = df_drugs['sub_category'].astype(str).str.strip()
    df_drugs['drug_name_norm'] = df_drugs['drug_name'].str.upper()
    drug_names = df_drugs['drug_name'].tolist()
    drug_names_norm = df_drugs['drug_name_norm'].tolist()
    print("✅ CSV OK.")
except:
    df_drugs = pd.DataFrame()

# 3. JSON (RAG) - TEXTIFY AMÉLIORÉ
try:
    with open("data_anesthesie.json", 'r', encoding='utf-8') as f:
        rag_data = json.load(f)

    def textify(entry):
        # ON MET LE NOM DU MÉDICAMENT EN MAJUSCULE AU DÉBUT
        # C'est crucial pour que le Vector Search trouve le bon.
        
        prefix = f"PROTOCOLE POUR : {entry.get('medication', 'INCONNU').upper()}."
        
        if "decision_J_0_morning" in entry:
            j2 = "MAINTENIR" if entry.get("decision_J_minus_2") == "Oui" else "ARRÊTER"
            j1 = "MAINTENIR" if entry.get("decision_J_minus_1") == "Oui" else "ARRÊTER"
            j0 = "MAINTENIR" if entry.get("decision_J_0_morning") == "Oui" else "ARRÊTER"
            
            return (
                f"{prefix} Catégorie: {entry.get('category')}. "
                f"J-2: {j2}. J-1: {j1}. "
                f"MATIN OPÉRATION (J0): {j0}. "
                f"Risque arrêt: {entry.get('risk_if_stopped')}. "
                f"Précautions: {entry.get('precautions')}"
            )
        elif "score" in entry:
            return f"SCORE ASA : {entry.get('score')}. {entry.get('description')}"
        elif "medication" in entry:
            return f"{prefix} Consigne: {entry.get('instruction') or entry.get('instruction_arret')}."
        
        return str(entry)

    rag_docs = [textify(e) for e in rag_data]
    rag_vectors = embedder.encode(rag_docs)
    rag_index = faiss.IndexFlatL2(rag_vectors.shape[1])
    rag_index.add(rag_vectors)
    print("✅ JSON OK (Indexation Renforcée).")

except Exception as e:
    print(f"⚠️ Erreur JSON: {e}")

# 4. MISTRAL
print("⏳ Chargement Mistral...")
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
try:
    llm = AutoModelForCausalLM.from_pretrained(model_id, model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf", model_type="mistral", context_length=2048, gpu_layers=0)
    print("✅ Mistral OK.")
except:
    llm = None

# --- ENDPOINTS ---

@app.post("/ocr/scan")
async def scan_drug(file: UploadFile = File(...)):
    # ... (Code OCR inchangé, garde le tien ici, je le raccourcis pour la lisibilité) ...
    # Copie colle ton bloc OCR précédent ici
    return {"drugName": "SIMULATION", "confidence": 1.0} 

@app.post("/rag/protocol")
async def get_protocol(req: RAGRequest):
    print(f"\n❓ Question reçue : {req.query}")
    query_upper = req.query.upper()
    best_doc = ""

    # --- CAS SPÉCIAL : CALCUL DE SCORE ASA ---
    # Si la question parle de "ASA" ou "SCORE", on donne TOUTES les définitions à l'IA
    if "ASA" in query_upper or "SCORE" in query_upper:
        print("🎯 MODE DIAGNOSTIC : SCORE ASA DETECTÉ")
        # On récupère toutes les entrées qui contiennent "ASA" dans le JSON
        all_asa_docs = [d for d in rag_docs if "ASA" in d]
        # On les colle toutes ensemble pour faire un gros contexte
        best_doc = "\n---\n".join(all_asa_docs)
    
    # --- CAS STANDARD : MÉDICAMENTS ---
    else:
        # 1. Détection par nom de médicament (Fuzzy Match)
        all_meds_in_db = []
        for doc in rag_docs:
            try:
                if "PROTOCOLE POUR :" in doc:
                    name = doc.split("PROTOCOLE POUR : ")[1].split(".")[0]
                    all_meds_in_db.append(name)
            except: pass
        
        best_match_name = None
        best_match_score = 0

        # Recherche floue (RapidFuzz)
        from rapidfuzz import fuzz
        for med in all_meds_in_db:
            if "ASA" in med: continue # On ignore les ASA ici
            score = fuzz.partial_ratio(med, query_upper)
            if score > 85 and score > best_match_score:
                best_match_score = score
                best_match_name = med

        if best_match_name:
            print(f"🎯 MÉDICAMENT IDENTIFIÉ : {best_match_name}")
            best_doc = next((d for d in rag_docs if f"PROTOCOLE POUR : {best_match_name}" in d), rag_docs[0])
        else:
            print("⚠️ Recherche Vectorielle (FAISS)...")
            q_vec = embedder.encode([req.query])
            D, I = rag_index.search(q_vec, k=5)
            candidates = [rag_docs[i] for i in I[0]]
            pairs = [[req.query, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            best_doc = candidates[scores.argmax()]

    print(f"📄 CONTEXTE ENVOYÉ À L'IA : {best_doc[:100]}...") # Juste le début pour debug

    # 3. GENERATION
    if llm is None: return {"protocol": best_doc, "riskLevel": "Unknown"}

    prompt = (
        f"[INST] <<SYS>>\n"
        f"Tu es un assistant anesthésiste expert. Réponds en FRANÇAIS uniquement.\n"
        f"Si on te demande un Score ASA, analyse bien les descriptions pour choisir la bonne classe (I, II, III...).\n"
        f"<</SYS>>\n\n"
        f"CONTEXTE MÉDICAL :\n{best_doc}\n\n"
        f"QUESTION : {req.query}\n\n"
        f"RÉPONSE (En français, justifiée) : [/INST]"
    )
    
    raw_answer = llm(prompt, max_new_tokens=250, temperature=0.1)
    natural_answer = raw_answer.strip()
    
    is_high_risk = "ARRÊTER" in natural_answer.upper() or "NE DOIT PAS" in best_doc.upper()
    risk = "High" if is_high_risk else "Low"

    return {"protocol": natural_answer, "riskLevel": risk}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)