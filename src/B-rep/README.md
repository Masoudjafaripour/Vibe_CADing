# B-Rep Generation + Retrieval (LLM-Driven)

## Goal

Build a pipeline that:

1. Converts **user text instructions → parametric CAD programs** using an LLM.
2. Executes those programs to generate **valid B-rep CAD files** (STEP/IGES).
3. Indexes generated CAD models and enables **RAG-based retrieval** for reuse and editing.

---

## What This Folder Does

### 1. Text → CAD Program (LLM)

* User provides textual instruction (e.g., "create a cube of side 1m").
* LLM outputs a structured CAD program (JSON/DSL).
* The program defines operations (sketch, extrude, fillet, boolean) and parameters.

### 2. CAD Program → B-Rep (Execution)

* The program is executed using a geometric kernel (e.g., OpenCascade).
* Produces a valid **B-rep solid**.
* Exported as `.step` / `.igs`.

### 3. B-Rep Encoding + Retrieval (RAG)

* Each generated CAD model is encoded (topology + geometry features).
* Embeddings are stored in a vector database.
* New user queries retrieve similar CAD models.
* Retrieved models can be reused, edited, or refined.

---

## File Structure

* `LLM_B_rep.py` → Text-to-CAD program generation
* `example_B_rep.py` → Example B-rep generation + export
* `B_rep_Retrieval.py` → Encoding, embedding, and retrieval logic
* `results/` → Generated CAD files

---

## Conceptual Architecture

Text → LLM → CAD Program → OCC Execution → B-Rep (.STEP)
↓
Embedding Index
↓
RAG Retrieval

---

## Core Idea

LLM = planner
CAD kernel = geometry engine
Vector DB = memory

This project connects language, exact geometry, and retrieval into a unified CAD generation system.
