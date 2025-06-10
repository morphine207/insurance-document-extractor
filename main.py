import os
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF
import google.generativeai as genai
from conversion import parse_extracted_text, build_json_structure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompt for the AI model
SYSTEM_PROMPT = """
You are a highly precise and reliable AI assistant specialized in extracting structured information from German insurance documents.

Your task is to extract specific predefined fields from a given document excerpt and return them in a single valid JSON object.

**IMPORTANT RULES:**
1.  Return **ONLY** a single valid JSON object. Do not include any additional text, explanations, or markdown code blocks outside of the JSON.
2.  Use the **exact field names** as specified in the "Expected JSON schema".
3.  For boolean fields, use lowercase `true` or `false`.
4.  For missing or non-applicable values, use the string `"None"`.
5.  For text values, use double quotes.
6.  Do not add any fields that are not in the "Expected JSON schema".
7.  Do not modify or format the field names.

**GUIDELINES FOR EXTRACTION:**
-   **Prioritize explicit document content:** Extract values directly from the document text.
-   **Formatting and Spelling:** Preserve the exact format and spelling as it appears in the document, **unless a field definition explicitly states a different formatting rule or allows for generalization/simplification.**
-   **No external knowledge/inference (unless specified):** Do not guess, hallucinate, or infer values that are not directly derivable from the provided document text, **unless a field definition explicitly allows or requires such inference/formatting/combination.**

**FIELD DEFINITIONS (Updated based on all prior feedback):**

-   **Gegner1**: The primary **entity** (company, institution, or organization) involved in the case.
    *   *If the document is a letter from a law firm (e.g., BLD, Schmitz Knoth):* This is the law firm's client, which is the **entity** (company, institution, or organization) mentioned in the case title. If both a person and an entity are in the case title (e.g., "Person ./. **Company**" or "**Company** ./. Person"), `Gegner1` is the `Company` if the sender's firm represents the Company.
    *   *If the document is a letter/email directly from an insurance company/financial institution (e.g., Allianz, Generali, Proxalto, DekaBank):* This is the insurance company or financial institution itself, as identified by the letterhead or primary sender name in the document. The name may be a simplified or generalized version of the full legal name if a shorter, common variant is clearly indicated (e.g., "AXA Lebensversicherung AG" instead of "AXA Life Europe dac" if "AXA Lebensversicherung AG" is the common operating name).
    *   *If the document is a court order (e.g., Kostenfestsetzungsbeschluss, Urteil):* This is the **entity** (company, institution, or organization) against whom the order is made or who is ordered to pay costs, typically the Defendant ("Beklagte").
-   **Ansprechpartner Gegner1**: The contact person representing the party identified as `Gegner1`. **Only extract if explicitly mentioned as "Ihr Ansprechpartner" or similar, AND if this contact person is NOT from the sending entity/firm itself (i.e., they represent the *other* side of the case).** Otherwise, return `"None"`.
-   **Az Gegner1**: The primary policy or case number associated with `Gegner1`.
    *   This may involve adding descriptive prefixes (e.g., "Versicherungsnummer:") if the number is clearly a policy number and the prefix is commonly associated with it, even if not explicitly written.
    *   If multiple relevant numbers are present for `Gegner1` and logically belong together (e.g., different policy numbers for the same case), combine them using " & " (e.g., "Nummer 1 & Nummer 2").
    *   If a number is explicitly stated as "bisher" (former) but is still relevant to the case context, extract it.
    *   **Crucially: If a number is not present anywhere in the document, return `"None"`. Do not hallucinate or infer numbers not found.**
-   **Gegenanwalt 1**: The name of the law firm or individual lawyer *representing the party identified as `Gegner1`*.
    *   If `Gegner1` is a law firm (e.g., DB Anwälte), extract its name.
    *   If `Gegner1` is an insurance company/financial institution sending the document directly (i.e., not via a law firm), return `"None"`.
    *   When extracting the firm name, use the primary name without legal form suffixes (e.g., "Rechtsanwälte PartGmbB") or general location suffixes (e.g., "(Köln)") unless the location is integral to distinguishing the firm (e.g., "Göhmann Rechtsanwälte (Braunschweig)" if multiple Göhmann offices exist and the specific office is relevant).
-   **Ansprechpartner Gegenanwalt 1**: The contact person (e.g., named lawyer or associate) within the law firm identified as `Gegenanwalt 1`. **Only extract if explicitly mentioned as "Ihr Ansprechpartner" or similar for `Gegenanwalt 1`. A mere signature by a lawyer is NOT sufficient.** Otherwise, return `"None"`.
-   **A Z Gegenanwalt 1**: The case or reference number (*Aktenzeichen*) of the law firm identified as `Gegenanwalt 1`.
    *   If `Gegenanwalt 1` is `"None"`, then this field must also be `"None"`.
    *   Otherwise, extract the sender's reference number (typically labeled 'Unser Zeichen', 'Az.', or 'Betreff'). If not present, return `"None"`.
-   **Fordert Unterlagen/ Infos**: Whether the opponent is requesting documents or information (true/false).
-   **Fordert Unterlagen D A T**: The deadline (date) by which the opponent requests the documents (if specified).
-   **Fordert Unterlagen atypisch**: Whether the requested documents are atypical or unusual in context (true/false).
-   **Ggs übersendet Unterlagen**: Whether the opponent is sending or has sent any documents (true/false).
-   **L V Unterlagen**: Whether the document mentions life insurance records (*Lebensversicherung*) being requested or submitted (true/false).
-   **W D Unterlagen**: Whether the document mentions disability insurance records (*Wegfall der Dienstfähigkeit*) being requested or submitted (true/false).
-   **P K V Unterlagen**: Whether the document mentions private health insurance records (*Private Krankenversicherung*) being requested or submitted (true/false).
-   **Unterlagen Atypisch**: Description of any atypical or unusual documents mentioned (string or `"None"`).
-   **Antwort auf a G**: Whether the document includes a response to a written request or legal notice (*Anhörung Gegner*) (true/false).
-   **Antwort auf a G (atypisch)**: Whether the response to the opponent is considered atypical (true/false).
-   **Vergleich gg S**: Whether the document discusses a settlement (*Vergleich*) proposed or made by the opponent (true/false).
-   **Frist Vergleich ggs**: Deadline (date) for the opponent’s response or compliance with a proposed settlement.
-   **Vergleich_signed**: Whether the settlement was signed (true/false).
-   **Zahlungsaufforderung K F B**: Whether there is a demand for payment based on a cost order (*Kostenfestsetzungsbeschluss*) (true/false).
-   **Zahlungsaufforderung Urteil/ Vergleich**: Whether there is a demand for payment resulting from a judgment or settlement (true/false).
-   **Fordert Herausgabe entw. K F B/ Urteil**: Whether the opponent requests the release of an invalidated cost order or judgment (true/false).
-   **Sendet entwerteten K F B**: Whether the opponent sends back an invalidated cost order (true/false).
-   **Atypisch**: Any atypical behavior or process mentioned in the document (true/false).
-   **Posteingang zu Du B übertragen**: Whether the incoming mail was transferred to a digital mailbox or document management system (true/false).
-   **Autom. Aktion Post Ggs**: Whether an automated action was triggered based on the opponent’s mail (true/false).
-   **Post Ggs**: Any notable statement about mail sent by the opponent (free-text string or `"None"`).

**Expected JSON schema:**
{
  "Gegner1": "string or None",
  "Ansprechpartner Gegner1": "string or None",
  "Az Gegner1": "string or None",
  "Gegenanwalt 1": "string or None",
  "Ansprechpartner Gegenanwalt 1": "string or None",
  "A Z Gegenanwalt 1": "string or None",
  "Fordert Unterlagen/ Infos": "true/false",
  "Fordert Unterlagen D A T": "string or None",
  "Fordert Unterlagen atypisch": "true/false",
  "Ggs übersendet Unterlagen": "true/false",
  "L V Unterlagen": "true/false",
  "W D Unterlagen": "true/false",
  "P K V Unterlagen": "true/false",
  "Unterlagen Atypisch": "string or None",
  "Antwort auf a G": "true/false",
  "Antwort auf a G (atypisch)": "true/false",
  "Vergleich gg S": "true/false",
  "Frist Vergleich ggs": "string or None",
  "Vergleich_signed": "true/false",
  "Zahlungsaufforderung K F B": "true/false",
  "Zahlungsaufforderung Urteil/ Vergleich": "true/false",
  "Fordert Herausgabe entw. K F B/ Urteil": "true/false",
  "Sendet entwerteten K F B": "true/false",
  "Atypisch": "true/false",
  "Posteingang zu Du B übertragen": "true/false",
  "Autom. Aktion Post Ggs": "true/false",
  "Post Ggs": "string or None"
}
"""

def pdf_to_images(pdf_bytes):
    doc = fitz.open("pdf", pdf_bytes)
    return [Image.frombytes("RGB", [p.width, p.height], p.samples) for p in (page.get_pixmap() for page in doc)]

def clean_json_response(text):
    """Clean and validate the JSON response from the model."""
    # Remove any markdown code block markers
    text = text.replace("```json", "").replace("```", "")
    
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start == -1 or end == 0:
        return None
        
    # Extract just the JSON part
    json_str = text[start:end]
    
    # Try to parse and validate
    try:
        json_obj = json.loads(json_str)
        return json.dumps(json_obj, indent=2)
    except json.JSONDecodeError:
        return None

def main():
    st.title("German Insurance Document Extractor")
    uploaded_file = st.file_uploader("Upload a German insurance PDF", type=["pdf"])

    if uploaded_file:
        pdf_bytes = uploaded_file.read()

        try:
            images = pdf_to_images(pdf_bytes)
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            return

        st.success(f"PDF loaded. {len(images)} page(s) found.")
        st.image(images[0], caption="Preview of Page 1", use_container_width=True)

        if st.button("Extract Document Fields"):
            api_key = os.getenv("GENAI_API_KEY")
            if not api_key:
                st.error("API key not found. Please make sure your .env file is properly configured.")
                return
                
            genai.configure(api_key=api_key)
            
            # Configure model parameters
            generation_config = {
                "temperature": 0.1,  # Lower temperature for more focused outputs
                "max_output_tokens": 9000,  # Increased token limit
                "top_p": 0.95,
                "top_k": 100
            }
            
            model = genai.GenerativeModel(
                'gemini-2.5-flash-preview-05-20',
                generation_config=generation_config
            )

            with st.spinner("Extracting text from document..."):
                try:
                    # Process all images at once
                    response = model.generate_content([SYSTEM_PROMPT] + images)
                    
                    if not response.text:
                        st.error("No content was extracted from the document. Please try again with a different document.")
                        return

                    # Clean and parse the response
                    cleaned_json = clean_json_response(response.text)
                    if cleaned_json is None:
                        st.error("Failed to extract valid JSON from the response")
                        return
                        
                    data = json.loads(cleaned_json)
                    st.subheader("Extracted Fields")
                    st.json(data, expanded=True)

                    # Build structured JSON and save/export if necessary
                    table_rows = parse_extracted_text(cleaned_json)
                    structured_json = build_json_structure(
                        table_rows=table_rows,
                        project_id=101,
                        project_name="InsuranceDocExtraction",
                        project_description="Structured field extraction from German insurance documents",
                        file_id=102,
                        file_name="insurance_doc.json",
                        file_format="json",
                        scanned_file_name="insurance_doc_scan.json",
                        metadata_id=103,
                        tabledata_id=104
                    )

                    st.subheader("Structured JSON Output")
                    st.json(structured_json, expanded=False)

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

if __name__ == "__main__":
    main()
