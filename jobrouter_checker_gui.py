#!/usr/bin/env python3
# Test Direct Commit
# TEST - Bearbeitet am 03.11.2025 mit Claude Code
"""
JobRouter Rechnungs-Checker - Version 5.0
Mit erweiterten Prüfpunkten und transparenter KI-Analyse
"""

import asyncio
import re
import sys
import io
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, 
    QTextEdit, QLabel, QVBoxLayout, QMessageBox, 
    QProgressBar, QMenuBar, QMenu, QTabWidget, QHBoxLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QAction, QColor

# Playwright & PDF
from playwright.async_api import async_playwright
from PyPDF2 import PdfReader


# ========== KI-HELPER (Erweitert mit Transparenz) ==========

class AIHelper:
    """KI-Assistent mit transparenter Analyse"""
    
    def __init__(self, debug_callback=None):
        self.llm = None
        self.available = False
        self.debug_callback = debug_callback
        self.init_llm()
    
    def log_debug(self, message: str):
        """Logging-Hilfsfunktion"""
        if self.debug_callback:
            self.debug_callback(message)
        print(message)
    
    def init_llm(self):
        """Versucht Llama.cpp zu laden"""
        try:
            from llama_cpp import Llama
            import os
            
            # Suche Modell-Datei
            model_paths = [
                os.path.expanduser("~/llama-3.2-3b.gguf"),
                os.path.expanduser("~/Downloads/llama-3.2-3b.gguf"),
                os.path.expanduser("~/.models/llama-3.2-3b.gguf"),
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,  # Mehr Kontext für bessere Analyse
                    n_threads=4,
                    n_gpu_layers=1
                )
                self.available = True
                self.log_debug(f"✅ KI geladen: {model_path}")
            else:
                self.log_debug("ℹ️  KI-Modell nicht gefunden - nutze nur Regex")
                
        except ImportError:
            self.log_debug("ℹ️  llama-cpp-python nicht installiert - nutze nur Regex")
        except Exception as e:
            self.log_debug(f"⚠️  KI-Initialisierung fehlgeschlagen: {e}")
    
    def analyze_field(self, field_name: str, pdf_text: str, expected_value: str = None) -> Dict:
        """
        Universelle Feld-Analyse mit transparentem Prozess
        
        Returns:
            {
                'found': str,           # Gefundener Wert
                'confidence': float,    # 0-1
                'method': str,          # 'regex', 'ai', 'hybrid'
                'reasoning': str,       # Erklärung
                'context': str,         # Kontext wo gefunden
            }
        """
        if not self.available:
            return {
                'found': None,
                'confidence': 0.0,
                'method': 'unavailable',
                'reasoning': 'KI nicht verfügbar',
                'context': ''
            }
        
        self.log_debug(f"\n🔍 KI analysiert: {field_name}")
        self.log_debug(f"   Erwarteter Wert: {expected_value or 'unbekannt'}")
        
        # Erstelle spezifischen Prompt basierend auf Feld
        prompt = self._create_prompt(field_name, pdf_text, expected_value)
        
        try:
            self.log_debug(f"   📤 Sende Anfrage an KI...")
            
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            answer = response['choices'][0]['message']['content'].strip()
            
            self.log_debug(f"   📥 KI-Antwort:\n{answer}")
            
            # Parse Antwort
            result = self._parse_ai_response(field_name, answer, pdf_text)
            
            self.log_debug(f"   ✅ Gefunden: {result['found']} (Konfidenz: {result['confidence']:.0%})")
            
            return result
            
        except Exception as e:
            self.log_debug(f"   ❌ KI-Fehler: {e}")
            return {
                'found': None,
                'confidence': 0.0,
                'method': 'error',
                'reasoning': f'Fehler: {e}',
                'context': ''
            }
    
    def _create_prompt(self, field_name: str, pdf_text: str, expected_value: str = None) -> str:
        """Erstellt feldspezifischen Prompt"""
        
        # Begrenze Text auf relevante Teile
        text_preview = pdf_text[:2000] if len(pdf_text) > 2000 else pdf_text
        
        prompts = {
            'liegenschaft': f"""Analysiere diese Rechnung und finde die LIEGENSCHAFT (Adresse).
Suche nach:
- Straßenname mit Hausnummer (z.B. "Musterstrasse 42")
- Ort/Stadt
- PLZ + Ort Kombination

Ignoriere Rechnungssteller-Adressen, suche die Liegenschaft!

Text:
{text_preview}

Antworte im Format:
GEFUNDEN: [Straße Nummer, PLZ Ort] oder [nicht gefunden]
KONTEXT: [Zeile wo gefunden]
SICHERHEIT: [hoch/mittel/niedrig]""",

            'kreditor': f"""Finde den KREDITOR (Rechnungssteller-Name) in dieser Rechnung.
Der Kreditor steht meist oben oder im QR-Teil.

Text:
{text_preview}

Antworte im Format:
GEFUNDEN: [Firmenname] oder [nicht gefunden]
KONTEXT: [Wo gefunden]
SICHERHEIT: [hoch/mittel/niedrig]""",

            'rechnungsnummer': f"""Finde die RECHNUNGSNUMMER in diesem Dokument.
Suche nach:
- "Rechnung Nr.", "Invoice No.", "Rechnungs-Nr."
- Nummer nach diesen Begriffen

Text:
{text_preview}

Antworte im Format:
GEFUNDEN: [Nummer] oder [nicht gefunden]
KONTEXT: [Wo gefunden]
SICHERHEIT: [hoch/mittel/niedrig]""",

            'bruttobetrag': f"""Finde den BRUTTOBETRAG / TOTAL in CHF.
WICHTIG: Suche ZUERST im QR-Code Bereich unten!
Falls dort kein Betrag: Suche Total/Brutto im Hauptteil.

Text:
{text_preview}

Antworte im Format:
GEFUNDEN: [Betrag] oder [nicht gefunden]
KONTEXT: [QR-Teil oder Rechnung]
SICHERHEIT: [hoch/mittel/niedrig]""",

            'iban': f"""Finde die IBAN (Schweizer Format: CH.. ).
WICHTIG: Suche ZUERST im QR-Code Bereich!
Ignoriere Formatierung (Leerzeichen egal).

Text:
{text_preview}

Antworte im Format:
GEFUNDEN: [IBAN ohne Leerzeichen] oder [nicht gefunden]
KONTEXT: [QR-Teil oder Rechnung]
SICHERHEIT: [hoch/mittel/niedrig]""",

            'qr_ref': f"""Finde die QR-REFERENZNUMMER (26-27 Ziffern).
Suche im QR-Code Bereich nach langer Ziffernfolge.
Ignoriere Formatierung.

Text:
{text_preview}

Antworte im Format:
GEFUNDEN: [nur Ziffern] oder [nicht gefunden]
KONTEXT: [Wo gefunden]
SICHERHEIT: [hoch/mittel/niedrig]"""
        }
        
        return prompts.get(field_name, f"Finde '{field_name}' in:\n{text_preview}")
    
    def _parse_ai_response(self, field_name: str, answer: str, full_text: str) -> Dict:
        """Parst strukturierte KI-Antwort"""
        
        result = {
            'found': None,
            'confidence': 0.0,
            'method': 'ai',
            'reasoning': answer,
            'context': ''
        }
        
        # Extrahiere strukturierte Teile
        if 'GEFUNDEN:' in answer:
            found_match = re.search(r'GEFUNDEN:\s*(.+?)(?:\n|$)', answer)
            if found_match:
                found_value = found_match.group(1).strip()
                if 'nicht gefunden' not in found_value.lower():
                    result['found'] = found_value
        
        if 'KONTEXT:' in answer:
            context_match = re.search(r'KONTEXT:\s*(.+?)(?:\n|$)', answer)
            if context_match:
                result['context'] = context_match.group(1).strip()
        
        if 'SICHERHEIT:' in answer:
            conf_match = re.search(r'SICHERHEIT:\s*(.+?)(?:\n|$)', answer)
            if conf_match:
                confidence_str = conf_match.group(1).strip().lower()
                if 'hoch' in confidence_str:
                    result['confidence'] = 0.9
                elif 'mittel' in confidence_str:
                    result['confidence'] = 0.6
                else:
                    result['confidence'] = 0.3
        
        return result


# ========== HILFSFUNKTIONEN ==========

def normalize_amount(val) -> Optional[float]:
    """Konvertiert '1'234.56 oder 1,234.56 zu 1234.56"""
    if not val:
        return None
    s = str(val).replace("'", "").replace(" ", "")
    s = re.sub(r"\.(?=\d{3}(\D|$))", "", s)
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None


def clean_iban(iban: str) -> str:
    """Entfernt Leerzeichen aus IBAN"""
    if not iban:
        return ""
    cleaned = re.sub(r"\s", "", iban.upper())
    # Entferne alles nach der IBAN (z.B. Firmenname)
    # Schweizer IBAN: CH + 19 weitere Zeichen = 21 total
    if cleaned.startswith('CH') and len(cleaned) >= 21:
        return cleaned[:21]
    return cleaned


def only_digits(s: str) -> str:
    """Extrahiert nur Ziffern"""
    return re.sub(r"\D", "", s) if s else ""


def validate_iban(iban: str) -> bool:
    """Prüft IBAN Checksumme"""
    if not iban:
        return False
    clean = clean_iban(iban)
    if len(clean) < 15:
        return False
    moved = clean[4:] + clean[:4]
    num = ""
    for c in moved:
        if c.isdigit():
            num += c
        else:
            num += str(ord(c) - 55)
    try:
        return int(num) % 97 == 1
    except:
        return False


def validate_qr_ref(ref: str) -> bool:
    """Prüft QR-Referenz mit Modulo 10"""
    digits = only_digits(ref)
    if not digits or len(digits) < 5:
        return False
    table = [0, 9, 4, 6, 8, 2, 7, 1, 3, 5]
    carry = 0
    for d in digits[:-1]:
        carry = table[(carry + int(d)) % 10]
    return ((10 - carry) % 10) == int(digits[-1])


def compare_flexible(value1: str, value2: str) -> Tuple[bool, float]:
    """
    Flexibler Vergleich von Werten (ignoriert Formatierung)

    Returns:
        (match: bool, similarity: float)
    """
    if not value1 or not value2:
        return False, 0.0

    # Normalisiere beide Werte
    v1_clean = only_digits(str(value1))
    v2_clean = only_digits(str(value2))

    if v1_clean == v2_clean:
        return True, 1.0

    # Berechne Ähnlichkeit
    if len(v1_clean) > 0 and len(v2_clean) > 0:
        matches = sum(c1 == c2 for c1, c2 in zip(v1_clean, v2_clean))
        max_len = max(len(v1_clean), len(v2_clean))
        similarity = matches / max_len
        return similarity > 0.9, similarity

    return False, 0.0


def normalize_street_name(text: str) -> str:
    """
    Normalisiert Straßennamen für besseren Vergleich

    Ersetzt Abkürzungen und alternative Schreibweisen
    """
    if not text:
        return ""

    normalized = text.lower().strip()

    # Abkürzungen für Straßentypen
    abbreviations = {
        r'\bstr\.?\b': 'strasse',
        r'\bstrasse\b': 'strasse',
        r'\bstraße\b': 'strasse',
        r'\bweg\b': 'weg',
        r'\bg\.?\b': 'gasse',
        r'\bgasse\b': 'gasse',
        r'\bpl\.?\b': 'platz',
        r'\bplatz\b': 'platz',
        r'\ballee\b': 'allee',
        r'\ball\.?\b': 'allee',
        r'\bring\b': 'ring',
        r'\bhof\b': 'hof',
        r'\bsteig\b': 'steig',
    }

    for pattern, replacement in abbreviations.items():
        normalized = re.sub(pattern, replacement, normalized)

    # Entferne mehrfache Leerzeichen
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized


def fuzzy_match_address(address1: str, address2: str) -> Tuple[bool, float]:
    """
    Flexibler Vergleich von Adressen (ignoriert Reihenfolge, Formatierung und Abkürzungen)

    Returns:
        (match: bool, similarity: float)
    """
    if not address1 or not address2:
        return False, 0.0

    # Normalisiere beide Adressen (Abkürzungen auflösen)
    norm1 = normalize_street_name(address1)
    norm2 = normalize_street_name(address2)

    # Extrahiere Wörter (bereits normalisiert)
    words1 = set(re.findall(r'\b[\wäöüÄÖÜ]+\b', norm1))
    words2 = set(re.findall(r'\b[\wäöüÄÖÜ]+\b', norm2))

    if not words1 or not words2:
        return False, 0.0

    # Berechne Jaccard-Ähnlichkeit
    intersection = words1 & words2
    union = words1 | words2

    similarity = len(intersection) / len(union) if union else 0.0

    # Match wenn mindestens 50% Übereinstimmung
    return similarity >= 0.5, similarity


def fuzzy_match_name(name1: str, name2: str) -> Tuple[bool, float]:
    """
    Flexibler Vergleich von Namen/Firmen (toleriert Abkürzungen und Reihenfolge)

    Returns:
        (match: bool, similarity: float)
    """
    if not name1 or not name2:
        return False, 0.0

    # Normalisiere
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # Exakte Übereinstimmung
    if n1 == n2:
        return True, 1.0

    # Substring-Check (einer enthält den anderen)
    if n1 in n2 or n2 in n1:
        return True, 0.9

    # Wort-basierter Vergleich
    words1 = set(re.findall(r'\b[\wäöüÄÖÜ]+\b', n1))
    words2 = set(re.findall(r'\b[\wäöüÄÖÜ]+\b', n2))

    if not words1 or not words2:
        return False, 0.0

    # Berechne Übereinstimmung
    common = words1 & words2
    total = words1 | words2

    similarity = len(common) / len(total) if total else 0.0

    # Match wenn mindestens 60% gemeinsame Wörter
    return similarity >= 0.6, similarity


# ========== PDF VERIFIZIERUNG (Neue Logik) ==========

class PDFAnalyzer:
    """PDF-Verifizierung: Prüft ob Formular-Werte im PDF vorhanden sind"""

    def __init__(self, debug_callback=None):
        self.debug_callback = debug_callback

    def verify_form_values(self, pdf_bytes: bytes, form_data: Dict, ai_helper: AIHelper = None) -> Dict:
        """
        Verifiziert ob die Formular-Werte im PDF vorhanden sind

        Args:
            pdf_bytes: PDF als Bytes
            form_data: Dict mit erwarteten Werten aus dem Formular
            ai_helper: Optional AI-Helfer für schwierige Fälle

        Returns:
            Dict mit Verifizierungs-Ergebnissen für jedes Feld
        """
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            full_text = "\n".join(p.extract_text() or "" for p in reader.pages)

            if self.debug_callback:
                self.debug_callback(f"\n📄 PDF-Text extrahiert: {len(full_text)} Zeichen")

            # Identifiziere QR-Code Bereich (für bevorzugte Suche)
            qr_section = self._extract_qr_section(full_text)

            results = {}

            # Verifiziere jedes Feld einzeln
            if form_data.get('iban'):
                results['iban'] = self._verify_iban(
                    form_data['iban'],
                    qr_section if qr_section else full_text
                )

            if form_data.get('qr_ref'):
                results['qr_ref'] = self._verify_qr_ref(
                    form_data['qr_ref'],
                    qr_section if qr_section else full_text
                )

            if form_data.get('bruttobetrag'):
                results['bruttobetrag'] = self._verify_amount(
                    form_data['bruttobetrag'],
                    qr_section if qr_section else full_text
                )

            if form_data.get('liegenschaft'):
                results['liegenschaft'] = self._verify_address(
                    form_data['liegenschaft'],
                    full_text
                )

            if form_data.get('kreditor'):
                results['kreditor'] = self._verify_creditor(
                    form_data['kreditor'],
                    full_text
                )

            if form_data.get('rechnungsnummer'):
                results['rechnungsnummer'] = self._verify_invoice_number(
                    form_data['rechnungsnummer'],
                    full_text
                )

            return results

        except Exception as e:
            if self.debug_callback:
                self.debug_callback(f"\n❌ PDF-Verifizierung Fehler: {e}")
                import traceback
                self.debug_callback(traceback.format_exc())
            return {'error': str(e)}
    
    def _extract_qr_section(self, text: str) -> Optional[str]:
        """Extrahiert QR-Code Bereich aus Text (bleibt für bevorzugte Suche)"""
        qr_markers = [
            'Empfangsschein', 'Zahlteil', 'QR-Code',
            'Konto / Zahlbar an', 'Referenz',
            'Zusätzliche Information', 'Währung', 'Betrag'
        ]

        qr_start = -1
        for marker in qr_markers:
            idx = text.lower().find(marker.lower())
            if idx != -1:
                if qr_start == -1 or idx < qr_start:
                    qr_start = idx

        return text[qr_start:] if qr_start != -1 else None

    def _verify_iban(self, expected_iban: str, text: str) -> Dict:
        """Verifiziert ob die erwartete IBAN im PDF vorhanden ist"""
        expected_clean = clean_iban(expected_iban)

        if self.debug_callback:
            self.debug_callback(f"🔍 Suche IBAN: {expected_clean}")

        # Suche alle IBANs im Text
        iban_pattern = r'CH\d{2}\s?[\dA-Z\s]{15,}'
        found_ibans = []

        for match in re.finditer(iban_pattern, text, re.I):
            found_iban = clean_iban(match.group(0))
            if validate_iban(found_iban):
                found_ibans.append(found_iban)

        if self.debug_callback:
            self.debug_callback(f"   Gefundene IBANs im PDF: {len(found_ibans)}")
            if found_ibans:
                self.debug_callback(f"   → {found_ibans[:2]}")

        # Prüfe ob erwartete IBAN dabei ist
        for found in found_ibans:
            if found == expected_clean:
                if self.debug_callback:
                    self.debug_callback(f"   ✅ Match gefunden!")
                return {
                    'found': True,
                    'value': found,
                    'match': 'exact',
                    'confidence': 1.0
                }

        if self.debug_callback:
            self.debug_callback(f"   ❌ Erwartete IBAN nicht gefunden")

        return {
            'found': False,
            'value': None,
            'match': 'not_found',
            'confidence': 0.0,
            'found_alternatives': found_ibans[:3]  # Zeige max 3 gefundene
        }

    def _verify_qr_ref(self, expected_ref: str, text: str) -> Dict:
        """Verifiziert ob die erwartete QR-Referenz im PDF vorhanden ist"""
        expected_digits = only_digits(expected_ref)

        if self.debug_callback:
            self.debug_callback(f"🔍 Suche QR-Ref: {expected_digits}")

        # Suche lange Ziffernfolgen
        ref_patterns = [
            r'\b(\d{26,27})\b',
            r'\b(\d{2}\s?\d{5}\s?\d{5}\s?\d{5}\s?\d{5}\s?\d{5})\b'
        ]

        found_refs = []
        for pattern in ref_patterns:
            for match in re.finditer(pattern, text):
                ref_digits = only_digits(match.group(1))
                if 26 <= len(ref_digits) <= 27:
                    found_refs.append(ref_digits)

        if self.debug_callback:
            self.debug_callback(f"   Gefundene QR-Refs im PDF: {len(found_refs)}")

        # Prüfe Übereinstimmung
        for found in found_refs:
            if found == expected_digits:
                if self.debug_callback:
                    self.debug_callback(f"   ✅ Match gefunden!")
                return {
                    'found': True,
                    'value': found,
                    'match': 'exact',
                    'confidence': 1.0
                }

        if self.debug_callback:
            self.debug_callback(f"   ❌ Erwartete QR-Ref nicht gefunden")

        return {
            'found': False,
            'value': None,
            'match': 'not_found',
            'confidence': 0.0,
            'found_alternatives': found_refs[:3]
        }

    def _verify_amount(self, expected_amount: str, text: str) -> Dict:
        """Verifiziert ob der erwartete Betrag im PDF vorhanden ist"""
        expected_value = normalize_amount(expected_amount)

        if self.debug_callback:
            self.debug_callback(f"🔍 Suche Betrag: CHF {expected_value:.2f}")

        # Suche alle CHF-Beträge
        amount_patterns = [
            r'CHF\s*([\d\'\s,.]+)',
            r'([\d\'\s,.]+)\s*CHF'
        ]

        found_amounts = []
        for pattern in amount_patterns:
            for match in re.finditer(pattern, text, re.I):
                val = normalize_amount(match.group(1))
                if val and 0 < val < 1000000:
                    found_amounts.append(val)

        if self.debug_callback:
            unique_amounts = list(set(found_amounts))[:5]
            self.debug_callback(f"   Gefundene Beträge im PDF: {len(found_amounts)} ({len(set(found_amounts))} unique)")
            if unique_amounts:
                self.debug_callback(f"   → {[f'CHF {a:.2f}' for a in unique_amounts[:3]]}")

        # Prüfe ob erwarteter Betrag dabei ist (mit 0.01 Toleranz)
        for found in found_amounts:
            if abs(found - expected_value) < 0.01:
                if self.debug_callback:
                    self.debug_callback(f"   ✅ Match gefunden!")
                return {
                    'found': True,
                    'value': found,
                    'match': 'exact',
                    'confidence': 1.0
                }

        if self.debug_callback:
            self.debug_callback(f"   ❌ Erwarteter Betrag nicht gefunden")

        return {
            'found': False,
            'value': None,
            'match': 'not_found',
            'confidence': 0.0,
            'found_alternatives': list(set(found_amounts))[:5]
        }

    def _verify_address(self, expected_address: str, text: str) -> Dict:
        """Verifiziert ob die erwartete Adresse im PDF vorhanden ist (flexibel)"""
        if self.debug_callback:
            self.debug_callback(f"🔍 Suche Adresse: {expected_address}")

        # Extrahiere alle Adress-ähnlichen Muster (inkl. Abkürzungen!)
        street_pattern = r'([A-ZÄÖÜa-zäöü\s]+(?:strasse|straße|weg|gasse|platz|ring|allee|hof|steig|str\.?|g\.?|pl\.?|all\.?)\s+\d+[a-z]?)'
        city_pattern = r'(\d{4}\s+[A-ZÄÖÜa-zäöü\s]+)'

        found_addresses = []
        found_addresses.extend(re.findall(street_pattern, text, re.I))
        found_addresses.extend(re.findall(city_pattern, text, re.I))

        if self.debug_callback and found_addresses:
            self.debug_callback(f"   Gefundene Adressen im PDF: {found_addresses[:3]}")

        # Fuzzy-Match gegen erwartete Adresse
        best_match = None
        best_similarity = 0.0

        for found in found_addresses:
            match, similarity = fuzzy_match_address(expected_address, found)
            if self.debug_callback and similarity > 0.3:
                self.debug_callback(f"   Match '{found}': {similarity:.0%}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = found

        if best_similarity >= 0.5:
            return {
                'found': True,
                'value': best_match,
                'match': 'fuzzy',
                'confidence': best_similarity
            }

        return {
            'found': False,
            'value': None,
            'match': 'not_found',
            'confidence': 0.0,
            'found_alternatives': found_addresses[:3]
        }

    def _verify_creditor(self, expected_creditor: str, text: str) -> Dict:
        """Verifiziert ob der erwartete Kreditor im PDF vorhanden ist (flexibel)"""
        if self.debug_callback:
            self.debug_callback(f"🔍 Suche Kreditor: {expected_creditor}")

        # Suche in den ersten 30 Zeilen (Kopfbereich)
        lines = text.split('\n')[:30]

        best_match = None
        best_similarity = 0.0
        candidates = []

        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue

            match, similarity = fuzzy_match_name(expected_creditor, line)
            if similarity > 0.3:
                candidates.append((line, similarity))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = line

        if self.debug_callback and candidates:
            top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
            self.debug_callback(f"   Top Kandidaten:")
            for cand, sim in top_candidates:
                self.debug_callback(f"   → '{cand}' ({sim:.0%})")

        if best_similarity >= 0.6:
            if self.debug_callback:
                self.debug_callback(f"   ✅ Match gefunden: {best_similarity:.0%}")
            return {
                'found': True,
                'value': best_match,
                'match': 'fuzzy',
                'confidence': best_similarity
            }

        if self.debug_callback:
            self.debug_callback(f"   ❌ Kein ausreichender Match (beste Ähnlichkeit: {best_similarity:.0%})")

        return {
            'found': False,
            'value': None,
            'match': 'not_found',
            'confidence': 0.0
        }

    def _verify_invoice_number(self, expected_number: str, text: str) -> Dict:
        """Verifiziert ob die erwartete Rechnungsnummer im PDF vorhanden ist"""
        if self.debug_callback:
            self.debug_callback(f"🔍 Suche Rechnung-Nr: {expected_number}")

        # Normalisiere erwartete Nummer (nur alphanumerisch)
        expected_clean = re.sub(r'[^A-Z0-9]', '', expected_number.upper())

        # Suche Rechnungsnummern
        inv_patterns = [
            r'Rechnung[s]?[- ]?(?:Nr|Nummer|No)[.:]?\s*([A-Z0-9\-/]+)',
            r'Invoice\s*(?:No|Number)[.:]?\s*([A-Z0-9\-/]+)',
            r'Rechnungs[- ]?Nr[.:]?\s*([A-Z0-9\-/]+)'
        ]

        found_numbers = []
        for pattern in inv_patterns:
            for match in re.finditer(pattern, text, re.I):
                found_numbers.append(match.group(1).strip())

        if self.debug_callback:
            self.debug_callback(f"   Gefundene Rechnungsnummern im PDF: {len(found_numbers)}")
            if found_numbers:
                self.debug_callback(f"   → {found_numbers[:3]}")

        # Prüfe Übereinstimmung
        for found in found_numbers:
            found_clean = re.sub(r'[^A-Z0-9]', '', found.upper())
            if found_clean == expected_clean:
                if self.debug_callback:
                    self.debug_callback(f"   ✅ Match gefunden!")
                return {
                    'found': True,
                    'value': found,
                    'match': 'exact',
                    'confidence': 1.0
                }

        if self.debug_callback:
            self.debug_callback(f"   ❌ Erwartete Rechnungsnummer nicht gefunden")

        return {
            'found': False,
            'value': None,
            'match': 'not_found',
            'confidence': 0.0,
            'found_alternatives': found_numbers[:3]
        }


# ========== BROWSER WORKER THREAD ==========

class BrowserWorker(QThread):
    """Thread für Browser-Operations"""
    
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str, str)
    result_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    debug_signal = pyqtSignal(str)
    form_data_signal = pyqtSignal(str)  # NEU: Für Formular-Tab
    
    def __init__(self):
        super().__init__()
        self.browser = None
        self.page = None
        self.captured_pdfs = []
        self.running = True
        self.check_requested = False
        self.ai_helper = None
        self.pdf_analyzer = None
        
    def run(self):
        """Startet Browser"""
        asyncio.run(self.start_browser())
    
    async def start_browser(self):
        """Startet Playwright Browser"""
        try:
            # Initialisiere KI mit Debug-Callback
            self.ai_helper = AIHelper(debug_callback=lambda msg: self.debug_signal.emit(msg))
            self.pdf_analyzer = PDFAnalyzer(debug_callback=lambda msg: self.debug_signal.emit(msg))
            
            self.log_signal.emit("🌐 Starte Browser...")
            
            async with async_playwright() as pw:
                self.browser = await pw.chromium.launch(
                    headless=False,
                    args=["--start-maximized"]
                )
                
                context = await self.browser.new_context(
                    viewport=None,
                    no_viewport=True,
                    ignore_https_errors=True  # Akzeptiere selbst-signierte Zertifikate
                )
                
                self.page = await context.new_page()
                
                # PDF abfangen
                async def on_response(response):
                    try:
                        content_type = response.headers.get("content-type", "")
                        url = response.url
                        
                        if "application/pdf" in content_type or "pdf" in url.lower():
                            body = await response.body()
                            if len(body) > 1000:
                                self.captured_pdfs.append(body)
                                self.log_signal.emit(f"📥 PDF empfangen ({len(body)} bytes)")
                                self.debug_signal.emit(f"PDF URL: {url}")
                    except Exception as e:
                        self.debug_signal.emit(f"Response-Handler Fehler: {e}")
                
                self.page.on("response", lambda r: asyncio.create_task(on_response(r)))
                
                # Navigiere zu JobRouter
                await self.page.goto("https://jobrouter.crowdhouse.ch/jobrouter/")
                
                self.log_signal.emit("✅ Browser bereit!")
                self.status_signal.emit("✅ Browser bereit - Melde dich an", "#28a745")
                self.finished_signal.emit()
                
                # Halte Browser offen
                while self.running:
                    if hasattr(self, 'check_requested') and self.check_requested:
                        self.check_requested = False
                        await self.perform_check()
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            error_msg = f"Browser-Fehler: {e}"
            self.error_signal.emit(error_msg)
            import traceback
            self.debug_signal.emit(traceback.format_exc())
    
    def check_invoice(self):
        """Startet Rechnungsprüfung"""
        self.check_requested = True
    
    async def perform_check(self):
        """Führt Rechnungsprüfung durch"""
        try:
            self.log_signal.emit("\n" + "="*70)
            self.log_signal.emit("🔍 STARTE PRÜFUNG")
            self.log_signal.emit("="*70)
            self.status_signal.emit("🔍 Prüfung läuft...", "#007AFF")
            
            # Formular-Daten holen - VERBESSERT mit detailliertem Logging
            self.debug_signal.emit("\n📋 Lese Formular-Daten...")
            
            form_data = await self.page.evaluate("""
                () => {
                    const data = {};
                    const debug = [];
                    
                    // Helper: Finde Input/Select by verschiedene Attribute
                    function findField(fieldName, keywords) {
                        debug.push(`\\n🔍 Suche: ${fieldName}`);
                        
                        for (const kw of keywords) {
                            debug.push(`   Versuche Keyword: "${kw}"`);
                            
                            // 1. INPUT by name
                            let el = document.querySelector(`input[name*="${kw}" i]`);
                            if (el) {
                                debug.push(`   ✓ Gefunden INPUT via name: ${el.name}`);
                                if (el.value.trim()) {
                                    debug.push(`   ✓ Wert: ${el.value.trim()}`);
                                    return el.value.trim();
                                } else {
                                    debug.push(`   ✗ Feld ist leer`);
                                }
                            }
                            
                            // 2. INPUT by id
                            el = document.querySelector(`input[id*="${kw}" i]`);
                            if (el) {
                                debug.push(`   ✓ Gefunden INPUT via id: ${el.id}`);
                                if (el.value.trim()) {
                                    debug.push(`   ✓ Wert: ${el.value.trim()}`);
                                    return el.value.trim();
                                } else {
                                    debug.push(`   ✗ Feld ist leer`);
                                }
                            }
                            
                            // 3. INPUT by placeholder
                            el = document.querySelector(`input[placeholder*="${kw}" i]`);
                            if (el) {
                                debug.push(`   ✓ Gefunden INPUT via placeholder: ${el.placeholder}`);
                                if (el.value.trim()) {
                                    debug.push(`   ✓ Wert: ${el.value.trim()}`);
                                    return el.value.trim();
                                } else {
                                    debug.push(`   ✗ Feld ist leer`);
                                }
                            }
                            
                            // 4. SELECT (DROPDOWN) by name
                            el = document.querySelector(`select[name*="${kw}" i]`);
                            if (el) {
                                debug.push(`   ✓ Gefunden SELECT via name: ${el.name}`);
                                const selectedOption = el.options[el.selectedIndex];
                                if (selectedOption && selectedOption.text.trim()) {
                                    debug.push(`   ✓ Ausgewählter Wert: ${selectedOption.text.trim()}`);
                                    return selectedOption.text.trim();
                                } else {
                                    debug.push(`   ✗ Keine Auswahl getroffen`);
                                }
                            }
                            
                            // 5. SELECT by id
                            el = document.querySelector(`select[id*="${kw}" i]`);
                            if (el) {
                                debug.push(`   ✓ Gefunden SELECT via id: ${el.id}`);
                                const selectedOption = el.options[el.selectedIndex];
                                if (selectedOption && selectedOption.text.trim()) {
                                    debug.push(`   ✓ Ausgewählter Wert: ${selectedOption.text.trim()}`);
                                    return selectedOption.text.trim();
                                } else {
                                    debug.push(`   ✗ Keine Auswahl getroffen`);
                                }
                            }
                        }
                        
                        debug.push(`   ✗ Nicht gefunden mit Keywords: ${keywords.join(', ')}`);
                        return null;
                    }
                    
                    // 1. IBAN
                    // Bekannte Feldnamen: txt_bank_iban
                    data.iban = findField('IBAN', [
                        'txt_bank_iban',         // JobRouter spezifisch!
                        'bank_iban',
                        'iban', 
                        'bank'
                    ]);
                    
                    // 2. QR-Referenz
                    // Bekannte Feldnamen: txt_bank_esr_reference
                    data.qr_ref = findField('QR-Referenz', [
                        'txt_bank_esr_reference', // JobRouter spezifisch!
                        'bank_esr_reference',
                        'esr_reference',
                        'referenz', 
                        'reference', 
                        'qr'
                    ]);
                    
                    // 3. Bruttobetrag
                    // Bekannte Feldnamen: dec_doc_gros_amount
                    data.bruttobetrag = findField('Bruttobetrag', [
                        'dec_doc_gros_amount',   // JobRouter spezifisch!
                        'doc_gros_amount',
                        'gros_amount',
                        'brutto', 
                        'betrag', 
                        'amount', 
                        'total'
                    ]);
                    
                    // 4. Liegenschaft - AUCH DROPDOWNS!
                    // Bekannte Feldnamen: sql_buchhaltung_nr
                    data.liegenschaft = findField('Liegenschaft', [
                        'sql_buchhaltung_nr',    // JobRouter spezifisch!
                        'buchhaltung_nr',
                        'liegenschaft', 
                        'objekt', 
                        'property', 
                        'adresse', 
                        'immobilie'
                    ]);
                    
                    // 5. Kreditor - AUCH DROPDOWNS!
                    // Bekannte Feldnamen: txt_crd_name
                    data.kreditor = findField('Kreditor', [
                        'txt_crd_name',          // JobRouter spezifisch!
                        'crd_name',
                        'kreditor', 
                        'creditor', 
                        'lieferant', 
                        'firma', 
                        'supplier', 
                        'auftrag'
                    ]);
                    
                    // 6. Rechnungsnummer
                    // Bekannte Feldnamen: txt_doc_no
                    data.rechnungsnummer = findField('Rechnungsnummer', [
                        'txt_doc_no',            // JobRouter spezifisch!
                        'doc_no',
                        'rechnungsnummer', 
                        'rechnung', 
                        'invoice', 
                        'nummer', 
                        'rechnungs-nr'
                    ]);
                    
                    // Zusätzlich: Liste ALLE Felder zur Analyse
                    debug.push('\\n📝 ALLE gefundenen INPUT-Felder:');
                    const allInputs = document.querySelectorAll('input[type="text"], input[type="number"]');
                    allInputs.forEach((input, idx) => {
                        const name = input.name || 'kein name';
                        const id = input.id || 'kein id';
                        const placeholder = input.placeholder || 'kein placeholder';
                        const value = input.value.trim() || '(leer)';
                        debug.push(`   ${idx+1}. INPUT: name="${name}" id="${id}" placeholder="${placeholder}" value="${value}"`);
                    });
                    
                    debug.push('\\n🔽 ALLE gefundenen SELECT-Felder (Dropdowns):');
                    const allSelects = document.querySelectorAll('select');
                    allSelects.forEach((select, idx) => {
                        const name = select.name || 'kein name';
                        const id = select.id || 'kein id';
                        const selectedOption = select.options[select.selectedIndex];
                        const value = selectedOption ? selectedOption.text.trim() : '(keine Auswahl)';
                        debug.push(`   ${idx+1}. SELECT: name="${name}" id="${id}" ausgewählt="${value}"`);
                    });
                    
                    return { data, debug: debug.join('\\n') };
                }
            """)
            
            # Extrahiere Debug-Info
            form_debug = form_data.get('debug', '')
            actual_data = form_data.get('data', {})
            
            # Zeige detailliertes Debug
            if form_debug:
                self.debug_signal.emit(form_debug)
            
            # Sende schön formatierte Formular-Daten an eigenen Tab
            form_display = "📝 FORMULAR-DATEN (aus JobRouter-Seite)\n"
            form_display += "="*70 + "\n"
            form_display += f"Zeitpunkt: {datetime.now().strftime('%H:%M:%S')}\n"
            form_display += "="*70 + "\n\n"
            
            for key, val in actual_data.items():
                status = "✅" if val else "❌"
                form_display += f"{status} {key.upper()}:\n"
                form_display += f"   Wert: {val if val else '(leer)'}\n\n"
            
            form_display += "\n" + "="*70 + "\n"
            form_display += "💡 HINWEIS:\n"
            form_display += "Wenn Felder leer sind, prüfe:\n"
            form_display += "  1. Sind die Felder im JobRouter ausgefüllt?\n"
            form_display += "  2. Stimmen die Feldnamen (siehe Debug-Tab für Details)\n"
            form_display += "  3. JavaScript-Selektoren müssen evtl. angepasst werden\n"
            
            self.form_data_signal.emit(form_display)
            
            self.log_signal.emit("\n📋 FORMULAR-WERTE:")
            self.log_signal.emit("-"*70)
            for key, val in actual_data.items():
                display_val = val if val else "❌ Leer"
                self.log_signal.emit(f"  • {key}: {display_val}")
            
            # PDF prüfen - VERBESSERTE METHODE
            if not self.captured_pdfs:
                self.debug_signal.emit("\n⚠️  Kein PDF durch Response abgefangen!")
                self.log_signal.emit("🔍 Suche eingebettete PDFs auf der Seite...")

                # Methode 1: Suche nach iframe/embed/object-Tags mit PDF
                self.debug_signal.emit("\n📌 Methode 1: Suche iframe/embed/object-Tags...")
                pdf_elements = await self.page.evaluate("""
                    () => {
                        const elements = [];

                        // Suche alle iframes
                        document.querySelectorAll('iframe').forEach((el, idx) => {
                            elements.push({
                                type: 'iframe',
                                src: el.src || el.getAttribute('src'),
                                id: el.id,
                                class: el.className,
                                index: idx
                            });
                        });

                        // Suche alle embeds
                        document.querySelectorAll('embed').forEach((el, idx) => {
                            elements.push({
                                type: 'embed',
                                src: el.src || el.getAttribute('src'),
                                contentType: el.type,
                                index: idx
                            });
                        });

                        // Suche alle objects
                        document.querySelectorAll('object').forEach((el, idx) => {
                            elements.push({
                                type: 'object',
                                src: el.data || el.getAttribute('data'),
                                contentType: el.type,
                                index: idx
                            });
                        });

                        return elements;
                    }
                """)

                self.debug_signal.emit(f"Gefundene Elemente: {len(pdf_elements) if pdf_elements else 0}")
                if pdf_elements:
                    for elem in pdf_elements:
                        if elem is None:
                            continue
                        try:
                            elem_type = elem.get('type', 'unbekannt')
                            elem_src = elem.get('src', 'keine src')
                            if elem_src and len(elem_src) > 100:
                                elem_src = elem_src[:100]
                            self.debug_signal.emit(f"  • {elem_type}: {elem_src}")
                        except Exception as e:
                            self.debug_signal.emit(f"  • Fehler beim Anzeigen: {str(e)}")

                # Versuche PDF aus jedem gefundenen Element zu laden
                if pdf_elements:
                    for elem in pdf_elements:
                        if elem is None:
                            continue

                        src = elem.get('src')
                        if not src:
                            continue

                        # Ignoriere leere oder ungültige URLs
                        if src in ['about:blank', '', None]:
                            continue

                        # Prüfe ob es ein PDF sein könnte
                        is_pdf_candidate = (
                            'pdf' in src.lower() or
                            elem.get('contentType', '').lower() == 'application/pdf' or
                            '.pdf' in src.lower()
                        )

                        if is_pdf_candidate:
                            try:
                                self.log_signal.emit(f"📄 PDF-Element gefunden ({elem.get('type', 'unbekannt')}): {src[:80]}")
                                self.debug_signal.emit(f"Versuche PDF zu laden von: {src}")

                                # Methode A: Versuche direkten Download via Request
                                try:
                                    response = await self.page.context.request.get(src)
                                    if response.ok:
                                        body = await response.body()
                                        if len(body) > 1000:
                                            self.captured_pdfs.append(body)
                                            self.log_signal.emit(f"✅ PDF aus {elem.get('type', 'unbekannt')} geladen ({len(body)} bytes)")
                                            break
                                    else:
                                        self.debug_signal.emit(f"   ✗ Request Response status: {response.status}")
                                except Exception as req_error:
                                    self.debug_signal.emit(f"   ✗ Request Fehler: {str(req_error)[:100]}")

                                    # Methode B: Versuche mit neuer Page (Fallback für SSL-Probleme)
                                    if not self.captured_pdfs:
                                        self.debug_signal.emit(f"   → Versuche mit neuer Page...")
                                        try:
                                            pdf_page = await self.page.context.new_page()

                                            # Warte auf Response beim Navigieren
                                            async with pdf_page.expect_response(lambda r: True) as response_info:
                                                await pdf_page.goto(src, wait_until="networkidle", timeout=10000)

                                            response = await response_info.value
                                            if response.ok:
                                                body = await response.body()
                                                if len(body) > 1000:
                                                    self.captured_pdfs.append(body)
                                                    self.log_signal.emit(f"✅ PDF mit neuer Page geladen ({len(body)} bytes)")
                                                    await pdf_page.close()
                                                    break
                                            await pdf_page.close()
                                        except Exception as page_error:
                                            self.debug_signal.emit(f"   ✗ Page Fehler: {str(page_error)[:100]}")

                            except Exception as e:
                                self.debug_signal.emit(f"   ✗ Allgemeiner Fehler: {str(e)[:100]}")

                # Methode 2: Suche JavaScript-generierte PDF-URLs
                if not self.captured_pdfs:
                    self.debug_signal.emit("\n📌 Methode 2: Suche JavaScript-generierte PDFs...")

                    try:
                        # Suche nach allen window.open oder ähnlichen JavaScript-Aufrufen
                        js_pdf_info = await self.page.evaluate("""
                            () => {
                                const info = {
                                    blobUrls: [],
                                    dataUrls: [],
                                    pdfViewers: []
                                };

                                // Suche nach PDF.js oder ähnlichen PDF-Viewern
                                const scripts = Array.from(document.querySelectorAll('script'));
                                scripts.forEach(script => {
                                    const content = script.textContent || '';
                                    if (content.includes('pdf.js') || content.includes('pdfjs')) {
                                        info.pdfViewers.push('PDF.js gefunden');
                                    }
                                });

                                // Suche nach Blob URLs im gesamten DOM
                                const allElements = document.querySelectorAll('*');
                                allElements.forEach(el => {
                                    ['src', 'href', 'data'].forEach(attr => {
                                        const value = el.getAttribute(attr);
                                        if (value) {
                                            if (value.startsWith('blob:')) {
                                                info.blobUrls.push(value);
                                            } else if (value.startsWith('data:application/pdf')) {
                                                info.dataUrls.push(value.substring(0, 100) + '...');
                                            }
                                        }
                                    });
                                });

                                // Entferne Duplikate
                                info.blobUrls = [...new Set(info.blobUrls)];
                                info.dataUrls = [...new Set(info.dataUrls)];

                                return info;
                            }
                        """)

                        self.debug_signal.emit(f"PDF-Viewer: {js_pdf_info.get('pdfViewers', [])}")
                        self.debug_signal.emit(f"Blob URLs: {len(js_pdf_info.get('blobUrls', []))}")
                        self.debug_signal.emit(f"Data URLs: {len(js_pdf_info.get('dataUrls', []))}")

                        # Versuche Data URLs zu laden
                        for data_url in js_pdf_info.get('dataUrls', [])[:3]:
                            try:
                                self.log_signal.emit(f"📄 Versuche Data-URL PDF zu laden...")
                                # Data URL direkt dekodieren
                                import base64
                                if ',' in data_url:
                                    data = data_url.split(',', 1)[1]
                                    pdf_data = base64.b64decode(data)
                                    if len(pdf_data) > 1000:
                                        self.captured_pdfs.append(pdf_data)
                                        self.log_signal.emit(f"✅ PDF aus Data-URL geladen ({len(pdf_data)} bytes)")
                                        break
                            except Exception as e:
                                self.debug_signal.emit(f"   ✗ Data-URL Fehler: {e}")

                        # Wenn PDF.js gefunden wurde, suche nach dem PDF-Canvas
                        if js_pdf_info.get('pdfViewers') and not self.captured_pdfs:
                            self.debug_signal.emit("Suche nach PDF.js Canvas...")
                            # Warte kurz, damit PDF.js laden kann
                            await self.page.wait_for_timeout(2000)

                            # Versuche PDF-URL aus PDF.js zu extrahieren
                            pdf_js_url = await self.page.evaluate("""
                                () => {
                                    // Suche nach PDFViewerApplication (Standard PDF.js)
                                    if (window.PDFViewerApplication && window.PDFViewerApplication.url) {
                                        return window.PDFViewerApplication.url;
                                    }
                                    // Suche nach anderen PDF-Viewer-Variablen
                                    if (window.pdfDocument && window.pdfDocument.url) {
                                        return window.pdfDocument.url;
                                    }
                                    return null;
                                }
                            """)

                            if pdf_js_url:
                                self.log_signal.emit(f"📄 PDF.js URL gefunden: {pdf_js_url[:80]}")
                                try:
                                    response = await self.page.context.request.get(pdf_js_url)
                                    if response.ok:
                                        body = await response.body()
                                        if len(body) > 1000:
                                            self.captured_pdfs.append(body)
                                            self.log_signal.emit(f"✅ PDF aus PDF.js geladen ({len(body)} bytes)")
                                except Exception as e:
                                    self.debug_signal.emit(f"   ✗ PDF.js URL Fehler: {e}")
                            else:
                                self.debug_signal.emit("Keine PDF.js URL gefunden")

                    except Exception as e:
                        self.debug_signal.emit(f"JavaScript-PDF-Suche Fehler: {e}")

                # Methode 3: Prüfe alle Frames (Fallback)
                if not self.captured_pdfs:
                    self.debug_signal.emit("\n📌 Methode 3: Prüfe alle Frames...")
                    frames = self.page.frames
                    self.debug_signal.emit(f"Anzahl Frames: {len(frames)}")

                    for i, frame in enumerate(frames):
                        try:
                            url = frame.url
                            self.debug_signal.emit(f"Frame {i}: {url[:100] if url else 'keine URL'}")

                            if url and ("pdf" in url.lower() or url.endswith(".pdf")):
                                self.log_signal.emit(f"📄 PDF-Frame gefunden: {url[:80]}")

                                # Versuche mit Request API
                                try:
                                    response = await self.page.context.request.get(url)
                                    if response.ok:
                                        body = await response.body()
                                        if len(body) > 1000:
                                            self.captured_pdfs.append(body)
                                            self.log_signal.emit(f"✅ PDF aus Frame geladen ({len(body)} bytes)")
                                            break
                                except Exception as req_err:
                                    self.debug_signal.emit(f"   ✗ Request Fehler: {str(req_err)[:80]}")

                                    # Fallback: Versuche mit neuer Page
                                    if not self.captured_pdfs:
                                        try:
                                            pdf_page = await self.page.context.new_page()
                                            async with pdf_page.expect_response(lambda r: True) as response_info:
                                                await pdf_page.goto(url, wait_until="networkidle", timeout=10000)

                                            response = await response_info.value
                                            if response.ok:
                                                body = await response.body()
                                                if len(body) > 1000:
                                                    self.captured_pdfs.append(body)
                                                    self.log_signal.emit(f"✅ PDF aus Frame mit neuer Page geladen ({len(body)} bytes)")
                                                    await pdf_page.close()
                                                    break
                                            await pdf_page.close()
                                        except Exception as page_err:
                                            self.debug_signal.emit(f"   ✗ Page Fehler: {str(page_err)[:80]}")
                        except Exception as e:
                            self.debug_signal.emit(f"Frame {i} Fehler: {str(e)[:100]}")

                # Methode 4: Suche nach PDF-Links und lade das erste
                if not self.captured_pdfs:
                    self.debug_signal.emit("\n📌 Methode 4: Suche PDF-Links auf Seite...")
                    pdf_links = await self.page.evaluate("""
                        () => {
                            const links = [];
                            document.querySelectorAll('a[href*=".pdf"], a[href*="pdf"]').forEach(a => {
                                links.push(a.href);
                            });
                            return links;
                        }
                    """)

                    self.debug_signal.emit(f"Gefundene PDF-Links: {len(pdf_links)}")
                    for link in pdf_links[:3]:
                        self.debug_signal.emit(f"  • {link[:100]}")

                    # Versuche ersten PDF-Link
                    if pdf_links:
                        link = pdf_links[0]
                        self.log_signal.emit(f"📄 Versuche PDF-Link: {link[:80]}")

                        # Versuche mit Request API
                        try:
                            response = await self.page.context.request.get(link)
                            if response.ok:
                                body = await response.body()
                                if len(body) > 1000:
                                    self.captured_pdfs.append(body)
                                    self.log_signal.emit(f"✅ PDF aus Link geladen ({len(body)} bytes)")
                        except Exception as req_err:
                            self.debug_signal.emit(f"   ✗ Link Request Fehler: {str(req_err)[:80]}")

                            # Fallback: Versuche mit neuer Page
                            if not self.captured_pdfs:
                                try:
                                    pdf_page = await self.page.context.new_page()
                                    async with pdf_page.expect_response(lambda r: True) as response_info:
                                        await pdf_page.goto(link, wait_until="networkidle", timeout=10000)

                                    response = await response_info.value
                                    if response.ok:
                                        body = await response.body()
                                        if len(body) > 1000:
                                            self.captured_pdfs.append(body)
                                            self.log_signal.emit(f"✅ PDF aus Link mit neuer Page geladen ({len(body)} bytes)")
                                    await pdf_page.close()
                                except Exception as page_err:
                                    self.debug_signal.emit(f"   ✗ Link Page Fehler: {str(page_err)[:80]}")
            
            if not self.captured_pdfs:
                error_msg = "❌ Kein PDF gefunden! Bitte Rechnung neu laden (F5)."
                self.error_signal.emit(error_msg)
                self.status_signal.emit("⚠️ Kein PDF", "#ffc107")
                self.result_signal.emit({'success': False, 'error': 'Kein PDF gefunden'})
                return
            
            # PDF-Daten verifizieren mit neuer Logik
            self.log_signal.emit("\n📄 VERIFIZIERE FORMULAR-WERTE IM PDF...")
            verification_results = self.pdf_analyzer.verify_form_values(
                self.captured_pdfs[-1],
                actual_data,
                self.ai_helper
            )

            self.log_signal.emit("\n📄 VERIFIZIERUNGS-ERGEBNIS:")
            self.log_signal.emit("-"*70)
            for key, result in verification_results.items():
                if key != 'error':
                    if result['found']:
                        confidence = f" ({result['confidence']:.0%})" if result.get('confidence') else ""
                        self.log_signal.emit(f"  ✅ {key}: Gefunden{confidence}")
                        if result.get('value'):
                            self.log_signal.emit(f"      → {result['value']}")
                    else:
                        self.log_signal.emit(f"  ❌ {key}: Nicht gefunden")
                        if result.get('found_alternatives'):
                            alts = result['found_alternatives']
                            self.log_signal.emit(f"      Alternativen: {alts[:2]}")

            # Erstelle Prüf-Ergebnisse
            results = self._build_results(actual_data, verification_results)
            
            # Ergebnis ausgeben
            all_ok = all(r['ok'] for r in results if r['ok'] is not None)
            
            self.log_signal.emit("\n" + "="*70)
            self.log_signal.emit("📊 PRÜF-ERGEBNIS:")
            self.log_signal.emit("="*70)
            
            for r in results:
                status = "✅" if r['ok'] else "❌" if r['ok'] is False else "⚠️"
                self.log_signal.emit(f"{status} {r['field'].upper()}: {r['msg']}")
                if not r['ok']:
                    self.log_signal.emit(f"   Formular: {r['expected']}")
                    self.log_signal.emit(f"   PDF: {r['found']}")
                    if r.get('detail'):
                        self.log_signal.emit(f"   Detail: {r['detail']}")
            
            if all_ok:
                self.status_signal.emit("✅ Alle Prüfungen bestanden!", "#28a745")
            else:
                failed_count = sum(1 for r in results if r['ok'] is False)
                self.status_signal.emit(f"❌ {failed_count} Fehler gefunden", "#dc3545")
            
            self.result_signal.emit({
                'success': all_ok,
                'results': results,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            
        except Exception as e:
            error_msg = f"Fehler bei Prüfung: {e}"
            self.error_signal.emit(error_msg)
            import traceback
            trace = traceback.format_exc()
            self.debug_signal.emit(trace)
            self.status_signal.emit("❌ Fehler bei Prüfung", "#dc3545")
            self.result_signal.emit({'success': False, 'error': str(e)})
    
    def _build_results(self, form_data: Dict, verification_results: Dict) -> List[Dict]:
        """Erstellt nutzerfreundliche Ergebnis-Liste aus Verifizierungs-Daten"""
        results = []

        # Alle Felder durchgehen
        fields = ['iban', 'qr_ref', 'bruttobetrag', 'liegenschaft', 'kreditor', 'rechnungsnummer']

        for field in fields:
            form_value = form_data.get(field)
            verification = verification_results.get(field, {})

            # Feld nicht im Formular ausgefüllt
            if not form_value:
                results.append({
                    'field': field,
                    'ok': None,
                    'expected': "Nicht gesetzt",
                    'found': "N/A",
                    'msg': 'Feld nicht ausgefüllt',
                    'detail': None
                })
                continue

            # Feld wurde verifiziert
            found = verification.get('found', False)
            confidence = verification.get('confidence', 0.0)
            pdf_value = verification.get('value')
            match_type = verification.get('match', 'unknown')

            if found:
                # Erfolgreich gefunden
                if match_type == 'exact':
                    msg = '✔ Exakte Übereinstimmung'
                elif match_type == 'fuzzy':
                    msg = f'✔ Gefunden (Ähnlichkeit: {confidence:.0%})'
                else:
                    msg = '✔ Gefunden'

                results.append({
                    'field': field,
                    'ok': True,
                    'expected': str(form_value),
                    'found': str(pdf_value) if pdf_value else str(form_value),
                    'msg': msg,
                    'detail': None
                })
            else:
                # Nicht gefunden
                alternatives = verification.get('found_alternatives', [])
                detail = None

                if alternatives:
                    # Zeige was stattdessen gefunden wurde
                    if field in ['iban', 'qr_ref', 'rechnungsnummer']:
                        detail = f"Gefunden: {alternatives[:2]}"
                    elif field == 'bruttobetrag':
                        detail = f"Gefunden: {[f'CHF {a:.2f}' for a in alternatives[:3]]}"
                    else:
                        detail = f"Gefunden: {alternatives[:2]}"

                results.append({
                    'field': field,
                    'ok': False,
                    'expected': str(form_value),
                    'found': "Nicht gefunden",
                    'msg': '✗ Wert nicht im PDF gefunden',
                    'detail': detail
                })

        return results
    
    def stop(self):
        """Stoppt Browser"""
        self.running = False


# ========== GUI WINDOW (Erweitert) ==========

class JobRouterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.worker = None
        self.browser_started = False
        self.init_ui()
        
        # Initiale Nachrichten
        self.add_log("🚀 JobRouter Checker V5.0 gestartet")
        self.add_log(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        self.add_log("\n✨ NEU in V5.0:")
        self.add_log("  • 6 Prüfpunkte (inkl. Liegenschaft, Kreditor, Rechnung-Nr.)")
        self.add_log("  • QR-Code bevorzugte Extraktion")
        self.add_log("  • Transparente KI-Analyse")
        self.add_log("  • Verbesserte Fehlerbehandlung")
        self.add_log("\n💡 Klicke 'Browser starten' um zu beginnen\n")
    
    def init_ui(self):
        """Erstellt GUI"""
        self.setWindowTitle("JobRouter Rechnungs-Checker V5.0")
        self.setGeometry(100, 100, 1000, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Titel
        title = QLabel("JobRouter Rechnungs-Checker V5.0")
        title.setFont(QFont(".AppleSystemUIFont", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #1d1d1f; padding: 10px;")
        layout.addWidget(title)
        
        # Status
        self.status_label = QLabel("⚪ Bereit - Klicke 'Browser starten'")
        self.status_label.setFont(QFont(".AppleSystemUIFont", 14))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            padding: 12px;
            background-color: #f5f5f7;
            border-radius: 8px;
            color: #1d1d1f;
        """)
        layout.addWidget(self.status_label)
        
        # Button Container
        button_layout = QHBoxLayout()
        
        # Browser Button
        self.browser_button = QPushButton("🌐 Browser starten")
        self.browser_button.setFont(QFont(".AppleSystemUIFont", 14, QFont.Weight.Bold))
        self.browser_button.setMinimumHeight(50)
        self.browser_button.setStyleSheet("""
            QPushButton {
                background-color: #34C759;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #2DB04A; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.browser_button.clicked.connect(self.start_browser)
        button_layout.addWidget(self.browser_button)
        
        # Check Button
        self.check_button = QPushButton("🔍 Rechnung prüfen")
        self.check_button.setFont(QFont(".AppleSystemUIFont", 14, QFont.Weight.Bold))
        self.check_button.setMinimumHeight(50)
        self.check_button.setEnabled(False)
        self.check_button.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #0051D5; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.check_button.clicked.connect(self.run_check)
        button_layout.addWidget(self.check_button)
        
        layout.addLayout(button_layout)
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #007AFF;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007AFF;
            }
        """)
        layout.addWidget(self.progress)
        
        # Result Label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont(".AppleSystemUIFont", 12))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setVisible(False)
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d1d1d6;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f5f5f7;
                color: #1d1d1f;
                border: 1px solid #d1d1d6;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
                font-weight: bold;
            }
        """)
        
        # Protokoll Tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Menlo", 11))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #1d1d1f;
                border: none;
                padding: 10px;
            }
        """)
        self.tabs.addTab(self.log_text, "📋 Protokoll")
        
        # Formular-Daten Tab (NEU!)
        self.form_text = QTextEdit()
        self.form_text.setReadOnly(True)
        self.form_text.setFont(QFont("Menlo", 11))
        self.form_text.setStyleSheet("""
            QTextEdit {
                background-color: #f0f8ff;
                color: #1d1d1f;
                border: none;
                padding: 10px;
            }
        """)
        self.tabs.addTab(self.form_text, "📝 Formular-Daten")
        
        # Fehler Tab
        self.error_text = QTextEdit()
        self.error_text.setReadOnly(True)
        self.error_text.setFont(QFont("Menlo", 11))
        self.error_text.setStyleSheet("""
            QTextEdit {
                background-color: #fff5f5;
                color: #dc3545;
                border: none;
                padding: 10px;
            }
        """)
        self.tabs.addTab(self.error_text, "⚠️ Fehler")
        
        # Debug/KI Tab
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setFont(QFont("Menlo", 10))
        self.debug_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                color: #495057;
                border: none;
                padding: 10px;
            }
        """)
        self.tabs.addTab(self.debug_text, "🔬 Debug / KI-Analyse")
        
        layout.addWidget(self.tabs)
        
        # Menu
        self.create_menu()
    
    def create_menu(self):
        """Erstellt Menüleiste"""
        menubar = self.menuBar()
        
        # Datei
        file_menu = menubar.addMenu("Datei")
        
        export_action = QAction("Log exportieren", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_log)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Beenden", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Bearbeiten
        edit_menu = menubar.addMenu("Bearbeiten")
        
        clear_action = QAction("Log leeren", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.clear_log)
        edit_menu.addAction(clear_action)
        
        # Hilfe
        help_menu = menubar.addMenu("Hilfe")
        
        about_action = QAction("Über", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def start_browser(self):
        """Startet Browser-Worker"""
        if not self.browser_started:
            self.browser_button.setEnabled(False)
            self.browser_button.setText("⏳ Browser wird gestartet...")
            
            self.worker = BrowserWorker()
            self.worker.log_signal.connect(self.add_log)
            self.worker.error_signal.connect(self.add_error)
            self.worker.debug_signal.connect(self.add_debug)
            self.worker.form_data_signal.connect(self.add_form_data)  # NEU!
            self.worker.status_signal.connect(self.update_status)
            self.worker.result_signal.connect(self.show_result)
            self.worker.finished_signal.connect(self.on_browser_ready)
            self.worker.start()
            self.browser_started = True
    
    def on_browser_ready(self):
        """Browser ist bereit"""
        self.browser_button.setText("✅ Browser läuft")
        self.check_button.setEnabled(True)
    
    def run_check(self):
        """Startet Prüfung"""
        if self.worker:
            self.progress.setVisible(True)
            self.result_label.setVisible(False)
            self.check_button.setEnabled(False)
            
            # Clear previous debug
            self.debug_text.clear()
            
            QTimer.singleShot(100, self.worker.check_invoice)
    
    def show_result(self, result):
        """Zeigt Ergebnis"""
        self.progress.setVisible(False)
        self.check_button.setEnabled(True)
        
        if 'error' in result:
            self.result_label.setText(f"❌ Fehler: {result['error']}")
            self.result_label.setStyleSheet("padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px;")
            self.result_label.setVisible(True)
            return
        
        if result['success']:
            self.result_label.setText(f"✅ Alle 6 Prüfungen bestanden! ({result.get('timestamp', '')})")
            self.result_label.setStyleSheet("padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px;")
        else:
            results = result.get('results', [])
            failed = [r for r in results if r['ok'] is False]
            warning = [r for r in results if r['ok'] is None]
            
            msg = f"❌ {len(failed)} Fehler"
            if warning:
                msg += f" • ⚠️ {len(warning)} Warnungen"
            msg += f" ({result.get('timestamp', '')})"
            
            self.result_label.setText(msg)
            self.result_label.setStyleSheet("padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px;")
        
        self.result_label.setVisible(True)
    
    def update_status(self, text, color):
        """Aktualisiert Status"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            padding: 12px;
            background-color: {color};
            border-radius: 8px;
            color: white;
            font-weight: bold;
        """)
    
    def add_log(self, message):
        """Fügt Log hinzu"""
        self.log_text.append(message)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
    
    def add_error(self, message):
        """Fügt Fehler hinzu"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_entry = f"[{timestamp}] {message}"
        self.error_text.append(error_entry)
        self.error_text.moveCursor(QTextCursor.MoveOperation.End)
        
        # Update tab badge
        error_lines = [line for line in self.error_text.toPlainText().split('\n') if line.strip()]
        error_count = len(error_lines)
        self.tabs.setTabText(2, f"⚠️ Fehler ({error_count})")  # Index 2 wegen neuem Tab
        
        # Log auch im Debug
        self.add_debug(f"ERROR: {message}")
    
    def add_debug(self, message):
        """Fügt Debug-Info hinzu"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.debug_text.append(f"[{timestamp}] {message}")
        self.debug_text.moveCursor(QTextCursor.MoveOperation.End)
    
    def add_form_data(self, message):
        """Fügt Formular-Daten hinzu"""
        self.form_text.setPlainText(message)
        self.form_text.moveCursor(QTextCursor.MoveOperation.End)
    
    def clear_log(self):
        """Leert Logs"""
        reply = QMessageBox.question(
            self,
            "Logs leeren",
            "Möchtest du alle Logs löschen?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.log_text.clear()
            self.form_text.clear()
            self.error_text.clear()
            self.debug_text.clear()
            self.tabs.setTabText(2, "⚠️ Fehler")  # Index angepasst
            self.add_log("📋 Logs geleert\n")
    
    def export_log(self):
        """Exportiert Logs"""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Logs exportieren",
            f"jobrouter_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write("JOBROUTER CHECKER V5.0 - LOG\n")
                    f.write(f"Exportiert: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
                    f.write("="*70 + "\n\n")
                    
                    f.write("📋 PROTOKOLL:\n")
                    f.write("-"*70 + "\n")
                    f.write(self.log_text.toPlainText())
                    f.write("\n\n")
                    
                    f.write("📝 FORMULAR-DATEN:\n")
                    f.write("-"*70 + "\n")
                    f.write(self.form_text.toPlainText() or "Keine Daten")
                    f.write("\n\n")
                    
                    f.write("⚠️ FEHLER:\n")
                    f.write("-"*70 + "\n")
                    f.write(self.error_text.toPlainText() or "Keine Fehler")
                    f.write("\n\n")
                    
                    f.write("🔬 DEBUG / KI-ANALYSE:\n")
                    f.write("-"*70 + "\n")
                    f.write(self.debug_text.toPlainText() or "Keine Debug-Info")
                
                QMessageBox.information(self, "✅ Erfolg", f"Log gespeichert:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "❌ Fehler", f"Fehler beim Speichern:\n{e}")
    
    def show_about(self):
        """Zeigt About-Dialog"""
        QMessageBox.about(
            self,
            "Über JobRouter Checker V5.0",
            """
            <h2>JobRouter Rechnungs-Checker</h2>
            <p><b>Version 5.0 - Extended Edition</b></p>
            <p>Automatische Prüfung von Rechnungen in JobRouter.</p>
            
            <p><b>Neue Features V5.0:</b></p>
            <ul>
            <li><b>6 Prüfpunkte:</b> IBAN, QR-Ref, Betrag, Liegenschaft, Kreditor, Rechnung-Nr.</li>
            <li><b>QR-Code Fokus:</b> Bevorzugte Extraktion aus QR-Teil</li>
            <li><b>Transparente KI:</b> Zeigt KI-Denkprozess im Debug-Tab</li>
            <li><b>Flexible Vergleiche:</b> Format-unabhängige Prüfung</li>
            <li><b>Bessere Fehlerbehandlung:</b> Detailliertes Error-Logging</li>
            </ul>
            
            <p><b>Optional:</b> Llama.cpp für KI-Unterstützung</p>
            """
        )
    
    def closeEvent(self, event):
        """Beim Schließen"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()


# ========== MAIN ==========

def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("JobRouter Checker V5.0")
    app.setOrganizationName("JobRouter Tools")
    
    window = JobRouterWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
