#!/usr/bin/env python3
"""
stage1_parquet_to_filtered_json.py
Stage-1 pipeline : read tokenized Parquet OR decoded JSON, apply enhanced + auxiliary heuristics,
filter to categories, write JSONL (text only), and log scores/statistics to a log file.
"""

import argparse, glob, os, re, json, random, datetime
from typing import List, Dict, Any, Optional, Iterable, Set, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------
# Patterns (unchanged + added)
# -----------------------------
RE_DIALOG_TURNS = re.compile(r'(?im)^(?:user|assistant|system|human|bot|ai|q|a|question|answer|prompt|response|narrator|interviewer|interviewee)\s*[:>\-]')
RE_BLOCK_HEADERS = re.compile(r'(?im)^###\s*(instruction|input|output|assistant|user|response|system)\b')
RE_XML_TOKENS = re.compile(r'(?i)<\|(?:system|assistant|user)\|>')

RE_ROLEPLAY = re.compile(r'(?i)\b(role[-\s]?play|act as|stay in character|in character|you are now|pretend to be|as a (?:wizard|doctor|lawyer|teacher|dungeon master|dm))\b')
RE_GAMEY = re.compile(r'(?i)\b(dungeon master|dm:|campaign|npc|quest|inventory|initiative|roll a d\d+)\b')

RE_REASONING = re.compile(r"""
(?ix)
\b(
   step[-\s]?by[-\s]?step
 | let['’]s\s+(?:think|reason|break\s+(?:it|this)\s+down|analy[sz]e)
 | (?:our\s+)?(?:approach|strategy|plan|analysis|reasoning|solution(?:\s+outline)?)
 | therefore | hence | thus | it\s+follows | we\s+have | we\s+can\s+see
 | assume | suppose | given\s+that | consider
 | by\s+(?:contradiction|induction) | contradiction | counterexample
 | base\s+case | inductive\s+step | w\.?l\.?o\.?g\.?
 | q\.?e\.?d\.? | ∴ | ⇒ | →
)\b
""", re.VERBOSE)

RE_MATHY = re.compile(r'[\=\+\-\*/\^]\s*\d|\bO\([\w^]+\)|\bETA\b')

RE_RAG = re.compile(r'(?i)\b(reference[s]?|citation[s]?|doc(?:ument)?|retriev(?:al|ed)|knowledge base|kb|source[s]?|context[: ]|chunk[: ]|page\s+\d+|url[: ]|doi[: ])\b')
RE_RAG_BRACKETS = re.compile(r'\[(?:\d{1,3}|[A-Za-z]\w{0,10})\]|\((?:\d{1,3})\)')

# ChatRAG strict signals (high-precision)
RE_CHAT_JSONROLES = re.compile(r'(?i)"role"\s*:\s*"(?:user|assistant|system|agent)"')
RE_MULTI_USERS    = re.compile(r'(?i)"role"\s*:\s*"user"')
RE_CONTEXT_HEAD   = re.compile(r'(?im)^(?:context|document|source|sources|references|evidence|facts|knowledge base|kb|retrieved)\s*:\s', re.MULTILINE)
RE_URL            = re.compile(r'https?://\S+')
RE_DOI            = re.compile(r'(?i)\b(?:doi:|10\.\d{4,9}/\S+)\b')
RE_PAGINATION     = re.compile(r'(?i)\bpage\s+\d+\b|\bchunk\s*:\s*\d+\b|\bsection\s*:\s*\w+')
RE_AUTHOR_YEAR    = re.compile(r'\([A-Z][A-Za-z]+,\s*(19|20)\d{2}[a-z]?\)')


#RE_FUNC_JSON = re.compile(r'(?s)"name"\s*:\s*"[A-Za-z0-9_\-]+"\s*,\s*"arguments"\s*:\s*(?:\{.*?\}|(?:"[^"]*"|\'[^\']*\'))')
RE_FUNC_JSON = re.compile(
    r'''(?is)
    ["']name["']\s*:\s*["']([A-Za-z0-9_\-]+)["']\s*,\s*
    ["']arguments["']\s*:\s*
       (?:\{.*?\} | (["'])(?:\\.|(?!\1).)*\1)
    ''',
    re.VERBOSE
)

#RE_FUNC_TAG    = re.compile(r'(?i)<\s*functioncall\b')
#RE_FUNC_RESULT = re.compile(r'(?im)^\s*(Function|assistant_tool_result|tool[_\s-]?result)\s*:\s*\{\s*.*\}')


# A more inclusive regex for the call itself
RE_FUNC_TAG = re.compile(
    r"""(?ix) # Case-insensitive and verbose flags
    # XML-style tags
    <\s*(?:function_?call|tool_?call|invoke)\b
    |
    # JSON-style keys (often followed by a colon and brackets/braces)
    "tool_calls"\s*:\s*\[
    """
)

# A slightly more robust version that just finds the start
# The idea is to find the marker and then process the following lines
# as potential JSON in your script.
RE_FUNC_RESULT = re.compile(
    r"""(?im) # Case-insensitive and multi-line
    ^s* # Start of a line with optional whitespace
    (?:
        Function\sResult |
        assistant_tool_result |
        tool[_\s-]?result |
        tool\soutput
    )
    \s*:\s* # Colon separator
    """
)



RE_FUNC_BLOCKS = re.compile(r'(?is)\b(function_call|tool_calls?|tool_call_id|assistant_tool_result|tool_result)\b')
RE_TOOL_OBJ = re.compile(r'(?is)"tool"\s*:\s*\{[^{}]*"function"\s*:\s*\{[^{}]*\}[^{}]*\}')
RE_JSON_SCHEMA = re.compile(r'(?is)"type"\s*:\s*"object"\s*,\s*"properties"\s*:\s*\{')
RE_CODEBLOCK = re.compile(r'```[a-zA-Z0-9_+\-]*\n.*?\n```', re.DOTALL)

# Structured QA: Input: Question ... (Output|Answer|Solution|Response):
RE_QA_IO = re.compile(r'(?is)\binput\s*:\s*question\b.*?\b(?:output|answer|final\s*answer|solution|response)\s*:', re.DOTALL)

# Stepwise / headings / proofy cues
RE_REASON_HEAD = re.compile(r'(?im)^(analysis|approach|reasoning|solution|proof)\s*:', re.MULTILINE)
RE_ENUM_STEPS  = re.compile(r'(?im)^\s*(?:step\s*\d+|[0-9]{1,2}[.)])\s')
RE_PROOFY      = re.compile(r'(?i)\b(by\s+(?:contradiction|induction)|contradiction|base\s+case|inductive\s+step|q\.?e\.?d\.?|thus|therefore|hence|it\s+follows)\b')
RE_LETS_THINK  = re.compile(r"(?i)\blet['’]s\s+(?:think|reason|break\s+(?:it|this)\s+down|analy[sz]e)\b")

# Numbers / equations / LaTeX-ish
RE_NUMBER      = re.compile(r'\b\d+\b')
RE_EQUATION    = re.compile(r'(?:=|⇒|→|<=|>=|<|>|\bO\([\w^]+\)|\\frac\{|\\begin\{align\}|\\\(|\\\)|\$)')

# Structural reasoning cues (optional post-filter; Option B)
RE_REASONING_STRUCTURAL = re.compile(
    r'(?is)('
    r'Input\s*:\s*Question\s*:.*?(?:Answer|Output)\s*:'
    r'|^\s*(Solution|Proof)\s*:'
    r'|\bStep\s*\d\b'
    r'|='
    r'|⇒'
    r'|→'
    r'|\\frac'
    r'|\\begin\{align\}'
    r')'
)

# Comparative / word-problem
RE_COMPARATIVE = re.compile(
    r'(?i)(?:\b(who|which)\s+(?:has|is)\s+(?:more|less|greater|smaller|bigger|fewer)\b)'
    r'|(?:\b(?:more|less|greater|smaller|bigger|fewer)\s+than\b)'
    r'|(?:\bmax(?:imum)?\b|\bmin(?:imum)?\b)'
)
RE_WORDPROB    = re.compile(r'(?i)\bhow many\b|\baltogether\b|\bin total\b|\bin all\b|\bnow\?\b|\bleft\b')

# -----------------------------
# Auxiliary detectors
# -----------------------------

def has_fc_struct(text: str) -> bool:
    t = text
    return bool(
        RE_FUNC_JSON.search(t) or
        RE_TOOL_OBJ.search(t) or
        RE_FUNC_BLOCKS.search(t) or
        RE_FUNC_TAG.search(t) or
        re.search(r'(?i)"function_call"|"tool_calls"', t)
    )

def has_fc_result(text: str) -> bool:
    """Detect a tool result block."""
    return bool(RE_FUNC_RESULT.search(text))


def is_function_calling_extra(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in [
        '"function_call"', '"tool_calls"', '"function"', '"parameters"',
        'glaive-function-calling', '<functioncall>'
    ]): return True
    if re.search(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', t): return True
    return False

def is_reasoning_extra(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in [
        'step-by-step', "let's think step by step", 'logical reasoning',
        'deduction', 'inference', 'problem solving', 'if...then', 'conclusion is',
        'puzzle', 'riddle', 'logic puzzle', 'explain why', 'how would you', 'what is the logic'
    ]): return True
    if re.search(r'[Qq]:.*\n[Aa]:', text) and len(text) > 200: return True
    return False

def is_roleplay_extra(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in [
        'roleplay', 'role-play', 'you are a character', 'act as', 'pretend to be',
        'your name is', 'persona:'
    ]): return True
    if re.search(r'\n[A-Z][a-zA-Z]+:', text): return True
    if re.search(r'in a world where|imagine you are|you are tasked with', t): return True
    return False

def is_chatrag_extra(text: str) -> bool:
    has_user = re.search(r'"role"\s*:\s*"user"', text, re.IGNORECASE)
    has_assistant = re.search(r'"role"\s*:\s*"(assistant|agent)"', text, re.IGNORECASE)
    if has_user and has_assistant:
        if re.search(r'context:|document:|passage:|retrieved snippet', text, re.IGNORECASE): return True
        if len(re.findall(r'"role"\s*:\s*"user"', text, re.IGNORECASE)) > 1: return True
    return False

# -----------------------------
# English detection
# -----------------------------
STOPWORDS = set("""the of and to in is that for on with as are it this be or by from at an which you your have has was were can not but they their more one about will if into our use using used when how what who where why while during should could would may might also we i he she them those these there here then than""".split())
def is_likely_english(text: str) -> bool:
    try:
        from langdetect import detect
        if detect(text[:1000]) == "en":
            return True
    except Exception:
        pass
    if not text: return False
    total = len(text); ascii_count = sum(1 for ch in text if ord(ch) < 128)
    ascii_ratio = ascii_count / max(1, total)
    words = re.findall(r"[A-Za-z']+", text.lower())
    sw_hits = sum(1 for w in words if w in STOPWORDS)
    non_ascii_ratio = 1.0 - ascii_ratio
    return (ascii_ratio >= 0.85 and sw_hits / max(1, len(words)) >= 0.05 and non_ascii_ratio <= 0.15)

# -----------------------------
# Scoring + boosters
# -----------------------------
def score_categories(text: str) -> Dict[str, float]:
    t = text.strip()
    if not t: return {"reasoning":0, "chatrag":0, "roleplay":0, "function_calling":0}

    dlg = 1.0 if (RE_DIALOG_TURNS.search(t) or RE_BLOCK_HEADERS.search(t) or RE_XML_TOKENS.search(t)) else 0.0
    role = 0.6 if (RE_ROLEPLAY.search(t) or RE_GAMEY.search(t)) else 0.0
    role = min(1.0, role + 0.3*dlg)

    rs = 0.5 if RE_REASONING.search(t) else 0.0
    rs += 0.15 if RE_REASON_HEAD.search(t) else 0.0
    rs += 0.15 if RE_ENUM_STEPS.search(t) else 0.0
    rs += 0.2  if RE_MATHY.search(t) else 0.0
    rs += 0.1  if RE_CODEBLOCK.search(t) else 0.0

    rag = 0.5 if (RE_RAG.search(t) or RE_RAG_BRACKETS.search(t)) else 0.0
    rag += 0.2 if len(t) > 400 and (t.count("\n") > 3) else 0.0
    rag += 0.1 if "http" in t or "doi.org" in t else 0.0

    fc = 0.0
    if RE_FUNC_JSON.search(t) or RE_TOOL_OBJ.search(t): fc += 0.6
    if RE_FUNC_BLOCKS.search(t) or RE_JSON_SCHEMA.search(t): fc += 0.2
    if RE_CODEBLOCK.search(t): fc += 0.2

    return {
        "reasoning": min(1.0, rs),
        "chatrag": min(1.0, rag),
        "roleplay": role,
        "function_calling": min(1.0, fc),
    }

def reasoning_boost(text: str) -> float:
    t = text
    #print("boost",RE_QA_IO.search(t))
    if RE_QA_IO.search(t): return 1.8
    boost = 0.0
    if len(RE_NUMBER.findall(t)) >= 2 and RE_COMPARATIVE.search(t): boost = max(boost, 0.25)
    if len(RE_NUMBER.findall(t)) >= 2 and RE_WORDPROB.search(t):   boost = max(boost, 0.25)
    try:
        if is_reasoning_extra(t): boost = max(boost, 0.25)
    except NameError:
        pass
    return boost

def reasoning_signal_count(text: str) -> int:
    t = text; signals = 0
    if RE_QA_IO.search(t):       signals += 2
    if RE_REASON_HEAD.search(t): signals += 1
    if RE_ENUM_STEPS.search(t):  signals += 1
    if RE_PROOFY.search(t):      signals += 1
    if RE_LETS_THINK.search(t):  signals += 1
    if (len(RE_NUMBER.findall(t)) >= 2) or bool(RE_EQUATION.search(t)): signals += 1
    if RE_COMPARATIVE.search(t) or RE_WORDPROB.search(t): signals += 1

    #print("signals=", signals)
    return signals

def passes_reasoning_strict(text: str, min_words: int, require_k: int) -> bool:

    word_count = len(text.split())

    if word_count < min_words: return False
    if word_count > 200: return False
    return reasoning_signal_count(text) >= max(1, require_k)


def chatrag_signal_count(text: str) -> int:
    """Count distinct high-precision ChatRAG signals present in the text."""
    t = text
    signals = 0
    # Chat structure (any)
    if RE_CHAT_JSONROLES.search(t) or RE_DIALOG_TURNS.search(t) or RE_XML_TOKENS.search(t) or RE_BLOCK_HEADERS.search(t):
        signals += 1
    # Explicit context / RAG keywords
    if RE_CONTEXT_HEAD.search(t) or RE_RAG.search(t):
        signals += 1
    # Citations / doc metadata
    if RE_RAG_BRACKETS.search(t) or RE_AUTHOR_YEAR.search(t) or RE_PAGINATION.search(t):
        signals += 1
    # URLs / DOIs
    if RE_URL.search(t) or RE_DOI.search(t):
        signals += 1
    # Multiple user turns in JSON chat logs
    try:
        if len(RE_MULTI_USERS.findall(t)) > 1:
            signals += 1
    except Exception:
        pass
    return signals

def passes_chatrag_strict(text: str, min_words: int, require_k: int) -> bool:
    word_count = len(text.split())
    if word_count < min_words:
        return False
    return chatrag_signal_count(text) >= max(1, require_k)
# -----------------------------
# Keep decision (single place)
# -----------------------------

def decide_keep(
    scores: Dict[str,float],
    text: str,
    categories: List[str],
    global_threshold: float,
    reasoning_threshold: Optional[float],
    reasoning_strict: bool,
    reasoning_min_words: int,
    reasoning_require_k: int,
    # ChatRAG strict knobs
    chatrag_threshold: Optional[float] = None,
    chatrag_strict: bool = False,
    chatrag_min_words: int = 50,
    chatrag_require_k: int = 2,
    # Roleplay and function_calling thresholds
    roleplay_threshold: Optional[float] = None,
    function_calling_threshold: Optional[float] = None,
) -> Tuple[bool, List[str], List[str]]:
    """Return (keep, labels_kept, rejection_reasons)."""
    labels: List[str] = []
    rej: List[str] = []

    r_thr = reasoning_threshold if reasoning_threshold is not None else global_threshold
    c_thr = chatrag_threshold if chatrag_threshold is not None else global_threshold
    rp_thr = roleplay_threshold if roleplay_threshold is not None else global_threshold
    fc_thr = function_calling_threshold if function_calling_threshold is not None else global_threshold

    # Reasoning (special rules)
    if "reasoning" in categories:
        r_score = scores.get("reasoning", 0.0)
        
        if (r_score >= r_thr) and ( not reasoning_strict or passes_reasoning_strict(text, reasoning_min_words, reasoning_require_k))     :
            labels.append("reasoning")
        else:
            rej.append(f"reasoning_score_{r_score:.3f}")

    # ChatRAG (special rules)
    if "chatrag" in categories:
        c_score = scores.get("chatrag", 0.0)
        if c_score >= c_thr and (not chatrag_strict or passes_chatrag_strict(text, chatrag_min_words, chatrag_require_k)):
            labels.append("chatrag")
        else:
            rej.append(f"chatrag_score_{c_score:.3f}")


    # FC (special rules)
    fc_relaxed = True
    fc_require_result = False
    fc_min_words = 10

    if "function_calling" in categories:
        f_score = scores.get("function_calling", 0.0)
        f_thr = fc_thr #fc_threshold if fc_threshold is not None else global_threshold
        ok = False
        if f_score >= f_thr:
            ok = True
        elif fc_relaxed and has_fc_struct(text):
            ok = True
        if ok:
            if fc_require_result and not has_fc_result(text):
                rej.append("function_calling_missing_result")
            else:
                if fc_relaxed:
                    if len(text.split()) >= fc_min_words:
                        labels.append("function_calling")
                    else:
                        rej.append("function_calling_too_short")
                else:
                    labels.append("function_calling")
        else:
            rej.append(f"function_calling_score_{f_score:.3f}")

    # Roleplay (separate threshold)
    if "roleplay" in categories:
        rp_score = scores.get("roleplay", 0.0)
        if rp_score >= rp_thr:
            labels.append("roleplay")
        else:
            rej.append(f"roleplay_score_{rp_score:.3f}")
    

    return (len(labels) > 0), labels, rej

# -----------------------------
# Multiprocessing helpers
# -----------------------------
def get_optimal_workers():
    n = multiprocessing.cpu_count()
    if n <= 2: return 1
    if n <= 4: return n - 1
    if n <= 8: return n - 2
    return 8

def process_chunk_parquet(chunk):
    tokenizer      = chunk['tokenizer']  # Pass the actual tokenizer object
    token_column   = chunk['token_column']
    min_words      = chunk['min_words']
    max_words      = chunk['max_words']
    max_tokens     = chunk['max_tokens']  # Add token limit
    english_only   = chunk['english_only']
    categories     = chunk['categories']
    thr            = chunk['threshold']
    no_legacy      = chunk['no_legacy_boost']
    r_thr          = chunk['reasoning_threshold']
    r_strict       = chunk['reasoning_strict']
    r_min_words    = chunk['reasoning_min_words']
    r_require_k    = chunk['reasoning_require_k']
    examples       = chunk['examples']
    debug_print    = chunk.get('debug_print', False)
    
    # Now we can use debug_print parameter
    if debug_print:
        print(f"DEBUG PARQUET: Processing chunk with {len(examples)} examples, token_column='{token_column}'")
        print(f"DEBUG PARQUET: Will process examples with keys: {list(examples[0].keys()) if examples else 'no examples'}")

    # Add debug print function for this specific function
    def debug_return():
        print(f"DEBUG PARQUET: Returning {len(kept)} kept, {len(rejected)} rejected")
    
    # Unique identifier for this function
    FUNCTION_TYPE = "PARQUET"
    # PARQUET_FUNCTION_DEBUG_MARKER

    kept, rejected = [], []
    # PARQUET_DEBUG_ADDED
    for ex in examples:
        # PARQUET_LOOP_START
        if token_column not in ex: continue
        ids = ex[token_column]
        
        # Check token count before decoding (hybrid approach)
        token_count = len(ids)
        if token_count > max_tokens:
            # Skip decoding entirely for very long texts
            rejected.append({'text': '', 'rejection_reason': f'token_count_{token_count}', 'scores': {}, 'length': 0, 'threshold': thr})
            continue
        
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if not text: continue

        L = len(text.split())
        if L < min_words or L > max_words: 
            rejected.append({'text': text, 'rejection_reason': f'len_{L}', 'scores': {}, 'length': L, 'threshold': thr})
            continue

        if english_only and not is_likely_english(text):
            rejected.append({'text': text, 'rejection_reason': 'non_english', 'scores': {}, 'length': L, 'threshold': thr})
            continue

        scores = score_categories(text)
        if not no_legacy:
            rb = reasoning_boost(text)
            if rb > 0: scores["reasoning"] = min(1.0, scores["reasoning"] + rb)
            if is_chatrag_extra(text):          scores["chatrag"]   = min(1.0, scores["chatrag"]   + 0.25)
            if is_roleplay_extra(text):         scores["roleplay"]  = min(1.0, scores["roleplay"]  + 0.25)

        if is_function_calling_extra(text): scores["function_calling"] = min(1.0, scores["function_calling"] + 0.35)    

        keep, labels, rej = decide_keep(scores, text, categories, thr, r_thr, r_strict, r_min_words, r_require_k, chatrag_threshold=chunk.get('chatrag_threshold'), chatrag_strict=chunk.get('chatrag_strict', False), chatrag_min_words=chunk.get('chatrag_min_words', 50), chatrag_require_k=chunk.get('chatrag_require_k', 2), roleplay_threshold=chunk.get('roleplay_threshold'), function_calling_threshold=chunk.get('function_calling_threshold'))
        if keep:
            if  ('reasoning' in labels):
                structural_ok = bool(RE_REASONING_STRUCTURAL.search(text))
                if structural_ok:
                    #reasoning_structural_pass += 1
                    kept.append({'text': text, 'labels': labels, 'scores': scores, 'length': L})
                else:
                    #reasoning_structural_fail += 1
                    rejected.append({'text': text, 'rejection_reason': 'reasoning_structural_miss', 'scores': scores, 'length': L, 'threshold': thr})
            #kept.append({'text': text, 'labels': labels, 'scores': scores, 'length': L})

            
        else:
            rejected.append({'text': text, 'rejection_reason': '|'.join(rej), 'scores': scores, 'length': L, 'threshold': thr})
    return {'kept': kept, 'rejected': rejected}

def process_chunk_json(chunk):
    text_column    = chunk['text_column']
    min_words      = chunk['min_words']
    max_words      = chunk['max_words']
    english_only   = chunk['english_only']
    categories     = chunk['categories']
    thr            = chunk['threshold']
    no_legacy      = chunk['no_legacy_boost']
    r_thr          = chunk['reasoning_threshold']
    r_strict       = chunk['reasoning_strict']
    r_min_words    = chunk['reasoning_min_words']
    r_require_k    = chunk['reasoning_require_k']
    examples       = chunk['examples']
    debug_print    = chunk.get('debug_print', False)
    
    # Now we can use debug_print parameter
    if debug_print:
        print(f"DEBUG JSON: Processing chunk with {len(examples)} examples, text_column='{text_column}'")
        print(f"DEBUG JSON: Will process examples with keys: {list(examples[0].keys()) if examples else 'no examples'}")

    # Unique identifier for this function
    FUNCTION_TYPE = "JSON"
    # JSON_FUNCTION_DEBUG_MARKER
    
    kept, rejected = [], []
    # JSON_DEBUG_ADDED
    for ex in examples:
        # JSON_LOOP_START
        if text_column in ex:
            text = ex[text_column]
        else:
            text = ex  # fallback (already-a-string cases)

        # robust extraction
        if not isinstance(text, str):
            if isinstance(text, dict):
                if 'instances' in text and isinstance(text['instances'], dict):
                    parts = []
                    if isinstance(text['instances'].get('text'), str):   parts.append(text['instances']['text'])
                    if isinstance(text['instances'].get('output'), str): parts.append(text['instances']['output'])
                    text = '\n\n'.join(parts) if parts else ''
                elif text_column in text and isinstance(text[text_column], str):
                    text = text[text_column]
                else:
                    str_vals = [v for v in text.values() if isinstance(v, str)]
                    text = str_vals[0] if str_vals else ''
            else:
                text = str(text)

        if not text: 
            rejected.append({'text': '', 'rejection_reason': 'no_text', 'scores': {}, 'length': 0, 'threshold': thr})
            continue

        L = len(text.split())
        if L < min_words or L > max_words: 
            rejected.append({'text': text, 'rejection_reason': f'len_{L}', 'scores': {}, 'length': L, 'threshold': thr})
            continue

        if english_only and not is_likely_english(text):
            rejected.append({'text': text, 'rejection_reason': 'non_english', 'scores': {}, 'length': L, 'threshold': thr})
            continue

        scores = score_categories(text)
        if not no_legacy:
            rb = reasoning_boost(text)
            if rb > 0: scores["reasoning"] = min(1.0, scores["reasoning"] + rb)
            if is_chatrag_extra(text):          scores["chatrag"]   = min(1.0, scores["chatrag"]   + 0.25)
            if is_roleplay_extra(text):         scores["roleplay"]  = min(1.0, scores["roleplay"]  + 0.25)
        
        if is_function_calling_extra(text): scores["function_calling"] = min(1.0, scores["function_calling"] + 0.25)

        keep, labels, rej = decide_keep(scores, text, categories, thr, r_thr, r_strict, r_min_words, r_require_k, chatrag_threshold=chunk.get('chatrag_threshold'), chatrag_strict=chunk.get('chatrag_strict', False), chatrag_min_words=chunk.get('chatrag_min_words', 50), chatrag_require_k=chunk.get('chatrag_require_k', 2), roleplay_threshold=chunk.get('roleplay_threshold'), function_calling_threshold=chunk.get('function_calling_threshold'))
        if keep:
            kept.append({'text': text, 'labels': labels, 'scores': scores, 'length': L})
        else:
            rejected.append({'text': text, 'rejection_reason': '|'.join(rej), 'scores': scores, 'length': L, 'threshold': thr})
    return {'kept': kept, 'rejected': rejected}

# -----------------------------
# Debug utilities
# -----------------------------
def debug_print(args, message):
    """Print debug message only if --debug-print flag is set"""
    if args.debug_print:
        print(message)

# -----------------------------
# IO / CLI
# -----------------------------
def parse_inputs(args) -> List[str]:
    files: List[str] = []
    files += list(args.inputs or [])
    if args.glob:
        files += glob.glob(args.glob)
    if args.dir:
        if args.input_type == "parquet":
            files += glob.glob(os.path.join(args.dir, "*.parquet"))
            files += glob.glob(os.path.join(args.dir, "**/*.parquet"), recursive=True)
        else:
            files += glob.glob(os.path.join(args.dir, "*.json"))
            files += glob.glob(os.path.join(args.dir, "**/*.json"), recursive=True)
            files += glob.glob(os.path.join(args.dir, "**/*.jsonl"), recursive=True)
    return sorted(set(files))

def open_log(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="Input files")
    ap.add_argument("--glob", help="Glob for input files")
    ap.add_argument("--dir", help="Directory to scan (recursive)")
    ap.add_argument("--input-type", choices=["parquet", "json"], default="parquet")

    # parquet controls
    ap.add_argument("--tokenizer-name", default="gpt2")
    ap.add_argument("--token-column",  default="tokens")

    # json controls
    ap.add_argument("--text-column", default="text")

    # outputs
    ap.add_argument("--text_output_key", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--include-labels", action="store_true")

    # filtering
    ap.add_argument("--categories", nargs="*", default=["reasoning","chatrag","roleplay","function_calling"])
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--english-only", default=True)
    ap.add_argument("--dedup-exact", default=True)
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--no-legacy-boost", default=True)

     # reasoning strict knobs
    ap.add_argument("--reasoning-threshold", type=float, default=0)
    ap.add_argument("--reasoning-strict", default=True)
    ap.add_argument("--reasoning-min-words", type=int, default=12,
                    help="Minimum word count for reasoning texts when using strict mode")
    ap.add_argument("--reasoning-require-k-signals", type=int, default=3)
    ap.add_argument("--reasoning-strict-out", default="")   

    # roleplay threshold
    ap.add_argument("--roleplay-threshold", type=float, default=0,
                    help="Score threshold for roleplay (overrides --threshold for roleplay only)")
    
    # function_calling threshold
    ap.add_argument("--function-calling-threshold", type=float, default=0.5,
                    help="Score threshold for function_calling (overrides --threshold for function_calling only)")

    # workers
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--debug-structure", action="store_true")
    ap.add_argument("--debug-print", action="store_true", help="Enable debug print statements")
    
    # rejected data control
    ap.add_argument("--collect-rejected", action="store_true", default=False,
                    help="Collect and write out rejected data (default: False)")
    
    # memory management
    ap.add_argument("--max-chunks-in-memory", type=int, default=None,
                    help="Maximum number of chunks to keep in memory before processing (default: auto-calculated based on workers). "
                         "Optimal: workers + 1-2 for good parallelism without excessive memory usage.")
    
    # token limit for parquet inputs (skip decoding if exceeded)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Maximum token count for parquet inputs (skip decoding if exceeded, default: 2048)")


    ap.add_argument("--chatrag-threshold", type=float, default=0.7,
                    help="Score threshold for ChatRAG (overrides --threshold for chatrag only)")
    ap.add_argument("--chatrag-strict", default=True,
                    help="Require multiple high-precision RAG signals (for ultra-clean ChatRAG)")
    ap.add_argument("--chatrag-min-words", type=int, default=50,
                    help="Minimum word count for texts to be considered ChatRAG when --chatrag-strict is on")
    ap.add_argument("--chatrag-require-k-signals", type=int, default=2,
                    help="How many distinct ChatRAG signals must be present in strict mode")
    ap.add_argument("--chatrag-strict-out", default="",
                    help="Optional extra JSONL to write only STRICT ChatRAG hits (text only)")
    ap.add_argument("--reasoning-structural-filter", action="store_true",
                    help="Enable structural post-filter for reasoning (Option B)")
    ap.add_argument("--reasoning-structural-out", default="",
                    help="Optional gold JSONL for reasoning items that pass the structural cue")
    ap.add_argument("--reasoning-structural-mode", choices=["require","prefer"], default="prefer",
                    help="require: drop non-structural reasoning; prefer: keep as usual, copy structural to gold file")



    args = ap.parse_args()

    files = parse_inputs(args)
    if not files:
        raise SystemExit("No inputs found. Use --inputs or --glob or --dir.")

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # category-specific output files
    cat_out = {}
    base = os.path.splitext(args.out)[0]
    if False:
        for c in ["reasoning","chatrag","roleplay","function_calling"]:
            if c in args.categories:
                p = f"{base}_{c}.jsonl"
                cat_out[c] = open(p, "w", encoding="utf-8")
                tqdm.write(f"Created category output: {p}")

    # rejected sink (conditional)
    rej_path = f"{base}_rejected.jsonl"
    rej_out = open(rej_path, "w", encoding="utf-8") if args.collect_rejected else None
    if args.collect_rejected:
        tqdm.write(f"Created rejected output: {rej_path}")
    else:
        tqdm.write("Rejected data collection disabled (use --collect-rejected to enable)")

    # optional strict-only sink (not currently used)
    strict_out = None

    # ChatRAG strict-only sink (optional)
    chatrag_strict_out = open(args.chatrag_strict_out, "w", encoding="utf-8") if args.chatrag_strict_out else None

    with open(args.out, "w", encoding="utf-8") as outf, open_log(args.log_file) as log:
        log.write("# Stage-1 Data Filtering Pipeline\n")
        log.write(f"datetime: {datetime.datetime.utcnow().isoformat()}Z\n")
        log.write(f"input_type: {args.input_type}\n")
        if args.input_type == "parquet":
            log.write(f"tokenizer: {args.tokenizer_name} (shared across workers)\n")
            tok_report = args.tokenizer_name
        else:
            log.write(f"text_column: {args.text_column}\n")
            tok_report = None
        log.write(f"files: {len(files)}\n")
        log.write(f"categories: {args.categories}\n")
        log.write(f"threshold: {args.threshold}\n")
        log.write(f"reasoning_threshold: {args.reasoning_threshold}\n")
        log.write(f"reasoning_strict: {args.reasoning_strict}\n")
        log.write(f"reasoning_min_words: {args.reasoning_min_words}\n")
        log.write(f"reasoning_structural_filter: {args.reasoning_structural_filter}\n")
        log.write(f"reasoning_structural_mode: {args.reasoning_structural_mode}\n")
        log.write(f"chatrag_threshold: {args.chatrag_threshold}\n")
        log.write(f"chatrag_strict: {args.chatrag_strict}\n")
        log.write(f"chatrag_min_words: {args.chatrag_min_words}\n")
        log.write(f"chatrag_require_k_signals: {args.chatrag_require_k_signals}\n")
        log.write(f"roleplay_threshold: {args.roleplay_threshold}\n")
        log.write(f"function_calling_threshold: {args.function_calling_threshold}\n")
        log.write(f"english_only: {args.english_only}\n")
        log.write(f"dedup_exact: {args.dedup_exact}\n")
        log.write(f"collect_rejected: {args.collect_rejected}\n")
        log.write(f"max_chunks_in_memory: {args.max_chunks_in_memory}\n")
        log.write(f"max_tokens: {args.max_tokens}\n")
        log.write(f"chunk_size: {args.chunk_size}\n\n")

        # Define reasoning_structural_out before using it
        reasoning_structural_out = open(args.reasoning_structural_out, "w", encoding="utf-8") if args.reasoning_structural_out else None
        
        for c, fo in cat_out.items():
            log.write(f"category_output_{c}: {fo.name}\n")

        if args.collect_rejected:
            log.write(f"rejected_output: {rej_path}\n")
        else:
            log.write("rejected_output: disabled\n")
        if reasoning_structural_out: log.write(f"reasoning_structural_output: {reasoning_structural_out.name}\n")

        if chatrag_strict_out: log.write(f"chatrag_strict_output: {chatrag_strict_out.name}\n")
        log.write("\n")

        # tokenizer created once in main process to avoid multiple HF downloads in workers
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name) if args.input_type == "parquet" else None

        # workers - unified approach always uses multiprocessing
        num_workers = get_optimal_workers() if args.workers is None else args.workers
        
        # Auto-calculate optimal chunks in memory based on worker count
        if args.max_chunks_in_memory is None:
            # Optimal: workers + 2 for buffering, but cap at reasonable memory usage
            #if num_workers <= 4:
            #    auto_chunks = num_workers + 2  # Small worker count: allow full buffering
            #elif num_workers <= 16:
           #     auto_chunks = min(num_workers + 1, 12)  # Medium: moderate buffering, cap at 12
           # else:
           #     auto_chunks = min(num_workers, 16)  # Large: minimal buffering, cap at 16

            auto_chunks = 2 * num_workers

            args.max_chunks_in_memory = auto_chunks
            print(f"Using {num_workers} worker processes (unified multiprocessing); chunk size {args.chunk_size}")
            print(f"Auto-calculated: max {args.max_chunks_in_memory} chunks in memory")
        else:
            print(f"Using {num_workers} worker processes (unified multiprocessing); chunk size {args.chunk_size}")
            print(f"Manual setting: max {args.max_chunks_in_memory} chunks in memory")
        
        if args.input_type == "parquet":
            print(f"Tokenizer '{args.tokenizer_name}' created once and shared across all workers")
            print(f"Token limit: {args.max_tokens} tokens (skip decoding if exceeded)")
        
        # Validate chunk buffer vs workers relationship
        if args.max_chunks_in_memory < num_workers:
            print(f"Warning: Only {args.max_chunks_in_memory} chunks in memory for {num_workers} workers")
            print(f"   This may cause workers to wait. Consider increasing --max-chunks-in-memory to {num_workers + 2}")
        elif args.max_chunks_in_memory > num_workers * 3:
            print(f"Warning: {args.max_chunks_in_memory} chunks in memory for {num_workers} workers")
            print(f"   This may use more memory than necessary. Consider reducing to {num_workers + 2}")
        else:
            print(f"Optimal chunk buffer: {args.max_chunks_in_memory} chunks for {num_workers} workers")
        
        # Estimate memory usage
        estimated_memory_mb = (args.max_chunks_in_memory * args.chunk_size * 2)  # Rough estimate: 2KB per example
        print(f"Estimated memory usage: ~{estimated_memory_mb} MB (varies by text length)")

        # stats
        total_rows = decoded_ok = too_short = too_long = non_english = duplicates = kept = 0
        too_many_tokens = 0  # Track token count rejections
        kept_by_class = {c:0 for c in ["reasoning","chatrag","roleplay","function_calling"]}
        score_hist = {c:[] for c in kept_by_class.keys()}
        #reasoning_structural_pass = reasoning_structural_fail = 0
        seen_texts: Set[str] = set()

        with tqdm(files, desc="Processing files", unit="file") as file_pbar:
            for fpath in file_pbar:
                file_pbar.set_postfix_str(os.path.basename(fpath))
                if not os.path.exists(fpath):
                    tqdm.write(f"Missing file: {fpath}")
                    continue

                try:
                    ds = load_dataset('parquet' if args.input_type=="parquet" else 'json',
                                      data_files={'train': fpath},
                                      streaming=True)['train']
                except Exception as e:
                    tqdm.write(f"Error loading {args.input_type} file: {e}")
                    continue

                # Use streaming to avoid loading entire file into memory
                total_examples = 0
                chunk_buffer = []
                chunks = []
                
                for ex in ds:
                    total_examples += 1
                    chunk_buffer.append(ex)
                    
                    # When buffer reaches chunk size, create a chunk
                    if len(chunk_buffer) >= args.chunk_size:
                        ex_chunk = chunk_buffer
                        chunk_buffer = []
                        debug_print(args, f"DEBUG: Created chunk {len(chunks)+1} with {len(ex_chunk)} examples")
                        
                        if args.input_type == "parquet":
                            chunks.append({
                                'examples': ex_chunk,
                                'tokenizer': tokenizer,  # Pass the actual tokenizer object
                                'token_column':  args.token_column,
                                'min_words': args.min_words,
                                'max_words': args.max_words,
                                'max_tokens': args.max_tokens,  # Add token limit
                                'english_only': args.english_only,
                                'categories': args.categories,
                                'threshold': args.threshold,
                                'no_legacy_boost': args.no_legacy_boost,
                                'reasoning_threshold': args.reasoning_threshold,
                                'reasoning_strict': args.reasoning_strict,
                                'reasoning_min_words': args.reasoning_min_words,
                                'reasoning_require_k': args.reasoning_require_k_signals,
                                'chatrag_threshold': args.chatrag_threshold,
                                'chatrag_strict': args.chatrag_strict,
                                'chatrag_min_words': args.chatrag_min_words,
                                'chatrag_require_k': args.chatrag_require_k_signals,
                                'roleplay_threshold': args.roleplay_threshold,
                                'function_calling_threshold': args.function_calling_threshold,
                                'debug_print': args.debug_print,
                            })
                        else:
                            chunks.append({
                                'examples': ex_chunk,
                                'text_column': args.text_column,
                                'min_words': args.min_words,
                                'max_words': args.max_words,
                                'english_only': args.english_only,
                                'categories': args.categories,
                                'threshold': args.threshold,
                                'no_legacy_boost': args.no_legacy_boost,
                                'reasoning_threshold': args.reasoning_threshold,
                                'reasoning_strict': args.reasoning_strict,
                                'reasoning_min_words': args.reasoning_min_words,
                                'reasoning_require_k': args.reasoning_require_k_signals,
                                'chatrag_threshold': args.chatrag_threshold,
                                'chatrag_strict': args.chatrag_strict,
                                'chatrag_min_words': args.chatrag_min_words,
                                'chatrag_require_k': args.chatrag_require_k_signals,
                                'roleplay_threshold': args.roleplay_threshold,
                                'function_calling_threshold': args.function_calling_threshold,
                                'debug_print': args.debug_print,
                            })
                        
                        # Process chunks immediately to free memory
                        if len(chunks) >= args.max_chunks_in_memory:  # Keep at most max_chunks_in_memory chunks in memory
                            debug_print(args, f"DEBUG: Processing {len(chunks)} chunks with {num_workers} workers")
                            debug_print(args, f"DEBUG: Using {'process_chunk_parquet' if args.input_type=='parquet' else 'process_chunk_json'} function")
                            
                            with ProcessPoolExecutor(max_workers=num_workers) as ex_pool:
                                futs = [ex_pool.submit(process_chunk_parquet if args.input_type=="parquet" else process_chunk_json, ch) for ch in chunks]
                                debug_print(args, f"DEBUG: Submitted {len(futs)} futures for processing")
                                for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing chunks", unit="chunk"):
                                    res = fut.result()
                                    debug_print(args, f"DEBUG: Chunk returned {len(res['kept'])} kept, {len(res['rejected'])} rejected")
                                    
                                    # Process results immediately
                                    for r in res['kept']:
                                        text = r['text']
                                        norm = " ".join(text.split())
                                        if args.dedup_exact and norm in seen_texts:
                                            duplicates += 1
                                            continue
                                        if args.dedup_exact: seen_texts.add(norm)

                                        decoded_ok += 1
                                        for c in score_hist: score_hist[c].append(r['scores'][c])
                                        for c in r['labels']: kept_by_class[c] += 1
                                        kept += 1

                                        rec = {args.text_output_key: text}
                                        if args.include_labels: rec["labels"] = r['labels']
                                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                       
                                        if False:
                                            for c in r['labels']:
                                                if c in cat_out:
                                                    rec_cat = {args.text_output_key: text}
                                                    if args.include_labels: rec_cat["labels"] = [c]
                                                    cat_out[c].write(json.dumps(rec_cat, ensure_ascii=False) + "\n")
                                                    if chatrag_strict_out and c=="chatrag" and args.chatrag_strict:
                                                        chatrag_strict_out.write(json.dumps({args.text_output_key: text}, ensure_ascii=False) + "\n")
                                    
                                    for r in res['rejected']:
                                        if args.collect_rejected:
                                            rej_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                                        decoded_ok += 1  # Count rejected samples as decoded too
                                        
                                        # Analyze rejection reasons for statistics
                                        if args.input_type == "parquet" and 'rejection_reason' in r:
                                            reason = r['rejection_reason']
                                            if reason.startswith('token_count_'):
                                                too_many_tokens += 1
                                    
                                    total_rows += (len(res['kept']) + len(res['rejected']))
                                    debug_print(args, f"DEBUG: Updated total_rows to {total_rows}, decoded_ok to {decoded_ok}")
                            
                            # Clear processed chunks to free memory
                            chunks = []
                
                # Process remaining examples in buffer
                if chunk_buffer:
                    ex_chunk = chunk_buffer
                    debug_print(args, f"DEBUG: Created final chunk with {len(ex_chunk)} examples")
                    
                    if args.input_type == "parquet":
                        chunks.append({
                            'examples': ex_chunk,
                            'tokenizer': tokenizer,
                            'token_column':  args.token_column,
                            'min_words': args.min_words,
                            'max_words': args.max_words,
                            'max_tokens': args.max_tokens,  # Add token limit
                            'english_only': args.english_only,
                            'categories': args.categories,
                            'threshold': args.threshold,
                            'no_legacy_boost': args.no_legacy_boost,
                            'reasoning_threshold': args.reasoning_threshold,
                            'reasoning_strict': args.reasoning_strict,
                            'reasoning_min_words': args.reasoning_min_words,
                            'reasoning_require_k': args.reasoning_require_k_signals,
                            'chatrag_threshold': args.chatrag_threshold,
                            'chatrag_strict': args.chatrag_strict,
                            'chatrag_min_words': args.chatrag_min_words,
                            'chatrag_require_k': args.chatrag_require_k_signals,
                            'roleplay_threshold': args.roleplay_threshold,
                            'function_calling_threshold': args.function_calling_threshold,
                            'debug_print': args.debug_print,
                        })
                    else:
                        chunks.append({
                            'examples': ex_chunk,
                            'text_column': args.text_column,
                            'min_words': args.min_words,
                            'max_words': args.max_words,
                            'english_only': args.english_only,
                            'categories': args.categories,
                            'threshold': args.threshold,
                            'no_legacy_boost': args.no_legacy_boost,
                            'reasoning_threshold': args.reasoning_threshold,
                            'reasoning_strict': args.reasoning_strict,
                            'reasoning_min_words': args.reasoning_min_words,
                            'reasoning_require_k': args.reasoning_require_k_signals,
                            'chatrag_threshold': args.chatrag_threshold,
                            'chatrag_strict': args.chatrag_strict,
                            'chatrag_min_words': args.chatrag_min_words,
                            'chatrag_require_k': args.chatrag_require_k_signals,
                            'roleplay_threshold': args.roleplay_threshold,
                            'function_calling_threshold': args.function_calling_threshold,
                            'debug_print': args.debug_print,
                        })
                
                tqdm.write(f"Processed {total_examples} examples from {fpath} in streaming mode")
                
                # Process any remaining chunks
                if chunks:
                    debug_print(args, f"DEBUG: Processing final {len(chunks)} chunks with {num_workers} workers")
                    debug_print(args, f"DEBUG: Using {'process_chunk_parquet' if args.input_type=='parquet' else 'process_chunk_json'} function")
                    
                    with ProcessPoolExecutor(max_workers=num_workers) as ex_pool:
                        futs = [ex_pool.submit(process_chunk_parquet if args.input_type=="parquet" else process_chunk_json, ch) for ch in chunks]
                        debug_print(args, f"DEBUG: Submitted {len(futs)} futures for processing")
                        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing final chunks", unit="chunk"):
                            res = fut.result()
                            debug_print(args, f"DEBUG: Chunk returned {len(res['kept'])} kept, {len(res['rejected'])} rejected")
                            
                            # Process results immediately
                            for r in res['kept']:
                                text = r['text']
                                norm = " ".join(text.split())
                                if args.dedup_exact and norm in seen_texts:
                                    duplicates += 1
                                    continue
                                if args.dedup_exact: seen_texts.add(norm)

                                decoded_ok += 1
                                for c in score_hist: score_hist[c].append(r['scores'][c])
                                for c in r['labels']: kept_by_class[c] += 1
                                kept += 1

                                rec = {args.text_output_key: text}
                                if args.include_labels: rec["labels"] = r['labels']
                                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                
                                for c in r['labels']:
                                    if c in cat_out:
                                        rec_cat = {args.text_output_key: text}
                                        if args.include_labels: rec_cat["labels"] = [c]
                                        cat_out[c].write(json.dumps(rec_cat, ensure_ascii=False) + "\n")
                                        if chatrag_strict_out and c=="chatrag" and args.chatrag_strict:
                                            chatrag_strict_out.write(json.dumps({args.text_output_key: text}, ensure_ascii=False) + "\n")
                            
                            for r in res['rejected']:
                                if args.collect_rejected:
                                    rej_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                                decoded_ok += 1  # Count rejected samples as decoded too
                                
                                # Analyze rejection reasons for statistics
                                if args.input_type == "parquet" and 'rejection_reason' in r:
                                    reason = r['rejection_reason']
                                    if reason.startswith('token_count_'):
                                        too_many_tokens += 1
                            
                            total_rows += (len(res['kept']) + len(res['rejected']))
                            debug_print(args, f"DEBUG: Updated total_rows to {total_rows}, decoded_ok to {decoded_ok}")

        # close files
        for c, fo in cat_out.items():
            fo.close()
            tqdm.write(f"Closed category output: {c}")
        if rej_out: rej_out.close()
        if reasoning_structural_out: reasoning_structural_out.close()
        if chatrag_strict_out: chatrag_strict_out.close()

        # summary
        def pct(x,y): return 100.0*x/max(1,y)
        with open(args.log_file, "a", encoding="utf-8") as log:
            log.write("\n# Summary\n")
            log.write(f"total_rows_seen: {total_rows}\n")
            print(f"total_rows_seen: {total_rows}")
            log.write(f"decoded_ok: {decoded_ok} ({pct(decoded_ok,total_rows):.2f}%)\n")
            print(f"decoded_ok: {decoded_ok} ({pct(decoded_ok,total_rows):.2f}%)")
            log.write(f"too_short: {too_short} (below {args.min_words} words)\n")
            print(f"too_short: {too_short} (below {args.min_words} words)")
            log.write(f"too_long: {too_long} (above {args.max_words} words)\n")
            print(f"too_long: {too_long} (above {args.max_words} words)")
            log.write(f"non_english: {non_english}\n")
            print(f"non_english: {non_english}")
            if args.input_type == "parquet":
                log.write(f"too_many_tokens: {too_many_tokens} (above {args.max_tokens} tokens)\n")
                print(f"too_many_tokens: {too_many_tokens} (above {args.max_tokens} tokens)")
            log.write(f"dedup_removed: {duplicates}\n")
            print(f"dedup_removed: {duplicates}")
            if args.collect_rejected:
                log.write(f"rejected_total: {total_rows - kept}\n")
                print(f"rejected_total: {total_rows - kept}")
            log.write(f"kept_total: {kept}\n")
            print(f"kept_total: {kept}")
            
            for c, v in kept_by_class.items():
                log.write(f"kept_{c}: {v}\n")
                print(f"kept_{c}: {v}")
            log.write("\n# score_stats\n")
            for c, vals in score_hist.items():
                if not vals: continue
                vals_sorted = sorted(vals)
                def q(p): 
                    k = int(p*(len(vals_sorted)-1)); 
                    return vals_sorted[k]
                log.write(f"{c}: n={len(vals_sorted)} min={vals_sorted[0]:.3f} p25={q(0.25):.3f} p50={q(0.5):.3f} p75={q(0.75):.3f} max={vals_sorted[-1]:.3f}\n")
            log.write("\n[DONE]\n")

if __name__ == "__main__":
    main()
