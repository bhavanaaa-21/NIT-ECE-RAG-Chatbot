# NIT Durgapur ECE Department Knowledge Assistant

import os
import re
import json

import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.json",
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db_dir",
)


# =========================================================
# CONFIG
# =========================================================

with open(CONFIG_PATH, "r") as f:
    config_data = json.load(f)

GROQ_API_KEY = config_data["GROQ_API_KEY"]

COLLEGE_NAME = config_data.get(
    "COLLEGE_NAME",
    "NIT Durgapur",
)

GROQ_MODEL = config_data.get(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# =========================================================
# CONSTANTS
# =========================================================

NOT_AVAILABLE = (
    "This information is not available in the "
    "provided ECE Department documents."
)


FAILURE_RESPONSE_MARKERS = (
    NOT_AVAILABLE,
    "The Groq API's daily usage limit",
    "The retrieved information is too large",
    "The language model could not "
    "generate a response",
)


def is_failure_response(text):

    if not text:
        return False

    stripped = text.strip()

    return any(
        stripped.startswith(marker)
        for marker in FAILURE_RESPONSE_MARKERS
    )


# Course codes such as:
# MAC331
# ECC301
# ECC302
# ECC304
# XEC02
# ECS551

COURSE_CODE_PATTERN = re.compile(
    r"\b[A-Z]{2,5}\s*\d{2,3}\*?\b",
    re.IGNORECASE,
)


MAX_CONTEXT_CHARS = 9000


# The generic "Professional/Depth Elective Paper N" slot
# code each semester's basket options fill, and share the
# credit of.
ELECTIVE_PLACEHOLDER_CODE_BY_SEMESTER = {
    5: "ECE510",
    6: "ECE610",
    7: "ECE710",
}


# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title=(
        "NIT Durgapur ECE "
        "Department Knowledge Assistant"
    ),
    page_icon="📡",
    layout="wide",
)


st.markdown(
    """
    <style>
    div.css-textbarboxtype {
        background-color: rgba(20, 25, 45, 0.65);
        border: 1px solid #4B528A;
        padding: 15px 18px;
        border-radius: 12px;
        color: #E8E8F0;
        margin-bottom: 12px;
    }

    div.css-textbarboxtype p {
        color: #E8E8F0;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# QUESTION DETECTION
# =========================================================

def detect_course(question):

    match = COURSE_CODE_PATTERN.search(
        question
    )

    if match:

        return re.sub(
            r"\s+",
            "",
            match.group(0),
        ).upper()

    return None


def detect_course_from_history(
    chat_history,
):

    for item in reversed(
        chat_history
    ):

        code = detect_course(
            item.get(
                "content",
                "",
            )
        )

        if code:

            return code

    return None


def detect_semester(question):

    q = question.lower()

    patterns = [

        (8, [
            r"\beighth\s+semester\b",
            r"\b8th\s+semester\b",
            r"\bsemester\s*viii\b",
            r"\bsemester\s*8\b",
        ]),

        (7, [
            r"\bseventh\s+semester\b",
            r"\b7th\s+semester\b",
            r"\bsemester\s*vii\b",
            r"\bsemester\s*7\b",
        ]),

        (6, [
            r"\bsixth\s+semester\b",
            r"\b6th\s+semester\b",
            r"\bsemester\s*vi\b",
            r"\bsemester\s*6\b",
        ]),

        (5, [
            r"\bfifth\s+semester\b",
            r"\b5th\s+semester\b",
            r"\bsemester\s*v\b",
            r"\bsemester\s*5\b",
        ]),

        (4, [
            r"\bfourth\s+semester\b",
            r"\b4th\s+semester\b",
            r"\bsemester\s*iv\b",
            r"\bsemester\s*4\b",
        ]),

        (3, [
            r"\bthird\s+semester\b",
            r"\b3rd\s+semester\b",
            r"\bsemester\s*iii\b",
            r"\bsemester\s*3\b",
        ]),

        (2, [
            r"\bsecond\s+semester\b",
            r"\b2nd\s+semester\b",
            r"\bsemester\s*ii\b",
            r"\bsemester\s*2\b",
        ]),

        (1, [
            r"\bfirst\s+semester\b",
            r"\b1st\s+semester\b",
            r"\bsemester\s*i\b",
            r"\bsemester\s*1\b",
        ]),
    ]

    for semester, regexes in patterns:

        if any(
            re.search(
                pattern,
                q,
            )
            for pattern in regexes
        ):

            return semester

    return None


def detect_semester_from_history(
    chat_history,
):

    for item in reversed(
        chat_history
    ):

        semester = detect_semester(
            item.get(
                "content",
                "",
            )
        )

        if semester is not None:

            return semester

    return None


GROUP_PATTERN = re.compile(
    r"\bgroup\s*[-]?\s*(ii|i|two|one|2|1)\b",
    re.IGNORECASE,
)

GROUP_WORD_MAP = {
    "1": 1,
    "i": 1,
    "one": 1,
    "2": 2,
    "ii": 2,
    "two": 2,
}


def detect_group(question):

    match = GROUP_PATTERN.search(
        question.lower()
    )

    if not match:
        return None

    return GROUP_WORD_MAP.get(
        match.group(1).lower()
    )


def is_credit_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "credit",
            "credits",
        ]
    )


def is_total_credit_question(question):

    q = question.lower()

    if not is_credit_question(question):

        return False

    return any(
        phrase in q
        for phrase in [
            "total",
            "how many",
            "sum of",
        ]
    )


def is_prerequisite_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "prerequisite",
            "prerequisites",
            "pre-requisite",
            "pre-requisites",
            "pre requisite",
            "pre requisites",
        ]
    )


# "which courses have tutorials?" / "what laboratories are
# included in the ECE curriculum?" are FILTER questions
# (asking to identify a subset matching some property), not
# a request for a semester's subject list - despite
# containing "course"/"curriculum", which would otherwise
# trigger is_subject_list_question. Deliberately narrow:
# only fires when "which"/"what" is immediately followed
# (within a couple of words) by one of these specific
# property nouns, so "what ARE THE SUBJECTS" is untouched.
FILTER_QUESTION_PATTERN = re.compile(
    r"\b(which|what)\s+(\w+\s+){0,2}"
    r"(laborator(?:y|ies)|labs?|"
    r"tutorials?|practicals?)\b",
    re.IGNORECASE,
)


def is_filtered_course_question(question):

    return bool(
        FILTER_QUESTION_PATTERN.search(
            question
        )
    )


def detect_filter_property(question):

    match = FILTER_QUESTION_PATTERN.search(
        question
    )

    if not match:

        return None

    property_word = match.group(
        3
    ).lower()

    if property_word.startswith(
        "labor"
    ) or property_word.startswith(
        "lab"
    ):

        return "laboratory"

    if property_word.startswith(
        "tutorial"
    ):

        return "tutorial"

    if property_word.startswith(
        "practical"
    ):

        return "practical"

    return None


COURSE_PRONOUN_PATTERN = re.compile(
    r"\b(it|its|this course|that course|the course|"
    r"this subject|that subject|the subject)\b",
    re.IGNORECASE,
)


def is_course_followup_reference(question):

    return bool(
        COURSE_PRONOUN_PATTERN.search(
            question
        )
    )


PLURAL_FOLLOWUP_PATTERN = re.compile(
    r"\b(their|them|these|those|"
    # "there credits" / "there prerequisites" etc. is a
    # common typo for "their credits" - treat it the same
    # way rather than silently missing the plural intent.
    r"there\s+(?:credits?|prerequisites?|modules?|"
    r"topics?|details?)|"
    r"all (?:of )?(?:the )?(?:courses?|electives?|"
    r"subjects?)|"
    r"all of them|each of them|"
    r"the (?:courses|electives|subjects)|"
    r"(?:courses?|electives?|subjects?) "
    r"(?:you |i )?(?:mentioned|listed|"
    r"shown|above))\b",
    re.IGNORECASE,
)


def is_plural_followup_reference(question):

    return bool(
        PLURAL_FOLLOWUP_PATTERN.search(
            question
        )
    )


def detect_requested_columns(question):

    q = question.lower()

    if not (
        "only" in q
        or "just" in q
    ):

        return None

    has_code = (
        "code" in q
    )

    has_title = (
        "title" in q
        or "name" in q
    )

    has_credit = (
        "credit" in q
    )

    if (
        has_code
        and not has_title
        and not has_credit
    ):

        return "code"

    if (
        has_title
        and not has_code
        and not has_credit
    ):

        return "title"

    if (
        has_credit
        and not has_code
        and not has_title
    ):

        return "credit"

    return None


def is_elective_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "elective",
            "electives",
            "depth elective",
            "professional elective",
            "programme elective",
            "program elective",
        ]
    )


def is_syllabus_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "syllabus",
            "syllabi",
            "topics",
            "topic covered",
            "topics covered",
            "module",
            "modules",
            "outline",
            "unit",
            "units",
        ]
    )


def is_subject_list_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "subject",
            "subjects",
            "course",
            "courses",
            "list",
            "curriculum",
            "paper",
            "papers",
        ]
    )


def is_credit_question(question):

    q = question.lower()

    return any(
        word in q
        for word in [
            "credit",
            "credits",
        ]
    )


def is_semester_question(question):

    q = question.lower()

    return (
        detect_semester(question) is not None
        and any(
            word in q
            for word in [
                "subject",
                "subjects",
                "course",
                "courses",
                "curriculum",
                "semester",
                "paper",
                "papers",
                "list",
                "credit",
                "credits",
            ]
        )
    )


# =========================================================
# VECTOR STORE
# =========================================================

@st.cache_resource
def setup_vectorstore():

    embeddings = HuggingFaceEmbeddings()

    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
    )


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        # A course with many modules (e.g. ECC304 has 12)
        # plus prerequisites, outcomes and textbooks easily
        # exceeds 900 tokens; a low cap was silently
        # truncating those answers mid-module. Kept well
        # below Groq's 8000 TPM per-request ceiling once
        # combined with prompt + context size.
        max_tokens=2000,
    )


# =========================================================
# METADATA FILTER
# =========================================================

def build_filter(
    course_code=None,
    semester=None,
    group=None,
    document_type=None,
):

    conditions = []

    if course_code:

        conditions.append(
            {
                "course_code": {
                    "$eq": course_code,
                }
            }
        )

    if semester is not None:

        conditions.append(
            {
                "semester": {
                    "$eq": semester,
                }
            }
        )

    if group is not None:

        conditions.append(
            {
                "group": {
                    "$eq": group,
                }
            }
        )

    if document_type:

        conditions.append(
            {
                "document_type": {
                    "$eq": document_type,
                }
            }
        )

    if not conditions:

        return None

    if len(conditions) == 1:

        return conditions[0]

    return {
        "$and": conditions,
    }


# =========================================================
# CHROMA RESULT -> DOCUMENTS
# =========================================================

def chroma_result_to_documents(result):

    raw_documents = result.get(
        "documents",
        [],
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    documents = []

    for content, metadata in zip(
        raw_documents,
        metadatas,
    ):

        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata=metadata or {},
            )
        )

    documents.sort(
        key=lambda doc: (
            safe_number(
                doc.metadata.get(
                    "page",
                    0,
                )
            ),
            safe_number(
                doc.metadata.get(
                    "chunk_index",
                    0,
                )
            ),
        )
    )

    return documents


def safe_number(value):

    try:
        return int(value)
    except Exception:
        return 0


# =========================================================
# EXACT METADATA RETRIEVAL
# =========================================================

def exact_metadata_retrieval(
    vectorstore,
    course_code=None,
    semester=None,
    group=None,
    document_type=None,
):

    metadata_filter = build_filter(
        course_code=course_code,
        semester=semester,
        group=group,
        document_type=document_type,
    )

    if metadata_filter is None:
        return []

    try:

        result = vectorstore.get(
            where=metadata_filter,
            include=[
                "documents",
                "metadatas",
            ],
        )

        return chroma_result_to_documents(
            result
        )

    except Exception as e:

        print(
            "Metadata retrieval failed:",
            repr(e),
        )

        return []


# =========================================================
# GET ALL DOCUMENTS
# =========================================================

def get_all_documents(vectorstore):

    try:

        result = vectorstore.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

        return chroma_result_to_documents(
            result
        )

    except Exception as e:

        print(
            "Full vector database retrieval failed:",
            repr(e),
        )

        return []


# =========================================================
# NORMALIZE COURSE CODE
# =========================================================

def normalize_course_code(code):

    return re.sub(
        r"\s+",
        "",
        code,
    ).upper()


# =========================================================
# COURSE RETRIEVAL
# =========================================================

# PDF OCR sometimes injects a stray space/newline inside a
# course code itself (e.g. "ECC302" -> "ECC3 02"), not just
# between its letter and digit runs - the same kind of
# artifact already handled elsewhere for words like
# "Electiv e" or "Laborator y". COURSE_CODE_PATTERN's regex
# can't detect a code broken up like that at all, so build a
# fully whitespace-tolerant version of one specific code here
# for the places that need to find a code regardless of where
# the PDF happened to break it.
def build_code_pattern(code):

    return r"\s*".join(
        re.escape(character)
        for character in code
    )


def code_appears_in_text(code, text):

    return bool(
        re.search(
            r"\b" + build_code_pattern(code) + r"\b",
            text,
            re.IGNORECASE,
        )
    )


def looks_like_course_header(
    code,
    text,
):

    return bool(
        re.search(
            build_code_pattern(code)
            + r".{0,90}?\b(PCR|PEL)\b",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )


def extend_with_continuation_chunks(
    all_documents,
    seed_document,
    course_code,
    max_extra=8,
):

    try:

        start_position = next(
            i
            for i, document in enumerate(
                all_documents
            )
            if (
                document.metadata.get(
                    "page"
                )
                == seed_document.metadata.get(
                    "page"
                )
                and document.metadata.get(
                    "chunk_index"
                )
                == seed_document.metadata.get(
                    "chunk_index"
                )
            )
        )

    except StopIteration:

        return []

    seed_page = seed_document.metadata.get(
        "page"
    )

    seed_type = seed_document.metadata.get(
        "document_type"
    )

    extension = []

    for offset in range(1, max_extra + 1):

        position = start_position + offset

        if position >= len(all_documents):
            break

        candidate = all_documents[position]

        # A course's entry can run past the end of its
        # starting page (e.g. XEC02's own textbooks/reference
        # material land on the following page, after its
        # course-outcome mapping table) - allow the walk to
        # cross onto later pages too, not just stay within
        # seed_page. Still capped at a handful of pages so a
        # long run of untagged "GENERAL" pages can't wander
        # arbitrarily far past this course's real entry.
        if (
            candidate.metadata.get("page")
            is None
            or candidate.metadata.get("page")
            > seed_page + 3
        ):
            break

        if (
            candidate.metadata.get(
                "document_type"
            )
            != seed_type
        ):
            break

        candidate_codes = (
            extract_course_codes_from_text(
                candidate.page_content
            )
        )

        other_codes = [
            code
            for code in candidate_codes
            if code != course_code
        ]

        extension.append(
            candidate
        )

        if other_codes:
            break

    return extension


def retrieve_course_documents(
    vectorstore,
    course_code,
):

    course_code = normalize_course_code(
        course_code
    )

    # -----------------------------------------------------
    # METHOD 1:
    # Exact metadata
    #
    # NOTE: this only ever finds chunks from PDF pages that
    # had exactly this one course code on them during
    # ingestion. A page shared with other courses' content
    # (e.g. where this course's own "Pre-requisites" line
    # landed on the previous, multi-code page) never gets
    # tagged with this code at all, so an exact-match result
    # here is not necessarily complete. Don't return early -
    # always also run Method 2 below and merge, so a partial
    # Method 1 hit can't silently hide the rest of the entry.
    # -----------------------------------------------------

    exact_match_documents = exact_metadata_retrieval(
        vectorstore=vectorstore,
        course_code=course_code,
    )

    # -----------------------------------------------------
    # METHOD 2:
    # Search all chunks
    # -----------------------------------------------------

    all_documents = get_all_documents(
        vectorstore
    )

    matching_documents = []

    for document in all_documents:

        # Semester curriculum table rows only list the
        # code alongside dozens of others in that semester;
        # they are never the course's actual syllabus, so
        # a code match there is always noise here.
        if (
            document.metadata.get(
                "document_type"
            )
            == "semester_curriculum"
        ):
            continue

        text_codes = re.findall(
            COURSE_CODE_PATTERN,
            document.page_content.upper(),
        )

        normalized_codes = [
            normalize_course_code(code)
            for code in text_codes
        ]

        if (
            course_code in normalized_codes
            or code_appears_in_text(
                course_code,
                document.page_content,
            )
        ):

            matching_documents.append(
                document
            )

    # Merge in Method 1's exact-tagged chunks too (a page
    # tagged with this course code alone might not itself
    # satisfy the text-scan above if, say, the code only
    # appears once and gets stripped oddly - keep both).
    for document in exact_match_documents:

        key = (
            document.metadata.get(
                "page"
            ),
            document.metadata.get(
                "chunk_index"
            ),
        )

        if not any(
            (
                existing.metadata.get(
                    "page"
                ),
                existing.metadata.get(
                    "chunk_index"
                ),
            )
            == key
            for existing in matching_documents
        ):

            matching_documents.append(
                document
            )

    # A bare text-scan match is only trustworthy as the
    # course's OWN content if it looks like that course's
    # real header row. Otherwise it's almost always just a
    # citation inside some OTHER course's prerequisites list
    # (e.g. XEC02, a very commonly required prerequisite,
    # pulled in dozens of unrelated courses' chunks this way
    # before this filter existed). Method 1's exact-tagged
    # chunks are always trusted, since ingestion only tags a
    # page with a specific code when that page had exactly
    # one code on it - but that single-code detection can
    # itself be wrong (an OCR-mangled neighbouring code can
    # drop out of the scan, leaving this code as the page's
    # only detected match even though the page is actually
    # about a DIFFERENT course that merely cites this one as
    # a prerequisite - this happened for XEC02 on a page that
    # was really the tail of ECC301 / start of ECC302). So
    # only trust a whole exact-tagged page if it isn't showing
    # the classic mistag signature: the code cited near
    # "prerequisites" wording SOMEWHERE on the page, with no
    # chunk on that same page genuinely being this course's own
    # header row. A legitimate continuation-only page (this
    # course's header printed on an earlier page, this page
    # just carries later syllabus/textbook content with no
    # repeat of the code at all) shows neither signal, so it
    # stays trusted; only a real prerequisite-citation mistag
    # like XEC02's page 32 (ECC302's own prerequisites field
    # naming XEC02, with no XEC02 header anywhere on that page)
    # gets excluded.
    exact_pages = {
        document.metadata.get("page")
        for document in exact_match_documents
    }

    exact_pass_pages = set()

    for page in exact_pages:

        page_documents = [
            document
            for document in exact_match_documents
            if document.metadata.get("page")
            == page
        ]

        has_header = any(
            looks_like_course_header(
                course_code,
                document.page_content,
            )
            for document in page_documents
        )

        has_prerequisite_citation = any(
            is_likely_prerequisite_code(
                course_code,
                document.page_content,
            )
            for document in page_documents
        )

        if has_header or not has_prerequisite_citation:

            exact_pass_pages.add(page)

    matching_documents = [
        document
        for document in matching_documents
        if document.metadata.get("page")
        in exact_pass_pages
        or looks_like_course_header(
            course_code,
            document.page_content,
        )
    ]

    if matching_documents:

        seen_keys = {
            (
                document.metadata.get(
                    "page"
                ),
                document.metadata.get(
                    "chunk_index"
                ),
            )
            for document in matching_documents
        }

        extended_documents = list(
            matching_documents
        )

        for seed_document in matching_documents:

            # Only extend from a chunk that is genuinely
            # this course's own header row. A bare mention
            # elsewhere (e.g. as another course's listed
            # prerequisite) is not a starting point to walk
            # forward from - doing so would pull in several
            # chunks of that unrelated course instead.
            #
            # (is_likely_prerequisite_code is deliberately
            # NOT used here: a course's own entry always has
            # its own "Pre-requisites" field shortly after
            # its own header, so that check would also flag
            # - and block extension from - the genuine seed.)
            if not looks_like_course_header(
                course_code,
                seed_document.page_content,
            ):
                continue

            continuation = (
                extend_with_continuation_chunks(
                    all_documents,
                    seed_document,
                    course_code,
                )
            )

            for extra_document in continuation:

                key = (
                    extra_document.metadata.get(
                        "page"
                    ),
                    extra_document.metadata.get(
                        "chunk_index"
                    ),
                )

                if key in seen_keys:
                    continue

                seen_keys.add(key)

                extended_documents.append(
                    extra_document
                )

        extended_documents.sort(
            key=lambda document: (
                safe_number(
                    document.metadata.get(
                        "page",
                        0,
                    )
                ),
                safe_number(
                    document.metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
            )
        )

        return extended_documents

    # -----------------------------------------------------
    # METHOD 3:
    # Similarity fallback
    # -----------------------------------------------------

    try:

        documents = vectorstore.similarity_search(
            f"course {course_code}",
            k=10,
        )

        filtered_documents = []

        for document in documents:

            if (
                document.metadata.get(
                    "document_type"
                )
                == "semester_curriculum"
            ):
                continue

            text_codes = re.findall(
                COURSE_CODE_PATTERN,
                document.page_content.upper(),
            )

            normalized_codes = [
                normalize_course_code(code)
                for code in text_codes
            ]

            if course_code in normalized_codes:

                filtered_documents.append(
                    document
                )

        if filtered_documents:

            return filtered_documents

    except Exception as e:

        print(
            "Course similarity retrieval failed:",
            repr(e),
        )

    return []


# =========================================================
# SEMESTER RETRIEVAL
# =========================================================

def retrieve_semester_documents(
    vectorstore,
    semester,
    group=None,
):

    # -----------------------------------------------------
    # METHOD 1:
    # Exact semester metadata
    # -----------------------------------------------------

    documents = exact_metadata_retrieval(
        vectorstore=vectorstore,
        semester=semester,
        group=group,
    )

    if documents:
        return documents

    # -----------------------------------------------------
    # METHOD 2:
    # Inspect every stored document's metadata
    # -----------------------------------------------------

    all_documents = get_all_documents(
        vectorstore
    )

    semester_documents = []

    for document in all_documents:

        metadata = document.metadata or {}

        metadata_semester = metadata.get(
            "semester"
        )

        if (
            str(metadata_semester).strip()
            != str(semester)
        ):
            continue

        if group is not None:

            metadata_group = metadata.get(
                "group"
            )

            if (
                str(metadata_group).strip()
                != str(group)
            ):
                continue

        semester_documents.append(
            document
        )

    if semester_documents:

        semester_documents.sort(
            key=lambda doc: (
                safe_number(
                    doc.metadata.get(
                        "page",
                        0,
                    )
                ),
                safe_number(
                    doc.metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
            )
        )

        return semester_documents

    # -----------------------------------------------------
    # METHOD 3:
    # Search text
    # -----------------------------------------------------

    semester_documents = []

    patterns = {

        1: [
            r"\bsemester\s*i\b",
            r"\bsemester\s*1\b",
            r"\bsemester[-\s]*1\b",
            r"\bsemester[-\s]*i\b",
        ],

        2: [
            r"\bsemester\s*ii\b",
            r"\bsemester\s*2\b",
            r"\bsemester[-\s]*2\b",
            r"\bsemester[-\s]*ii\b",
        ],

        3: [
            r"\bsemester\s*iii\b",
            r"\bsemester\s*3\b",
            r"\bsemester[-\s]*3\b",
            r"\bsemester[-\s]*iii\b",
        ],

        4: [
            r"\bsemester\s*iv\b",
            r"\bsemester\s*4\b",
            r"\bsemester[-\s]*4\b",
            r"\bsemester[-\s]*iv\b",
        ],

        5: [
            r"\bsemester\s*v\b",
            r"\bsemester\s*5\b",
            r"\bsemester[-\s]*5\b",
            r"\bsemester[-\s]*v\b",
        ],

        6: [
            r"\bsemester\s*vi\b",
            r"\bsemester\s*6\b",
            r"\bsemester[-\s]*6\b",
            r"\bsemester[-\s]*vi\b",
        ],

        7: [
            r"\bsemester\s*vii\b",
            r"\bsemester\s*7\b",
            r"\bsemester[-\s]*7\b",
            r"\bsemester[-\s]*vii\b",
        ],

        8: [
            r"\bsemester\s*viii\b",
            r"\bsemester\s*8\b",
            r"\bsemester[-\s]*8\b",
            r"\bsemester[-\s]*viii\b",
        ],
    }

    target_patterns = patterns.get(
        semester,
        [],
    )

    for document in all_documents:

        text = document.page_content.lower()

        if any(
            re.search(
                pattern,
                text,
            )
            for pattern in target_patterns
        ):

            semester_documents.append(
                document
            )

    return semester_documents


# =========================================================
# COURSE-CODE EXTRACTION
# =========================================================

def extract_course_codes_from_text(
    text,
):

    matches = re.findall(
        COURSE_CODE_PATTERN,
        text.upper(),
    )

    codes = []

    for match in matches:

        normalized = normalize_course_code(
            match
        )

        if normalized not in codes:

            codes.append(
                normalized
            )

    return codes


# =========================================================
# COURSE CHUNKS
# =========================================================

def extract_course_chunks(
    documents,
):

    course_documents = []

    seen = set()

    for document in documents:

        codes = extract_course_codes_from_text(
            document.page_content
        )

        if not codes:
            continue

        page = safe_number(
            document.metadata.get(
                "page",
                0,
            )
        )

        chunk_index = safe_number(
            document.metadata.get(
                "chunk_index",
                0,
            )
        )

        key = (
            page,
            chunk_index,
        )

        if key in seen:
            continue

        seen.add(key)

        course_documents.append(
            document
        )

    course_documents.sort(
        key=lambda document: (
            safe_number(
                document.metadata.get(
                    "page",
                    0,
                )
            ),
            safe_number(
                document.metadata.get(
                    "chunk_index",
                    0,
                )
            ),
        )
    )

    return course_documents


# =========================================================
# SEMESTER SUBJECT RETRIEVAL
# =========================================================

def retrieve_semester_subject_documents(
    vectorstore,
    semester,
    group=None,
):

    # Prefer the clean per-semester overview table (short,
    # one row per course) when one exists. The large
    # detailed-syllabus block for a semester repeats each
    # course code multiple times across many pages, which
    # makes per-course title/credit extraction unreliable.
    table_documents = exact_metadata_retrieval(
        vectorstore=vectorstore,
        semester=semester,
        group=group,
        document_type="semester_curriculum",
    )

    if table_documents:

        return table_documents

    semester_documents = retrieve_semester_documents(
        vectorstore,
        semester,
        group=group,
    )

    if not semester_documents:

        return []

    course_documents = extract_course_chunks(
        semester_documents
    )

    if course_documents:

        return course_documents

    return semester_documents


# =========================================================
# IMPORTANT:
# EXCLUDE PREREQUISITE CODES
# =========================================================

def is_likely_prerequisite_code(
    code,
    text,
):

    upper_text = text.upper()

    code = normalize_course_code(
        code
    )

    # If the course code appears close to
    # prerequisite-related wording, don't treat
    # it as a semester subject.

    prerequisite_patterns = [
        "PREREQUISITE",
        "PREREQUISITES",
        "PRE-REQUISITE",
        "PRE REQUISITE",
        "REQUIRED COURSE",
    ]

    for pattern in prerequisite_patterns:

        positions = [
            match.start()
            for match in re.finditer(
                pattern,
                upper_text,
            )
        ]

        for position in positions:

            nearby_text = upper_text[
                max(
                    0,
                    position - 250,
                ):
                min(
                    len(upper_text),
                    position + 500,
                )
            ]

            if code in nearby_text:

                return True

    return False


# =========================================================
# BUILD SEMESTER COURSE ROWS
#
# This function extracts course information from the
# retrieved curriculum chunks BEFORE sending it to Groq.
#
# This is much safer than asking the LLM to discover
# every course itself.
# =========================================================

def extract_semester_course_rows(
    documents,
):

    rows = {}

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        codes = extract_course_codes_from_text(
            text
        )

        if not codes:
            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        # Iterate over every LINE that carries a course code,
        # not just each unique code. A placeholder code like
        # "ECE710" can legitimately head two distinct rows in
        # the same table (e.g. "Professional Elective Paper 4"
        # and "...Paper 5"); collapsing to unique codes first
        # would silently keep only one of them.

        code_occurrences = []

        for i, line in enumerate(lines):

            for line_code in extract_course_codes_from_text(
                line
            ):

                code_occurrences.append(
                    (i, line_code)
                )

        for (
            code_position,
            code,
        ) in code_occurrences:

            # "TOTAL 15 4 9 25 28" (the table's summary
            # row) matches the course-code shape but is
            # never a real course.
            if code.startswith("TOTAL"):
                continue

            if is_likely_prerequisite_code(
                code,
                text,
            ):
                continue

            # PyPDF2 sometimes wraps a long title onto the
            # next physical line, e.g. "...Principles" /
            # "of Management 3 0 0 3 3". If this row's own
            # line doesn't already end with the expected
            # trailing L/T/S/C/H numbers, and the next line
            # is plain continuation text (no course code of
            # its own), merge it back in before extracting
            # the title - otherwise it gets orphaned and
            # mistakenly picked up as the NEXT course's
            # title instead.

            if (
                code_position >= 0
                and code_position + 1
                < len(lines)
                and not re.search(
                    r"(?:\d+\s+){4}\d+\s*$",
                    lines[code_position],
                )
            ):

                next_line = lines[
                    code_position + 1
                ]

                if (
                    next_line
                    and not extract_course_codes_from_text(
                        next_line
                    )
                    and re.search(
                        r"[A-Za-z]{2,}",
                        next_line,
                    )
                ):

                    lines[code_position] = (
                        lines[
                            code_position
                        ].rstrip()
                        + " "
                        + next_line.strip()
                    )

                    # Blank the now-absorbed line so it
                    # isn't independently picked up again as
                    # a later course's title.
                    lines[
                        code_position + 1
                    ] = ""

            # Capture the L/T/P (Lecture/Tutorial/Practical)
            # hours too, from the same trailing five-number
            # sequence the credit comes from ("...3 1 0 4 4"
            # = L=3, T=1, P=0, Total=4, Credit=4). Reuses the
            # already-merged line above rather than
            # re-deriving it.
            lecture_hours = None

            tutorial_hours = None

            practical_hours = None

            if code_position >= 0:

                hours_match = re.search(
                    r"(\d+)\s+(\d+)\s+(\d+)"
                    r"\s+(\d+)\s+(\d+)\s*$",
                    lines[code_position],
                )

                if hours_match:

                    lecture_hours = (
                        hours_match.group(1)
                    )

                    tutorial_hours = (
                        hours_match.group(2)
                    )

                    practical_hours = (
                        hours_match.group(3)
                    )

            candidate_lines = []

            if code_position >= 0:

                start = max(
                    0,
                    code_position - 2,
                )

                end = min(
                    len(lines),
                    code_position + 6,
                )

                candidate_lines = [
                    (idx, lines[idx])
                    for idx in range(
                        start,
                        end,
                    )
                ]

            else:

                candidate_lines = [
                    (idx, lines[idx])
                    for idx in range(
                        0,
                        min(15, len(lines)),
                    )
                ]

            # -------------------------------------------------
            # Find a possible title and credit.
            # -------------------------------------------------

            title = ""
            credit = ""

            for (
                candidate_index,
                line,
            ) in candidate_lines:

                upper_line = line.upper()

                # Ignore pure course-code line.
                if normalize_course_code(
                    line
                ) == code:

                    continue

                # A line carrying ANY course code - the same
                # code repeated on another row (e.g. a second
                # "ECE710" for a different elective paper), or
                # a different code entirely - belongs to that
                # OTHER row, never to this occurrence's own
                # title, unless it's this occurrence's own
                # designated line. Without this check, a
                # neighboring row's full text can get greedily
                # picked up as this row's title.
                if (
                    candidate_index
                    != code_position
                    and extract_course_codes_from_text(
                        line
                    )
                ):

                    continue

                # Credit detection.
                credit_match = re.search(
                    r"\b([1-9]|1[0-2])\s*(?:CREDIT|CREDITS|CR)\b",
                    upper_line,
                )

                if credit_match:

                    credit = credit_match.group(1)

                # Common table patterns:
                #
                # ECC304 Digital Circuits and Systems 4
                #
                # ECC304 | Digital Circuits and Systems | 4
                #
                # Try removing course code first.

                cleaned = re.sub(
                    re.escape(code),
                    "",
                    line,
                    flags=re.IGNORECASE,
                )

                cleaned = cleaned.strip(
                    " |:-\t"
                )

                if not cleaned:
                    continue

                # Skip obvious headers.
                if cleaned.upper() in [
                    "COURSE CODE",
                    "COURSE TITLE",
                    "TITLE OF THE COURSE",
                    "CREDIT",
                    "CREDITS",
                ]:
                    continue

                # Skip the table's own column-header row
                # (e.g. "Sl. Code Subject L T S C H"),
                # which otherwise gets picked up as the
                # title for whichever course sits closest
                # to the top of the table.
                if (
                    "SUBJECT" in cleaned.upper()
                    and (
                        "CODE" in cleaned.upper()
                        or cleaned.upper().startswith(
                            "SL"
                        )
                    )
                ):
                    continue

                # Skip prerequisite labels.
                if cleaned.upper().startswith(
                    "PREREQUISITE"
                ):
                    continue

                # If line contains a number at the end,
                # separate it as credit.

                end_credit = re.search(
                    r"\s+([1-9]|1[0-2])\s*$",
                    cleaned,
                )

                if end_credit:

                    if not credit:

                        credit = (
                            end_credit.group(1)
                        )

                    cleaned = re.sub(
                        r"\s+([1-9]|1[0-2])\s*$",
                        "",
                        cleaned,
                    ).strip(
                        " |:-\t"
                    )

                # Remove table separators.

                cleaned = re.sub(
                    r"\s*\|\s*",
                    " | ",
                    cleaned,
                )

                # Avoid metadata-like text.

                bad_words = [
                    "semester",
                    "prerequisite",
                    "assessment",
                    "course outcome",
                    "contact hours",
                    "lecture hours",
                    "tutorial hours",
                    "practical hours",
                ]

                if any(
                    word in cleaned.lower()
                    for word in bad_words
                ):
                    continue

                # A reasonable title normally has letters.

                if re.search(
                    r"[A-Za-z]{3,}",
                    cleaned,
                ):

                    if not title:

                        title = cleaned

            # -------------------------------------------------
            # If we have no title yet, look for the next
            # meaningful line.
            # -------------------------------------------------

            if not title and code_position >= 0:

                for next_line in lines[
                    code_position + 1:
                ]:

                    cleaned = next_line.strip(
                        " |:-\t"
                    )

                    if not cleaned:
                        continue

                    # Any course code here (this one or a
                    # different one) means the line belongs
                    # to a table row, not a free-standing
                    # title.
                    if extract_course_codes_from_text(
                        cleaned
                    ):
                        continue

                    if (
                        "prerequisite"
                        in cleaned.lower()
                    ):
                        continue

                    if re.search(
                        r"[A-Za-z]{3,}",
                        cleaned,
                    ):

                        title = cleaned
                        break

            # -------------------------------------------------
            # Clean title
            # -------------------------------------------------

            title = re.sub(
                r"\s+",
                " ",
                title,
            ).strip()

            # Strip a leading table Sl.No (e.g. "3 Signals
            # and Systems" -> "Signals and Systems"), but
            # only when real letters remain afterward.
            title_without_number = re.sub(
                r"^\d+\s*[\.\)]?\s*",
                "",
                title,
            )

            if re.search(
                r"[A-Za-z]{3,}",
                title_without_number,
            ):

                title = title_without_number

            # Strip trailing leftover numeric table columns
            # (e.g. L T S from before the credit column was
            # split off).
            title = re.sub(
                r"(\s+\d+){1,4}\s*$",
                "",
                title,
            ).strip()

            # Don't accidentally use the whole table header
            # as a course title.

            if title.upper() in [
                "COURSE CODE",
                "COURSE TITLE",
                "TITLE OF THE COURSE",
                "TITLE OF THE",
            ]:

                title = ""

            # -------------------------------------------------
            # Store.
            #
            # If the same code appears in multiple chunks
            # with the SAME title, it's the same row seen
            # twice - merge, preferring the more complete
            # data. But a placeholder code like "ECE710" can
            # legitimately appear as two DISTINCT rows in the
            # curriculum (e.g. "Professional Elective Paper 4"
            # and "...Paper 5"), so the dedup key includes the
            # title, not just the code, or one of those rows
            # would silently vanish.
            # -------------------------------------------------

            row_key = (
                code,
                title,
            )

            existing = rows.get(
                row_key
            )

            document_group = (
                document.metadata.get(
                    "group",
                    0,
                )
                or 0
            )

            if existing is None:

                rows[row_key] = {
                    "code": code,
                    "title": title,
                    "credit": credit,
                    "lecture": lecture_hours,
                    "tutorial": tutorial_hours,
                    "practical": practical_hours,
                    "groups": {
                        document_group
                    },
                }

            else:

                if (
                    len(title)
                    > len(existing["title"])
                ):

                    existing["title"] = title

                if (
                    not existing["credit"]
                    and credit
                ):

                    existing["credit"] = credit

                if (
                    not existing.get(
                        "tutorial"
                    )
                    and tutorial_hours
                ):

                    existing["lecture"] = (
                        lecture_hours
                    )

                    existing["tutorial"] = (
                        tutorial_hours
                    )

                    existing["practical"] = (
                        practical_hours
                    )

                existing["groups"].add(
                    document_group
                )

    return list(
        rows.values()
    )


# =========================================================
# SEMESTER SUBJECT CONTEXT
#
# First try to build a compact course list.
# =========================================================

def build_semester_subject_context(
    documents,
    max_chars=7000,
):

    rows = extract_semester_course_rows(
        documents
    )

    # If any row came from a Group 1/Group 2 specific block,
    # show which group(s) each course belongs to explicitly,
    # instead of leaving the model to infer it from raw text.
    show_group_column = any(
        row["groups"] - {0}
        for row in rows
    )

    parts = []

    if rows:

        parts.append(
            "COURSE INFORMATION EXTRACTED FROM "
            "SEMESTER CURRICULUM:\n"
        )

        header = (
            "# | Course Code | Course Title | Credit"
        )

        if show_group_column:

            header += " | Group"

        parts.append(
            header
        )

        for index, row in enumerate(
            rows,
            start=1,
        ):

            title = (
                row["title"]
                if row["title"]
                else "Not specified"
            )

            credit = (
                row["credit"]
                if row["credit"]
                else "Not specified"
            )

            row_line = (
                f"{index} | "
                f"{row['code']} | "
                f"{title} | "
                f"{credit}"
            )

            if show_group_column:

                groups = (
                    row["groups"] - {0}
                )

                if groups == {1, 2}:

                    group_label = (
                        "Both Group 1 and Group 2"
                    )

                elif groups == {1}:

                    group_label = "Group 1 only"

                elif groups == {2}:

                    group_label = "Group 2 only"

                else:

                    group_label = "Common"

                row_line += (
                    f" | {group_label}"
                )

            parts.append(
                row_line
            )

    extracted_context = "\n".join(
        parts
    )

    # -----------------------------------------------------
    # Add source chunks after the extracted rows.
    # -----------------------------------------------------

    source_parts = []

    current_length = len(
        extracted_context
    )

    for document in documents:

        text = document.page_content.strip()

        if not text:
            continue

        page = document.metadata.get(
            "page",
            "",
        )

        source_part = (
            f"\n\n[PDF PAGE {page}]\n"
            f"{text}"
        )

        if (
            current_length
            + len(source_part)
            > max_chars
        ):

            break

        source_parts.append(
            source_part
        )

        current_length += len(
            source_part
        )

    if extracted_context:

        return (
            extracted_context
            + "".join(source_parts)
        )

    return "".join(
        source_parts
    )


# =========================================================
# WHOLE-CURRICULUM COURSE FILTERS (DIRECT RENDER)
#
# "Which courses have tutorials?" / "what laboratories are
# included?" need to check every course across every
# semester - exactly the kind of exhaustive lookup the
# already-verified per-semester extraction handles reliably,
# and an LLM given only a handful of similarity-matched
# chunks cannot.
# =========================================================

def gather_all_semester_course_rows(
    vectorstore,
):

    all_rows = []

    seen_codes = set()

    for semester in range(1, 8):

        documents = (
            retrieve_semester_subject_documents(
                vectorstore,
                semester,
                group=None,
            )
        )

        rows = extract_semester_course_rows(
            documents
        )

        for row in rows:

            if row["code"] in seen_codes:
                continue

            seen_codes.add(
                row["code"]
            )

            row["semester"] = semester

            all_rows.append(
                row
            )

    return all_rows


def filter_course_rows_by_property(
    rows,
    property_name,
):

    if property_name == "laboratory":

        # PDF text extraction sometimes injects a stray
        # space inside this word too (e.g. "Laborator y").
        laboratory_pattern = re.compile(
            r"labor\s*ator\s*y",
            re.IGNORECASE,
        )

        return [
            row
            for row in rows
            if laboratory_pattern.search(
                row["title"]
            )
        ]

    if property_name == "tutorial":

        return [
            row
            for row in rows
            if row.get("tutorial")
            not in (
                None,
                "0",
            )
        ]

    if property_name == "practical":

        return [
            row
            for row in rows
            if row.get("practical")
            not in (
                None,
                "0",
            )
        ]

    return []


def format_filtered_course_table(
    rows,
    property_name,
):

    if not rows:

        return ""

    property_labels = {
        "laboratory": "Laboratory",
        "tutorial": "Tutorial Hours",
        "practical": "Practical Hours",
    }

    detail_key = {
        "laboratory": None,
        "tutorial": "tutorial",
        "practical": "practical",
    }[
        property_name
    ]

    header = (
        "| # | Course Code | Course Title | "
        "Semester"
    )

    separator = "|---|---|---|---|"

    if detail_key:

        header += (
            f" | {property_labels[property_name]}"
        )

        separator += "---|"

    header += " |"

    lines = [
        header,
        separator,
    ]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        line = (
            f"| {index} | "
            f"{row['code']} | "
            f"{row['title']} | "
            f"{row['semester']}"
        )

        if detail_key:

            line += (
                f" | {row.get(detail_key)}"
            )

        line += " |"

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


# =========================================================
# SEMESTER SUBJECT TABLE (DIRECT RENDER)
#
# The subject list is already fully and correctly extracted
# in Python before an LLM is ever involved. Asking the model
# to transcribe a table it has no need to compose only risks
# it dropping/truncating rows, especially under the tighter
# limits of a smaller model. Render it directly instead.
# =========================================================

def format_semester_subject_table(
    rows,
    column_filter=None,
):

    if not rows:

        return ""

    if column_filter == "code":

        return ", ".join(
            row["code"]
            for row in rows
        )

    if column_filter == "title":

        return "\n".join(
            row["title"]
            if row["title"]
            else "Not specified"
            for row in rows
        )

    if column_filter == "credit":

        return "\n".join(
            f"{row['code']}: "
            + (
                row["credit"]
                if row["credit"]
                else "Not specified"
            )
            for row in rows
        )

    show_group_column = any(
        row["groups"] - {0}
        for row in rows
    )

    header = (
        "| # | Course Code | Course Title | Credit |"
    )

    separator = (
        "|---|---|---|---|"
    )

    if show_group_column:

        header += " Group |"
        separator += "---|"

    lines = [
        header,
        separator,
    ]

    for index, row in enumerate(
        rows,
        start=1,
    ):

        title = (
            row["title"]
            if row["title"]
            else "Not specified"
        )

        credit = (
            row["credit"]
            if row["credit"]
            else "Not specified"
        )

        line = (
            f"| {index} | "
            f"{row['code']} | "
            f"{title} | "
            f"{credit} |"
        )

        if show_group_column:

            groups = (
                row["groups"] - {0}
            )

            if groups == {1, 2}:

                group_label = (
                    "Both Group 1 and Group 2"
                )

            elif groups == {1}:

                group_label = "Group 1 only"

            elif groups == {2}:

                group_label = "Group 2 only"

            else:

                group_label = "Common"

            line += f" {group_label} |"

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


# =========================================================
# SINGLE COURSE ROW LOOKUP
#
# The clean per-semester overview tables already give a
# verified, correct code/title/credit row for every course.
# For a question about one specific course's credit, look it
# up there directly instead of asking the LLM to find a
# number buried in dense multi-page syllabus text.
# =========================================================

def find_course_row(
    vectorstore,
    course_code,
):

    course_code = normalize_course_code(
        course_code
    )

    all_documents = get_all_documents(
        vectorstore
    )

    table_documents = [
        document
        for document in all_documents
        if document.metadata.get(
            "document_type"
        )
        == "semester_curriculum"
    ]

    rows = extract_semester_course_rows(
        table_documents
    )

    for row in rows:

        if row["code"] == course_code:

            return row

    return None


# =========================================================
# ELECTIVE BASKET PARSING (DIRECT RENDER)
#
# A basket document's text is just a flat list of
# "CODE Title" lines (no Sl.No/L/T/P/Credit columns), so it
# needs its own simple parser rather than reusing
# extract_semester_course_rows, which expects a numbered
# table row shape.
# =========================================================

def parse_elective_basket_rows(
    basket_text,
):

    rows = []

    for line in basket_text.splitlines():

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"^([A-Z]{2,5}\s*\d{2,3})"
            r"\s+(.+)$",
            line,
            re.IGNORECASE,
        )

        if not match:
            continue

        code = normalize_course_code(
            match.group(1)
        )

        title = match.group(
            2
        ).strip()

        rows.append(
            {
                "code": code,
                "title": title,
            }
        )

    return rows


def find_elective_slot_credit(
    overview_text,
    placeholder_code,
):

    for line in overview_text.splitlines():

        if (
            placeholder_code
            in line.upper()
        ):

            numbers = re.findall(
                r"\b\d+\b",
                line,
            )

            if numbers:

                return numbers[-1]

    return None


def format_elective_basket_table(
    basket_rows,
    credit,
):

    if not basket_rows:

        return ""

    credit_label = (
        credit
        if credit
        else "Not specified"
    )

    lines = [
        "| # | Course Code | Course Title | Credit |",
        "|---|---|---|---|",
    ]

    for index, row in enumerate(
        basket_rows,
        start=1,
    ):

        lines.append(
            f"| {index} | "
            f"{row['code']} | "
            f"{row['title']} | "
            f"{credit_label} |"
        )

    return "\n".join(
        lines
    )


# =========================================================
# ALL-SEMESTER ELECTIVE RETRIEVAL
#
# "What are the elective courses and their credits?" names
# no semester at all - depth electives only exist for
# semesters 5, 6 and 7, each with its own basket. Gather all
# of them together, plus each semester's own overview-table
# row for the generic "Professional/Depth Elective Paper N"
# slot those electives fill, so the model can see the credit
# that slot carries (every option in a basket fills the same
# slot and shares its credit) instead of wrongly concluding
# credits aren't available at all.
# =========================================================

def retrieve_all_elective_documents(
    vectorstore,
):

    all_documents = get_all_documents(
        vectorstore
    )

    basket_documents = [
        document
        for document in all_documents
        if document.metadata.get(
            "document_type"
        )
        == "depth_elective_basket"
    ]

    # PDF text extraction sometimes injects a stray space
    # inside this word (e.g. "Electiv e"); a plain substring
    # check misses those rows entirely.
    elective_word_pattern = re.compile(
        r"elect\s*iv\s*e",
        re.IGNORECASE,
    )

    placeholder_documents = [
        document
        for document in all_documents
        if document.metadata.get(
            "document_type"
        )
        == "semester_curriculum"
        and elective_word_pattern.search(
            document.page_content
        )
    ]

    combined = (
        basket_documents
        + placeholder_documents
    )

    combined.sort(
        key=lambda document: (
            safe_number(
                document.metadata.get(
                    "semester",
                    0,
                )
            ),
            safe_number(
                document.metadata.get(
                    "page",
                    0,
                )
            ),
        )
    )

    return combined


# =========================================================
# ELECTIVE RETRIEVAL
# =========================================================

def retrieve_elective_documents(
    vectorstore,
    question,
    semester,
    group=None,
):

    documents = retrieve_semester_documents(
        vectorstore,
        semester,
        group=group,
    )

    if not documents:

        return []

    question_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            question.lower(),
        )
    )

    scored_documents = []

    for document in documents:

        text = document.page_content.lower()

        score = 0

        for word in question_words:

            if len(word) < 3:
                continue

            if word in text:

                score += 1

        if "elective" in text:

            score += 10

        if "pel" in text:

            score += 5

        if "depth elective" in text:

            score += 8

        # The depth-elective basket chunk is exactly what an
        # elective question needs; guarantee it wins the sort
        # even if its own wording doesn't score well on
        # keyword overlap.
        if (
            document.metadata.get(
                "document_type"
            )
            == "depth_elective_basket"
        ):

            score += 100

        scored_documents.append(
            (
                score,
                document,
            )
        )

    scored_documents.sort(
        key=lambda item: (
            -item[0],
            safe_number(
                item[1].metadata.get(
                    "page",
                    0,
                )
            ),
            safe_number(
                item[1].metadata.get(
                    "chunk_index",
                    0,
                )
            ),
        )
    )

    selected = [
        document
        for score, document
        in scored_documents
        if score > 0
    ]

    if selected:

        return selected

    return documents


# =========================================================
# COURSE SYLLABUS RETRIEVAL
# =========================================================

def retrieve_course_syllabus_documents(
    vectorstore,
    course_code,
):

    course_code = normalize_course_code(
        course_code
    )

    documents = retrieve_course_documents(
        vectorstore,
        course_code,
    )

    if not documents:

        return []

    # A course's "Topics Covered" section commonly spans
    # several chunks, and only the FIRST one contains a
    # syllabus keyword at all - later chunks are pure
    # continuation prose. Requiring every individual chunk
    # to independently repeat a keyword drops that real
    # content. Instead, only drop a chunk when it looks like
    # clear non-topic noise (a CO/PO correlation grid or a
    # references list) and carries no syllabus signal of its
    # own; keep everything else by default.

    # Only a pure CO/PO correlation grid counts as noise here.
    # Textbooks/reference material used to be excluded too,
    # but users asking for a course's topics/syllabus expect
    # its reading list included alongside them, not stripped
    # out.
    noise_markers = [
        "course articulation matrix",
        "mapping of co",
    ]

    signal_phrases = [
        "syllabus",
        "topics covered",
        "topics",
        "outline",
        "module",
        "modules",
        "unit",
        "units",
    ]

    syllabus_documents = []

    for document in documents:

        text = document.page_content.lower()

        # Only treat a chunk as noise when the marker sits
        # near the START of it - i.e. the chunk itself is
        # predominantly a references list or a CO/PO grid.
        # A chunk that's mostly real syllabus content but
        # happens to trail off into where "Text Books"
        # begins at the very end must not be discarded
        # wholesale over that tail.
        chunk_head = text[:200]

        is_noise = any(
            marker in chunk_head
            for marker in noise_markers
        )

        has_signal = any(
            phrase in text
            for phrase in signal_phrases
        )

        if is_noise and not has_signal:
            continue

        syllabus_documents.append(
            document
        )

    if syllabus_documents:

        return syllabus_documents

    return documents


# =========================================================
# SEMESTER SYLLABUS RETRIEVAL
#
# A "semester syllabus" question needs the ACTUAL detailed
# syllabus (topics covered, course outcomes, etc.), which
# lives on separate per-course pages in the PDF, not on the
# semester course-list table itself. This walks: semester
# (+ group) course list -> course codes -> each course's
# real syllabus documents.
# =========================================================

def retrieve_semester_syllabus_documents(
    vectorstore,
    semester,
    group=None,
):

    curriculum_documents = retrieve_semester_documents(
        vectorstore,
        semester,
        group=group,
    )

    if not curriculum_documents:

        return []

    course_codes = []

    for document in curriculum_documents:

        for code in extract_course_codes_from_text(
            document.page_content
        ):

            if code not in course_codes:

                course_codes.append(
                    code
                )

    syllabus_documents = []

    seen = set()

    for code in course_codes:

        code_documents = retrieve_course_syllabus_documents(
            vectorstore,
            code,
        )

        for document in code_documents:

            key = (
                document.metadata.get(
                    "page",
                    "",
                ),
                document.metadata.get(
                    "chunk_index",
                    0,
                ),
            )

            if key in seen:
                continue

            seen.add(key)

            syllabus_documents.append(
                document
            )

    if syllabus_documents:

        return syllabus_documents

    return curriculum_documents


# =========================================================
# SEMESTER GENERAL RETRIEVAL
# =========================================================

def retrieve_relevant_semester_documents(
    vectorstore,
    question,
    semester,
    group=None,
):

    if is_subject_list_question(
        question
    ):

        documents = retrieve_semester_subject_documents(
            vectorstore,
            semester,
            group=group,
        )

        if documents:

            return documents

    if is_elective_question(
        question
    ):

        documents = retrieve_elective_documents(
            vectorstore,
            question,
            semester,
            group=group,
        )

        if documents:

            return documents

    documents = retrieve_semester_documents(
        vectorstore,
        semester,
        group=group,
    )

    if not documents:

        return []

    # -----------------------------------------------------
    # For general semester questions, keep relevant
    # chunks but do not throw away the complete semester.
    # -----------------------------------------------------

    question_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            question.lower(),
        )
    )

    scored_documents = []

    for document in documents:

        text = document.page_content.lower()

        score = 0

        for word in question_words:

            if len(word) < 3:
                continue

            if word in text:

                score += 1

        if "course" in text:
            score += 2

        if "subject" in text:
            score += 2

        if "credit" in text:
            score += 1

        scored_documents.append(
            (
                score,
                document,
            )
        )

    scored_documents.sort(
        key=lambda item: (
            -item[0],
            safe_number(
                item[1].metadata.get(
                    "page",
                    0,
                )
            ),
            safe_number(
                item[1].metadata.get(
                    "chunk_index",
                    0,
                )
            ),
        )
    )

    selected = [
        document
        for score, document
        in scored_documents
        if score > 0
    ]

    if selected:

        return selected

    return documents


# =========================================================
# GENERAL RETRIEVAL
# =========================================================

def similarity_retrieval(
    vectorstore,
    question,
    k=8,
):

    try:

        return vectorstore.similarity_search(
            question,
            k=k,
        )

    except Exception as e:

        print(
            "Similarity retrieval failed:",
            repr(e),
        )

        return []


# =========================================================
# NORMAL CONTEXT
# =========================================================

def build_context(
    documents,
    max_chars=MAX_CONTEXT_CHARS,
):

    if not documents:

        return ""

    parts = []

    seen = set()

    current_length = 0

    for index, doc in enumerate(
        documents,
        start=1,
    ):

        metadata = doc.metadata or {}

        page = metadata.get(
            "page",
            "",
        )

        content = doc.page_content

        # Dedupe by content alone, not (page, chunk_index,
        # content). The same underlying text is often
        # ingested twice under different page/chunk_index
        # values (once from the per-page pass, once from the
        # semester-wide block), and counting it twice wastes
        # a large fraction of the context budget on content
        # that adds nothing new.
        content_key = content.strip()

        if content_key in seen:

            continue

        seen.add(
            content_key
        )

        source = metadata.get(
            "source",
            "ECE curriculum PDF",
        )

        semester = metadata.get(
            "semester",
            "",
        )

        course_code = metadata.get(
            "course_code",
            "",
        )

        document_type = metadata.get(
            "document_type",
            "",
        )

        part = (
            f"[DOCUMENT {index}]\n"
            f"Source: {source}\n"
            f"PDF Page: {page}\n"
            f"Semester Metadata: {semester}\n"
            f"Course Code Metadata: {course_code}\n"
            f"Document Type: {document_type}\n\n"
            f"{content}"
        )

        if (
            current_length
            + len(part)
            > max_chars
        ):

            remaining = (
                max_chars
                - current_length
            )

            if remaining > 300:

                parts.append(
                    part[:remaining]
                )

            break

        parts.append(
            part
        )

        current_length += len(
            part
        )

    return "\n\n".join(
        parts
    )


# =========================================================
# PROMPT
# =========================================================

def build_prompt(
    question,
    context,
    history,
    semester=None,
    course_code=None,
    group=None,
    semester_subject_mode=False,
    syllabus_question=False,
):

    if semester_subject_mode:

        special_instruction = """
The user is asking for the SUBJECTS/COURSES in a semester.

The application has already extracted course information
from the retrieved curriculum chunks.

You MUST use that extracted course information.

IMPORTANT:

1. Include EVERY distinct semester course present in the
   extracted course information.

2. Do NOT stop after the first two, three, four or five
   courses.

3. Do NOT invent additional courses.

4. Do NOT include prerequisite-only courses.

5. Remove duplicate course codes.

6. Preserve the exact course code, title and credit when
   they are explicitly available.

7. If a title or credit is marked "Not specified", do not
   guess it.

8. Return a clean Markdown table:

| # | Course Code | Course Title | Credit |

9. After the table, do not claim that the semester contains
   only the rows shown unless the retrieved context clearly
   establishes that it is the complete semester list.

10. Do not use outside knowledge.

11. Semesters I and II each have two curriculum tracks,
    Group 1 and Group 2, which share most courses but differ
    in a few. When the extracted course information includes
    a "Group" column, that column is authoritative: include
    every row regardless of its group value, this is expected
    and correct data, not conflicting or incomplete data.
"""

    elif course_code and syllabus_question:

        special_instruction = """
The user is asking specifically about this course's TOPICS
COVERED / SYLLABUS - not for the course's full details.

Different courses in this curriculum label their syllabus
sections differently (some use "Module 1, Module 2...",
others use "Unit I, Unit II..."). Reproduce whatever
structure and labels the retrieved context actually uses -
never invent or rename them.

Answer with ONLY the full topics/syllabus section (every
Module/Unit found, with its content) - not just the first one
or two. Do not omit any module/unit just because the answer
is getting long, and do not stop early.

Do NOT include the course title, credit, prerequisites,
course outcomes, or textbooks/reference material sections -
the user did not ask for those here, so leave them out
entirely rather than padding the answer with them.
"""

    elif course_code:

        special_instruction = """
The user is asking about a SPECIFIC COURSE. The retrieved
context is that course's own detailed curriculum entry.

Different courses in this curriculum label their syllabus
sections differently (some use "Module 1, Module 2...",
others use "Unit I, Unit II..."). Reproduce whatever
structure and labels the retrieved context actually uses -
never invent or rename them.

Give a COMPLETE answer, not a short summary. Specifically,
for every one of these sections that is present anywhere in
the retrieved context, include it in full:

1. Course title and credit.

2. Prerequisites (if listed).

3. Course Outcomes (if listed).

4. The full topics/syllabus section (every Module/Unit
   found, with its content) - not just the first one or two.

5. Textbooks and/or reference material (if listed).

Do not omit a section just because the answer is getting
long. Do not stop early. If the user asked a narrower
question (e.g. only about credits or only about
prerequisites), answer only that narrower part instead.
"""

    else:

        special_instruction = """
Answer the user's question using only the retrieved PDF
context.

If the information is present, answer it.

If it is genuinely absent from the retrieved context,
return the unavailable message.
"""

    return f"""
You are the {COLLEGE_NAME} Electronics and
Communication Engineering Department Knowledge Assistant.

The retrieved context comes from the official ECE
curriculum PDF supplied to this application.

THE PDF IS THE ONLY SOURCE OF FACTUAL INFORMATION.

GENERAL RULES:

1. Never use outside knowledge.

2. Never invent course codes.

3. Never invent course titles.

4. Never invent credits.

5. Never invent semester information.

6. Never invent electives.

7. Never invent syllabus topics.

8. Never assume a course title from a course code.

9. Preserve Group 1 / Group 2 distinctions when explicitly
   present in the PDF.

10. Do not manufacture Semester VIII information.

11. Do not claim information is unavailable when it is
    present in the retrieved context.

12. Remove duplicate information.

13. Keep answers concise but complete.

14. A course's contact-hours/credit row always lists five
    numbers in this exact order: Lecture hours, Tutorial
    hours, Practical hours, Total hours, then Credit LAST -
    e.g. "PCR 3 1 0 4 4" means L=3, T=1, P=0, Total=4,
    Credit=4, and "3 0 0 3 3" means L=3, T=0, P=0, Total=3,
    Credit=3. The Credit is whichever number comes LAST in
    that row, never the first (the first is Lecture hours,
    which is a different number from Credit and must not be
    reported as the credit value).

15. A depth/professional elective basket (a list of course
    codes a student can choose from, e.g. ECE510, ECE511...)
    does not list a credit for each individual option in the
    basket itself. Instead, that basket fills one generic
    slot in the semester's own course table (shown there as
    something like "ECE610 - Professional Elective Paper 2"
    with its own credit value) - every option in that basket
    carries that SAME credit, since the student's chosen
    course substitutes into that exact slot. If both the
    basket and its slot's credit are present in the retrieved
    context, state that shared credit for each basket course
    rather than saying credits are unavailable. Only say
    credits are unavailable if the slot's own credit truly
    isn't present in the retrieved context either.

SPECIAL INSTRUCTION:

{special_instruction}

Detected semester:
{semester if semester is not None else "Not specified"}

Detected course:
{course_code if course_code else "Not specified"}

Detected group:
{group if group else "Not specified"}

Previous conversation:
{history}

Retrieved PDF context:

{context}

User question:

{question}

Answer:
"""


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(
    question,
    vectorstore,
    llm,
):

    print(
        "DEBUG answer_question called with:",
        repr(question),
    )

    course_code = detect_course(
        question
    )

    semester = detect_semester(
        question
    )

    group = detect_group(
        question
    )

    elective_question = is_elective_question(
        question
    )

    semester_question = is_semester_question(
        question
    )

    syllabus_question = is_syllabus_question(
        question
    )

    subject_list_question = is_subject_list_question(
        question
    )

    # "the courses" / "the course" makes the generic
    # subject-list keyword check above match ANY question
    # that happens to mention a course in passing - including
    # "What are the prerequisites mentioned for the
    # courses?", which is clearly not a request for the
    # subject list itself. A specific syllabus or
    # prerequisite signal always wins that ambiguity. A
    # filter question ("which courses have tutorials?")
    # loses the same way - it's asking to identify a subset
    # by some property, not for the plain subject list.
    if (
        syllabus_question
        or is_prerequisite_question(
            question
        )
        or is_filtered_course_question(
            question
        )
    ):

        subject_list_question = False

    # =====================================================
    # FOLLOW-UP COURSE RESOLUTION
    #
    # "how many credits does it carry?" / "what are its
    # prerequisites?" after just being told about a specific
    # course has no course code of its own to detect. If this
    # question clearly needs one (credit, syllabus or
    # prerequisite question, or a plain pronoun reference like
    # "it"/"this course") but names no semester either, fall
    # back to whichever course was most recently discussed in
    # this chat.
    #
    # A PLURAL reference ("their credits", "credits of all
    # courses you mentioned") means the opposite: it's asking
    # about every course from a semester list just shown, not
    # one single course. Resolving that to a single course
    # code would silently answer for only the first course
    # mentioned - handled separately below instead.
    # =====================================================

    is_plural_reference = is_plural_followup_reference(
        question
    )

    if (
        not course_code
        and semester is None
        and not is_plural_reference
        and (
            is_credit_question(
                question
            )
            or syllabus_question
            or is_prerequisite_question(
                question
            )
            or is_course_followup_reference(
                question
            )
        )
    ):

        course_code = detect_course_from_history(
            st.session_state.chat_history
        )

    # =====================================================
    # FOLLOW-UP SEMESTER RESOLUTION
    #
    # "give me their course codes only" after just being
    # shown a semester's subject list has no semester of its
    # own to detect either - it just happens to contain the
    # word "course", which is enough to mark it as a subject-
    # list question. Fall back to whichever semester was most
    # recently discussed.
    #
    # A plural credit follow-up ("give me their credits")
    # doesn't contain any subject-list wording of its own
    # either, but it means the same thing here: resolve the
    # semester and render a credit-per-course view.
    # =====================================================

    credit_column_implied = False

    if (
        semester is None
        and not course_code
        and not elective_question
        and is_plural_reference
        and is_credit_question(
            question
        )
    ):

        semester = detect_semester_from_history(
            st.session_state.chat_history
        )

        if semester is not None:

            subject_list_question = True

            # Only imply a codes:credit view when the
            # semester had to be resolved from earlier
            # conversation - i.e. this really is a follow-up
            # ("give me their credits"). A direct question
            # that happens to use the word "their"
            # grammatically ("Semester III with their
            # credits") already names its own semester and
            # should still get the full table.
            credit_column_implied = True

    if (
        semester is None
        and not course_code
        and subject_list_question
        and not elective_question
    ):

        semester = detect_semester_from_history(
            st.session_state.chat_history
        )

    # A plural reference for anything else that still needs
    # a semester resolved (prerequisites, general questions
    # about "the courses") - not a deterministic lookup like
    # credits. Route through the semester-syllabus retrieval
    # (chains every course's own detailed entry, which is
    # where a "Pre-requisites" line actually lives) rather
    # than the generic semester path, which only sees the
    # short code/title/credit table and nothing else - or,
    # worse, falling through unscoped and silently locking
    # onto a single wrong course.
    if (
        semester is None
        and not course_code
        and not elective_question
        and is_plural_reference
        and not subject_list_question
    ):

        semester = detect_semester_from_history(
            st.session_state.chat_history
        )

        if semester is not None:

            syllabus_question = True

    # =====================================================
    # WHOLE-CURRICULUM COURSE FILTER
    #
    # "Which courses have tutorials?" / "what laboratories
    # are included?" - checked deterministically against
    # every semester's verified data rather than an LLM
    # skimming a handful of similarity-matched chunks.
    # =====================================================

    filter_property = detect_filter_property(
        question
    )

    if (
        filter_property
        and not course_code
    ):

        all_course_rows = (
            gather_all_semester_course_rows(
                vectorstore
            )
        )

        filtered_rows = (
            filter_course_rows_by_property(
                all_course_rows,
                filter_property,
            )
        )

        if not filtered_rows:

            return NOT_AVAILABLE

        return format_filtered_course_table(
            filtered_rows,
            filter_property,
        )

    # =====================================================
    # SEMESTER SUBJECT LIST
    #
    # THIS IS THE MOST IMPORTANT PATH.
    # =====================================================

    if (
        semester is not None
        and subject_list_question
        and not course_code
        and not elective_question
    ):

        documents = retrieve_semester_subject_documents(
            vectorstore,
            semester,
            group=group,
        )

        if not documents:

            return NOT_AVAILABLE

        subject_rows = extract_semester_course_rows(
            documents
        )

        if subject_rows:

            column_filter = detect_requested_columns(
                question
            )

            if (
                column_filter is None
                and credit_column_implied
            ):

                column_filter = "credit"

            return format_semester_subject_table(
                subject_rows,
                column_filter=column_filter,
            )

        context = build_semester_subject_context(
            documents,
            max_chars=7000,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = True

    # =====================================================
    # TOTAL CREDITS FOR A SEMESTER
    #
    # Answered from the PDF's own official "CREDIT UNIT OF
    # THE PROGRAM" summary table, looked up directly rather
    # than asking the LLM to sum a dense per-course table
    # itself (which it does unreliably).
    # =====================================================

    elif (
        semester is not None
        and is_total_credit_question(
            question
        )
        and not course_code
    ):

        credit_summary_documents = (
            exact_metadata_retrieval(
                vectorstore=vectorstore,
                semester=semester,
                document_type="credit_summary",
            )
        )

        if not credit_summary_documents:

            return NOT_AVAILABLE

        return credit_summary_documents[
            0
        ].page_content

    # =====================================================
    # SEMESTER SYLLABUS (NO SPECIFIC COURSE GIVEN)
    #
    # e.g. "semester 1 syllabus of group 2"
    # =====================================================

    elif (
        semester is not None
        and syllabus_question
        and not course_code
    ):

        documents = retrieve_semester_syllabus_documents(
            vectorstore,
            semester,
            group=group,
        )

        if not documents:

            return NOT_AVAILABLE

        context = build_context(
            documents,
            max_chars=11000,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # SINGLE COURSE CREDIT
    #
    # Looked up directly from the verified overview-table
    # data rather than asked of the LLM, which is unreliable
    # at finding a single number buried in dense syllabus
    # text.
    # =====================================================

    elif (
        course_code
        and is_credit_question(
            question
        )
        and not syllabus_question
    ):

        course_row = find_course_row(
            vectorstore,
            course_code,
        )

        if course_row is None:

            return NOT_AVAILABLE

        title = (
            course_row["title"]
            if course_row["title"]
            else "Not specified"
        )

        credit = (
            course_row["credit"]
            if course_row["credit"]
            else "Not specified"
        )

        return (
            f"{course_row['code']} "
            f"({title}) carries "
            f"{credit} credits."
        )

    # =====================================================
    # COURSE + SYLLABUS
    # =====================================================

    elif (
        course_code
        and syllabus_question
    ):

        documents = retrieve_course_syllabus_documents(
            vectorstore,
            course_code,
        )

        # A course with many modules (e.g. ECC304 has 12)
        # can easily exceed 8500 characters of real content;
        # a tighter budget here was silently cutting the
        # syllabus off partway through.
        context = build_context(
            documents,
            max_chars=15000,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # COURSE
    # =====================================================

    elif course_code:

        documents = retrieve_course_documents(
            vectorstore,
            course_code,
        )

        context = build_context(
            documents,
            max_chars=15000,
        )

        print(
            "DEBUG COURSE branch:",
            "course_code=",
            course_code,
            "| docs=",
            len(documents),
            "| context_chars=",
            len(context),
        )

        if not context.strip():

            print(
                "DEBUG: returning NOT_AVAILABLE "
                "from COURSE branch (empty context)"
            )

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # ELECTIVE - ALL SEMESTERS
    #
    # "What are the elective courses and their credits?"
    # names no semester at all, unlike a semester-scoped
    # elective question. Depth electives only exist for
    # semesters 5-7, so gather every basket together rather
    # than falling through to an unscoped generic search.
    # =====================================================

    elif (
        elective_question
        and semester is None
    ):

        documents = retrieve_all_elective_documents(
            vectorstore
        )

        context = build_context(
            documents,
            max_chars=12000,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # ELECTIVE - SPECIFIC SEMESTER (DIRECT RENDER)
    #
    # The basket is already a clean, verified code/title
    # list, and every option shares its slot's credit
    # (see rule 15). Render it directly rather than asking
    # the LLM to reproduce a list it already has.
    # =====================================================

    elif (
        elective_question
        and semester is not None
        and semester
        in ELECTIVE_PLACEHOLDER_CODE_BY_SEMESTER
    ):

        basket_documents = (
            exact_metadata_retrieval(
                vectorstore=vectorstore,
                semester=semester,
                document_type="depth_elective_basket",
            )
        )

        basket_rows = []

        for basket_document in basket_documents:

            basket_rows.extend(
                parse_elective_basket_rows(
                    basket_document.page_content
                )
            )

        if not basket_rows:

            return NOT_AVAILABLE

        overview_documents = (
            exact_metadata_retrieval(
                vectorstore=vectorstore,
                semester=semester,
                document_type="semester_curriculum",
            )
        )

        placeholder_code = (
            ELECTIVE_PLACEHOLDER_CODE_BY_SEMESTER[
                semester
            ]
        )

        credit = None

        for overview_document in overview_documents:

            credit = find_elective_slot_credit(
                overview_document.page_content,
                placeholder_code,
            )

            if credit:
                break

        return format_elective_basket_table(
            basket_rows,
            credit,
        )

    # =====================================================
    # ELECTIVE
    # =====================================================

    elif (
        elective_question
        and semester is not None
    ):

        documents = retrieve_elective_documents(
            vectorstore,
            question,
            semester,
            group=group,
        )

        context = build_context(
            documents,
            max_chars=8500,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # SEMESTER QUESTION
    # =====================================================

    elif (
        semester is not None
        and semester_question
    ):

        documents = retrieve_relevant_semester_documents(
            vectorstore,
            question,
            semester,
            group=group,
        )

        context = build_context(
            documents,
            max_chars=8500,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # GENERAL QUESTION
    # =====================================================

    else:

        documents = similarity_retrieval(
            vectorstore,
            question,
            k=8,
        )

        context = build_context(
            documents,
            max_chars=8500,
        )

        if not context.strip():

            return NOT_AVAILABLE

        semester_subject_mode = False

    # =====================================================
    # CHAT HISTORY
    # =====================================================

    # A prior FAILED answer (rate-limited, "not available",
    # etc.) must never be fed back to the model as context -
    # it will anchor on and simply repeat its own past
    # failure instead of trying again from the (correct)
    # freshly retrieved context.
    raw_history = st.session_state.chat_history

    usable_history = []

    index = 0

    while index < len(raw_history):

        item = raw_history[index]

        next_item = (
            raw_history[index + 1]
            if index + 1 < len(raw_history)
            else None
        )

        # Drop the WHOLE exchange (question + failed
        # answer), not just the failed reply - a dangling
        # duplicate of the same question with no visible
        # answer is just as likely to confuse the model as
        # the failure text itself.
        if (
            item.get("role") == "user"
            and next_item is not None
            and next_item.get("role")
            == "assistant"
            and is_failure_response(
                next_item.get(
                    "content",
                    "",
                )
            )
        ):

            index += 2
            continue

        usable_history.append(
            item
        )

        index += 1

    history_items = []

    for item in usable_history[-2:]:

        history_items.append(
            f"{item['role']}: "
            f"{item['content'][:400]}"
        )

    history = "\n".join(
        history_items
    )

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = build_prompt(
        question=question,
        context=context,
        history=history,
        semester=semester,
        course_code=course_code,
        group=group,
        semester_subject_mode=semester_subject_mode,
        syllabus_question=syllabus_question,
    )

    # =====================================================
    # LLM
    # =====================================================

    print(
        "DEBUG about to call LLM, prompt_chars=",
        len(prompt),
    )

    try:

        response = llm.invoke(
            prompt
        )

        print(
            "DEBUG LLM call succeeded, "
            "response_chars=",
            len(
                getattr(
                    response,
                    "content",
                    "",
                )
                or ""
            ),
        )

    except Exception as e:

        print(
            "LLM error:",
            repr(e),
        )

        if (
            "413" in str(e)
            or "Request too large" in str(e)
            or "rate_limit_exceeded" in str(e)
        ):

            try:

                smaller_context = context[
                    :4500
                ]

                smaller_prompt = build_prompt(
                    question=question,
                    context=smaller_context,
                    history="",
                    semester=semester,
                    course_code=course_code,
                    group=group,
                    semester_subject_mode=semester_subject_mode,
                    syllabus_question=syllabus_question,
                )

                response = llm.invoke(
                    smaller_prompt
                )

            except Exception as retry_error:

                print(
                    "LLM retry error:",
                    repr(retry_error),
                )

                if (
                    "rate_limit_exceeded"
                    in str(retry_error)
                    or "tokens per day"
                    in str(retry_error)
                ):

                    return (
                        "The Groq API's daily usage limit "
                        "for this account has been reached. "
                        "This is not a problem with your "
                        "question - please wait a while and "
                        "try again, or ask the app's "
                        "administrator to check the Groq "
                        "billing/quota page."
                    )

                return (
                    "The retrieved information is too large "
                    "for the current Groq model limit. "
                    "Please ask a more specific question."
                )

        else:

            return (
                "The language model could not "
                "generate a response.\n\n"
                f"Error: {str(e)}"
            )

    # =====================================================
    # ANSWER
    # =====================================================

    answer = getattr(
        response,
        "content",
        "",
    )

    if not answer:

        return NOT_AVAILABLE

    return answer.strip()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "About Bot"
    )

    st.markdown(
        "### NIT Durgapur ECE"
    )

    st.markdown(
        """
        This chatbot answers questions using the
        ECE curriculum PDF stored in the project's
        data folder.
        """
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Data Source"
    )

    st.markdown(
        "`ECE_Curriculum_2023_Onwards.pdf`"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Supported Questions"
    )

    st.markdown(
        """
        - Semester subjects
        - Course details
        - Course codes
        - Electives
        - Laboratories
        - Syllabus
        - Course outcomes
        - Prerequisites
        - Credits
        """
    )

    st.markdown(
        "---"
    )

    if st.button(
        "Clear Chat"
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN PAGE
# =========================================================

st.title(
    "📡 NIT Durgapur ECE "
    "Department Knowledge Assistant"
)

st.write(
    "Ask questions about the ECE curriculum, "
    "courses, semesters, electives and syllabus."
)


# =========================================================
# VECTOR DATABASE
# =========================================================

try:

    vectorstore = setup_vectorstore()

except Exception as e:

    st.error(
        "Could not load the vector database."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# LLM
# =========================================================

try:

    llm = get_llm()

except Exception as e:

    st.error(
        "Could not initialize the Groq model."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# INPUT
# =========================================================

question = st.chat_input(
    "Ask about the NIT Durgapur ECE curriculum..."
)


if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the ECE knowledge base..."
        ):

            try:

                answer = answer_question(
                    question=question,
                    vectorstore=vectorstore,
                    llm=llm,
                )

            except Exception as e:

                print(
                    "Chatbot error:",
                    repr(e),
                )

                answer = (
                    "An error occurred while "
                    "searching the knowledge base.\n\n"
                    f"Error: {str(e)}"
                )

        st.markdown(
            answer
        )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )