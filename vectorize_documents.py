import os
import re
import shutil

from PyPDF2 import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# PATHS
# =========================================================

WORKING_DIR = os.path.dirname(
    os.path.realpath(__file__)
)

PDF_PATH = os.path.join(
    WORKING_DIR,
    "data",
    "ECE_Curriculum_2023_Onwards.pdf"
)

VECTOR_DB_DIR = os.path.join(
    WORKING_DIR,
    "vector_db_dir"
)


# =========================================================
# COURSE CODE PATTERN
# =========================================================

COURSE_CODE_PATTERN = re.compile(
    r"\b[A-Z]{2,5}\s*\d{2,3}\b",
    re.IGNORECASE
)


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")

    # Strip the repeating "Page X of Y" header, which
    # otherwise gets misread as a course code (e.g. "of 140"
    # matches [A-Z]{2,5}\s*\d{2,3}) and breaks per-page
    # single-course-code detection below.
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# EXTRACT COURSE CODES
# =========================================================

def extract_course_codes(text):

    codes = COURSE_CODE_PATTERN.findall(text)

    cleaned = []

    for code in codes:

        code = re.sub(
            r"\s+",
            "",
            code
        ).upper()

        if code not in cleaned:
            cleaned.append(code)

    return sorted(cleaned)


# =========================================================
# SEMESTER PAGE RANGES
# =========================================================

SEMESTER_PAGE_RANGES = {

    3: (30, 40),

    4: (41, 51),

    5: (52, 86),

    6: (87, 113),

    7: (114, 139),

}


# =========================================================
# LOAD PDF
# =========================================================

if not os.path.exists(PDF_PATH):

    raise FileNotFoundError(
        f"\nPDF not found:\n{PDF_PATH}\n\n"
        "Make sure the PDF is inside the data folder."
    )


print()
print("=" * 60)
print("LOADING ECE CURRICULUM PDF")
print("=" * 60)

reader = PdfReader(PDF_PATH)

print(
    f"PDF: {os.path.basename(PDF_PATH)}"
)

print(
    f"Total pages: {len(reader.pages)}"
)


# =========================================================
# EXTRACT ALL PAGES
# =========================================================

page_texts = []

for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = clean_text(
        page.extract_text() or ""
    )

    page_texts.append(
        {
            "page": page_number,
            "text": text
        }
    )


# =========================================================
# CREATE DOCUMENTS
# =========================================================

documents = []


print()
print("=" * 60)
print("CREATING PDF DOCUMENTS")
print("=" * 60)


for page in page_texts:

    text = page["text"]

    if not text:
        continue

    page_number = page["page"]

    course_codes = extract_course_codes(
        text
    )

    if len(course_codes) == 1:

        course_code = course_codes[0]

        documents.append(
            Document(
                page_content=text,

                metadata={
                    "source":
                        os.path.basename(
                            PDF_PATH
                        ),

                    "page":
                        page_number,

                    "semester":
                        0,

                    "course_code":
                        course_code,

                    "document_type":
                        "course_document"
                }
            )
        )

    else:

        documents.append(
            Document(
                page_content=text,

                metadata={
                    "source":
                        os.path.basename(
                            PDF_PATH
                        ),

                    "page":
                        page_number,

                    "semester":
                        0,

                    "course_code":
                        "GENERAL",

                    "document_type":
                        "general_document"
                }
            )
        )


# =========================================================
# SEMESTER III-VIII OVERVIEW TABLES
#
# Pages 5-6 contain a clean per-semester course-list table
# for Semesters III-VIII (Sl.No/Code/Subject/L/T/S/C/H),
# mirroring the Group 1/2 tables on pages 2-3 for I/II. This
# is a far more reliable source for "what are the subjects
# in semester N" than mining it out of the large per-semester
# detailed-syllabus block created further below.
# =========================================================

print()
print("=" * 60)
print("CREATING SEMESTER III-VIII OVERVIEW TABLE DOCUMENTS")
print("=" * 60)

overview_source = (
    page_texts[4]["text"]
    + "\n\n"
    + page_texts[5]["text"]
)

ROMAN_TO_SEMESTER = {
    "VIII": 8,
    "VII": 7,
    "VI": 6,
    "V": 5,
    "IV": 4,
    "III": 3,
}

overview_header_pattern = re.compile(
    r"Semester\s*-\s*(VIII|VII|VI|V|IV|III)\b",
    re.IGNORECASE,
)

overview_matches = list(
    overview_header_pattern.finditer(
        overview_source
    )
)

for index, match in enumerate(overview_matches):

    roman = match.group(1).upper()

    semester_number = ROMAN_TO_SEMESTER[roman]

    start = match.start()

    end = (
        overview_matches[index + 1].start()
        if index + 1 < len(overview_matches)
        else len(overview_source)
    )

    block_text = overview_source[start:end]

    stop_marker = block_text.find(
        "CREDIT UNIT OF THE PROGRAM"
    )

    if stop_marker != -1:

        block_text = block_text[:stop_marker]

    block_text = block_text.strip()

    if not block_text:
        continue

    documents.append(
        Document(
            page_content=block_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    5,

                "semester":
                    semester_number,

                "group":
                    0,

                "course_code":
                    f"SEMESTER_{semester_number}",

                "document_type":
                    "semester_curriculum"
            }
        )
    )

    print(
        f"Semester {semester_number} overview "
        f"table: added"
    )


# =========================================================
# SEMESTER I & II CURRICULUM
# =========================================================

print()
print("=" * 60)
print("CREATING FIRST AND SECOND SEMESTER DOCUMENTS")
print("=" * 60)


# Pages 2 and 3 contain the complete first and second
# semester curriculum for Group 1 and Group 2.

page2 = page_texts[1]["text"]
page3 = page_texts[2]["text"]


# ---------------------------------------------------------
# GROUP 1
# ---------------------------------------------------------

group1_sem1_match = re.search(
    r"FIRST\s+SEMESTER(.*?)(?=SECOND\s+SEMESTER)",
    page2,
    re.IGNORECASE | re.DOTALL
)

group1_sem2_match = re.search(
    r"SECOND\s+SEMESTER(.*)",
    page2,
    re.IGNORECASE | re.DOTALL
)


group1_sem1_text = ""

group1_sem2_text = ""


if group1_sem1_match:

    group1_sem1_text = (
        "GROUP 1\nFIRST SEMESTER\n"
        + group1_sem1_match.group(1).strip()
    )


if group1_sem2_match:

    group1_sem2_text = (
        "GROUP 1\nSECOND SEMESTER\n"
        + group1_sem2_match.group(1).strip()
    )


# ---------------------------------------------------------
# GROUP 2
# ---------------------------------------------------------

group2_sem1_match = re.search(
    r"FIRST\s+SEMESTER(.*?)(?=SECOND\s+SEMESTER)",
    page3,
    re.IGNORECASE | re.DOTALL
)

group2_sem2_match = re.search(
    r"SECOND\s+SEMESTER(.*)",
    page3,
    re.IGNORECASE | re.DOTALL
)


group2_sem1_text = ""

group2_sem2_text = ""


if group2_sem1_match:

    group2_sem1_text = (
        "GROUP 2\nFIRST SEMESTER\n"
        + group2_sem1_match.group(1).strip()
    )


if group2_sem2_match:

    group2_sem2_text = (
        "GROUP 2\nSECOND SEMESTER\n"
        + group2_sem2_match.group(1).strip()
    )


# ---------------------------------------------------------
# ADD GROUP 1 SEMESTER 1
# ---------------------------------------------------------

if group1_sem1_text:

    documents.append(
        Document(
            page_content=group1_sem1_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    2,

                "semester":
                    1,

                "group":
                    1,

                "course_code":
                    "SEMESTER_I",

                "document_type":
                    "semester_curriculum"
            }
        )
    )

    print(
        "Semester 1 Group 1: added"
    )

else:

    print(
        "Semester 1 Group 1: NOT FOUND"
    )


# ---------------------------------------------------------
# ADD GROUP 2 SEMESTER 1
# ---------------------------------------------------------

if group2_sem1_text:

    documents.append(
        Document(
            page_content=group2_sem1_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    3,

                "semester":
                    1,

                "group":
                    2,

                "course_code":
                    "SEMESTER_I",

                "document_type":
                    "semester_curriculum"
            }
        )
    )

    print(
        "Semester 1 Group 2: added"
    )

else:

    print(
        "Semester 1 Group 2: NOT FOUND"
    )


# ---------------------------------------------------------
# ADD GROUP 1 SEMESTER 2
# ---------------------------------------------------------

if group1_sem2_text:

    documents.append(
        Document(
            page_content=group1_sem2_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    2,

                "semester":
                    2,

                "group":
                    1,

                "course_code":
                    "SEMESTER_II",

                "document_type":
                    "semester_curriculum"
            }
        )
    )

    print(
        "Semester 2 Group 1: added"
    )

else:

    print(
        "Semester 2 Group 1: NOT FOUND"
    )


# ---------------------------------------------------------
# ADD GROUP 2 SEMESTER 2
# ---------------------------------------------------------

if group2_sem2_text:

    documents.append(
        Document(
            page_content=group2_sem2_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    3,

                "semester":
                    2,

                "group":
                    2,

                "course_code":
                    "SEMESTER_II",

                "document_type":
                    "semester_curriculum"
            }
        )
    )

    print(
        "Semester 2 Group 2: added"
    )

else:

    print(
        "Semester 2 Group 2: NOT FOUND"
    )


# =========================================================
# DEPTH ELECTIVE COURSE BASKETS
#
# Pages 7-8 list the depth elective options per semester
# (5th/6th/7th) but sit outside SEMESTER_PAGE_RANGES, so
# without this they never get semester metadata and are
# invisible to semester-filtered retrieval.
# =========================================================

print()
print("=" * 60)
print("CREATING DEPTH ELECTIVE BASKET DOCUMENTS")
print("=" * 60)

basket_source = (
    page_texts[6]["text"]
    + "\n\n"
    + page_texts[7]["text"]
)

# PDF text extraction sometimes splits "Semester" with a
# stray space (e.g. "Seme ster"); normalize before matching.
basket_source = re.sub(
    r"Seme\s+ster",
    "Semester",
    basket_source,
)

basket_header_pattern = re.compile(
    r"\d+(?:st|nd|rd|th)\s+Semester",
    re.IGNORECASE,
)

basket_matches = list(
    basket_header_pattern.finditer(basket_source)
)

for index, match in enumerate(basket_matches):

    semester_number = int(
        re.match(r"\d+", match.group(0)).group(0)
    )

    start = match.start()

    end = (
        basket_matches[index + 1].start()
        if index + 1 < len(basket_matches)
        else len(basket_source)
    )

    basket_text = basket_source[
        start:end
    ].strip()

    if not basket_text:
        continue

    basket_text = (
        "DEPTH ELECTIVE COURSE BASKET "
        "(Professional / Depth Elective options)\n"
        + basket_text
    )

    documents.append(
        Document(
            page_content=basket_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    7,

                "semester":
                    semester_number,

                "group":
                    0,

                "course_code":
                    f"SEMESTER_{semester_number}_ELECTIVES",

                "document_type":
                    "depth_elective_basket"
            }
        )
    )

    print(
        f"Depth elective basket for semester "
        f"{semester_number}: added"
    )


# =========================================================
# OFFICIAL TOTAL CREDIT SUMMARY
#
# Page 6 has an authoritative "CREDIT UNIT OF THE PROGRAM"
# table giving the official total credits per semester. This
# is far more reliable for "total credits" questions than
# asking an LLM to sum a dense per-course table itself.
# =========================================================

print()
print("=" * 60)
print("CREATING CREDIT SUMMARY DOCUMENTS")
print("=" * 60)

credit_unit_match = re.search(
    r"Credit\s+Unit\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
    page_texts[5]["text"],
)

if credit_unit_match:

    credit_values = [
        int(value)
        for value in credit_unit_match.groups()
    ]

    # Order in the table: I+II, III, IV, V, VI, VII, VIII, TOTAL
    combined_i_ii_credit = credit_values[0]

    per_semester_credit = {
        3: credit_values[1],
        4: credit_values[2],
        5: credit_values[3],
        6: credit_values[4],
        7: credit_values[5],
        8: credit_values[6],
    }

    for (
        semester_number,
        total_credit,
    ) in per_semester_credit.items():

        documents.append(
            Document(
                page_content=(
                    f"Official total credit for "
                    f"Semester {semester_number}: "
                    f"{total_credit} credits.\n"
                    "(Source: CREDIT UNIT OF THE "
                    "PROGRAM table.)"
                ),

                metadata={
                    "source":
                        os.path.basename(
                            PDF_PATH
                        ),

                    "page":
                        6,

                    "semester":
                        semester_number,

                    "group":
                        0,

                    "course_code":
                        f"SEMESTER_{semester_number}"
                        f"_CREDIT_SUMMARY",

                    "document_type":
                        "credit_summary"
                }
            )
        )

        print(
            f"Semester {semester_number} credit "
            f"summary: {total_credit} credits"
        )

    # The official table only reports Semester I and
    # II as a COMBINED figure, not split individually.
    # Tag the same explanation under both semester
    # numbers so a lookup for either one finds it.

    combined_text = (
        "The official CREDIT UNIT OF THE PROGRAM table "
        "reports Semester I and Semester II ONLY as a "
        "COMBINED total, not split individually: "
        f"{combined_i_ii_credit} credits for Semester I "
        "and II together."
    )

    for semester_number in [1, 2]:

        documents.append(
            Document(
                page_content=combined_text,

                metadata={
                    "source":
                        os.path.basename(
                            PDF_PATH
                        ),

                    "page":
                        6,

                    "semester":
                        semester_number,

                    "group":
                        0,

                    "course_code":
                        "SEMESTER_I_II"
                        "_CREDIT_SUMMARY",

                    "document_type":
                        "credit_summary"
                }
            )
        )

    print(
        "Semester I+II combined credit summary: "
        f"{combined_i_ii_credit} credits"
    )

else:

    print(
        "WARNING: Could not parse the CREDIT UNIT OF "
        "THE PROGRAM table."
    )


print()
print(
    "Semester 1 Group 1 text length:",
    len(group1_sem1_text)
)

print(
    "Semester 1 Group 2 text length:",
    len(group2_sem1_text)
)

print(
    "Semester 2 Group 1 text length:",
    len(group1_sem2_text)
)

print(
    "Semester 2 Group 2 text length:",
    len(group2_sem2_text)
)


# =========================================================
# SEMESTER III-VII CURRICULUM
# =========================================================

print()
print("=" * 60)
print("CREATING SEMESTER III-VII DOCUMENTS")
print("=" * 60)


for semester, page_range in SEMESTER_PAGE_RANGES.items():

    start_page, end_page = page_range

    semester_pages = []

    for page in page_texts:

        page_number = page["page"]

        if (
            page_number >= start_page
            and
            page_number <= end_page
        ):

            if page["text"]:

                semester_pages.append(
                    page
                )


    if not semester_pages:

        print(
            f"Semester {semester}: "
            f"No content found"
        )

        continue


    semester_text = "\n\n".join(
        page["text"]
        for page in semester_pages
    )


    documents.append(
        Document(
            page_content=semester_text,

            metadata={
                "source":
                    os.path.basename(
                        PDF_PATH
                    ),

                "page":
                    start_page,

                "semester":
                    semester,

                "group":
                    0,

                "course_code":
                    f"SEMESTER_{semester}",

                # Unlike the Semester I/II blocks (which
                # are only a short course-list table), this
                # block spans the full "DETAILED SYLLABUS"
                # pages for the semester, so it is tagged
                # differently from "semester_curriculum".
                "document_type":
                    "semester_full_content"
            }
        )
    )


    print(
        f"Semester {semester}: "
        f"{len(semester_pages)} pages added"
    )


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

print()
print("=" * 60)
print("SPLITTING DOCUMENTS")
print("=" * 60)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


chunks = splitter.split_documents(
    documents
)


print(
    f"Total chunks created: "
    f"{len(chunks)}"
)


# =========================================================
# ADD CHUNK INDEX
# =========================================================

for index, chunk in enumerate(
    chunks
):

    chunk.metadata[
        "chunk_index"
    ] = index


# =========================================================
# DELETE OLD VECTOR DATABASE
# =========================================================

if os.path.exists(
    VECTOR_DB_DIR
):

    print()
    print(
        "Removing old vector database..."
    )

    shutil.rmtree(
        VECTOR_DB_DIR
    )

    print(
        "Old vector database removed."
    )


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print()
print("=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

embeddings = HuggingFaceEmbeddings()


# =========================================================
# CREATE CHROMA
# =========================================================

print()
print("=" * 60)
print("CREATING VECTOR DATABASE")
print("=" * 60)


Chroma.from_documents(
    documents=chunks,

    embedding=embeddings,

    persist_directory=
        VECTOR_DB_DIR
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 60)
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("=" * 60)

print(
    f"Total documents: "
    f"{len(documents)}"
)

print(
    f"Total chunks: "
    f"{len(chunks)}"
)


print()
print(
    "Semester curriculum documents:"
)


for semester in range(1, 8):

    count = sum(
        1
        for chunk in chunks

        if chunk.metadata.get(
            "document_type"
        )
        in (
            "semester_curriculum",
            "semester_full_content",
        )

        and chunk.metadata.get(
            "semester"
        )
        == semester
    )

    print(
        f"Semester {semester}: "
        f"{count} chunks"
    )


print()
print(
    "Semester 8: "
    "No separate curriculum section found"
)


print()
print(
    "Vector database:"
)

print(
    VECTOR_DB_DIR
)

print()
print("DONE.")

