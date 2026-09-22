import os
import pickle
import re

import numpy as np
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    redirect
)

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET = os.path.join(
    BASE_DIR,
    "updated_industrial_nlp_government_dataset.xlsx"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    ),
    static_folder=os.path.join(
        BASE_DIR,
        "static"
    )
)


# ============================================================
# LOAD DATASET
# ============================================================

industries = pd.read_excel(
    DATASET,
    sheet_name="Industries"
).fillna("")


approvals = pd.read_excel(
    DATASET,
    sheet_name="Approvals"
).fillna("")


approval_docs = pd.read_excel(
    DATASET,
    sheet_name="Approval_Documents"
).fillna("")


schemes = pd.read_excel(
    DATASET,
    sheet_name="Schemes"
).fillna("")


loan_schemes = pd.read_excel(
    DATASET,
    sheet_name="Loan_Schemes_Updated"
).fillna("")


profile_fields = pd.read_excel(
    DATASET,
    sheet_name="User_Profile_Fields"
).fillna("")


# ============================================================
# LOAD MODEL.PKL
# ============================================================

with open(
    MODEL_PATH,
    "rb"
) as f:

    bundle = pickle.load(f)


# Actual keys from your model.pkl:
# vectorizer
# matrix
# records

vectorizer = bundle[
    "vectorizer"
]

knowledge_matrix = bundle[
    "matrix"
]

knowledge_records = bundle[
    "records"
]


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def num(
    value,
    default=0.0
):

    try:

        return float(
            str(value)
            .replace(",", "")
            .replace("₹", "")
            .strip()
        )

    except Exception:

        return default


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k=8
):

    query = clean(
        query
    )

    if not query:

        return []

    q = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        q,
        knowledge_matrix
    ).ravel()

    indexes = np.argsort(
        scores
    )[::-1][:top_k]

    result = []

    for i in indexes:

        if scores[i] <= 0:

            continue

        row = dict(
            knowledge_records[
                int(i)
            ]
        )

        row["score"] = round(
            float(
                scores[i]
            ),
            4
        )

        result.append(
            row
        )

    return result


# ============================================================
# INDUSTRY IDENTIFICATION
# ============================================================

def identify_industry(
    query
):

    q = clean(
        query
    ).lower()

    best = None
    best_score = -1

    for _, row in industries.iterrows():

        name = clean(
            row[
                "industry_name"
            ]
        )

        tokens = [
            x
            for x in re.findall(
                r"[a-z0-9]+",
                name.lower()
            )
            if len(x) > 2
        ]

        score = sum(
            1
            for token in tokens
            if token in q
        )

        if name.lower() in q:

            score += 5

        if score > best_score:

            best_score = score
            best = row

    # Retrieval fallback

    if best_score <= 0:

        retrieved = retrieve(
            query,
            12
        )

        for r in retrieved:

            if (
                r.get(
                    "record_type"
                )
                == "industry"
            ):

                hit = industries[
                    industries[
                        "industry_id"
                    ]
                    .astype(str)
                    ==
                    str(
                        r.get(
                            "industry_id"
                        )
                    )
                ]

                if not hit.empty:

                    return hit.iloc[0]

    return best


# ============================================================
# INDUSTRY QUERY
# ============================================================

def industry_query(
    profile
):

    return " ".join([

        clean(
            profile.get(
                "business_type"
            )
        ),

        clean(
            profile.get(
                "business_name"
            )
        ),

        clean(
            profile.get(
                "description"
            )
        ),

        clean(
            profile.get(
                "business_stage"
            )
        ),

        clean(
            profile.get(
                "location_type"
            )
        ),

        clean(
            profile.get(
                "district"
            )
        )

    ]).strip()


# ============================================================
# APPROVALS
# ============================================================

def select_approvals(
    profile,
    industry
):

    if industry is None:

        return []

    iid = str(
        industry[
            "industry_id"
        ]
    )

    rows = approvals[
        approvals[
            "industry_id"
        ]
        .astype(str)
        ==
        iid
    ].copy()

    if rows.empty:

        rows = approvals[
            approvals[
                "sector"
            ]
            .astype(str)
            .str.lower()
            ==
            str(
                industry[
                    "sector"
                ]
            ).lower()
        ].copy()

    description = clean(
        profile.get(
            "description"
        )
    ).lower()

    result = []
    seen = set()

    for _, r in rows.iterrows():

        name = clean(
            r.get(
                "approval_name"
            )
        )

        if (
            not name
            or name in seen
        ):

            continue

        applicability = clean(
            r.get(
                "applicability"
            )
        ).lower()

        trigger = clean(
            r.get(
                "trigger_condition"
            )
        ).lower()

        context = {

            "industrial_establishment":
                True,

            "registered_business":
                True,

            "boiler_installed":
                "boiler"
                in description,

            "water_intensive":
                any(
                    x in description
                    for x in [
                        "water",
                        "dairy",
                        "food",
                        "processing"
                    ]
                )
        }

        trigger_ok = True

        if "=" in trigger:

            key, value = trigger.split(
                "=",
                1
            )

            trigger_ok = (
                context.get(
                    key.strip(),
                    False
                )
                ==
                (
                    value.strip()
                    == "true"
                )
            )

        if (
            applicability
            == "mandatory"
            or trigger_ok
        ):

            seen.add(
                name
            )

            result.append({

                "name":
                    name,

                "level":
                    clean(
                        r.get(
                            "applicability"
                        )
                    )
                    or "Conditional",

                "department":
                    clean(
                        r.get(
                            "department_id"
                        )
                    ),

                "fee":
                    num(
                        r.get(
                            "_fee_inr"
                        )
                    ),

                "timeline":
                    int(
                        num(
                            r.get(
                                "_timeline_days"
                            )
                        )
                    ),

                "application_type":
                    clean(
                        r.get(
                            "application_type"
                        )
                    ),

                "submission_mode":
                    clean(
                        r.get(
                            "submission_mode"
                        )
                    ),

                "trigger":
                    clean(
                        r.get(
                            "trigger_condition"
                        )
                    ),

                "approval_record_id":
                    clean(
                        r.get(
                            "approval_record_id"
                        )
                    )
            })

        if len(result) >= 12:

            break

    return result


# ============================================================
# DOCUMENTS
# ============================================================

def documents_for_approvals(
    selected
):

    ids = {
        x[
            "approval_record_id"
        ]
        for x in selected
    }

    if not ids:

        return []

    docs = approval_docs[
        approval_docs[
            "approval_record_id"
        ]
        .astype(str)
        .isin(ids)
    ]

    result = []
    seen = set()

    for _, r in docs.iterrows():

        item = {

            "document_id":
                clean(
                    r[
                        "document_id"
                    ]
                ),

            "requirement":
                clean(
                    r[
                        "requirement_level"
                    ]
                ),

            "format":
                clean(
                    r[
                        "format_hint"
                    ]
                )
        }

        key = tuple(
            item.values()
        )

        if key not in seen:

            seen.add(
                key
            )

            result.append(
                item
            )

        if len(result) >= 18:

            break

    return result


# ============================================================
# SCHEMES
# ============================================================

def select_schemes(
    profile,
    industry
):

    query = industry_query(
        profile
    ).lower()

    sector = (

        clean(
            industry[
                "sector"
            ]
        ).lower()

        if industry is not None

        else ""
    )

    results = []


    # Loan/support layer

    for _, r in loan_schemes.iterrows():

        hay = " ".join(

            str(
                r.get(
                    c,
                    ""
                )
            )

            for c in [

                "scheme_name",
                "scope",
                "eligible_profiles",
                "best_for"

            ]

        ).lower()

        score = 0

        for token in [

            "dairy",
            "food",
            "processing",
            "manufacturing",
            "msme",
            "startup",
            "farmer",
            "micro"

        ]:

            if (
                token in query
                and token in hay
            ):

                score += 2

        if (
            "dairy" in query
            and "dairy" in hay
        ):

            score += 5

        if (
            "food" in query
            and "food" in hay
        ):

            score += 3

        if score:

            results.append({

                "name":
                    clean(
                        r[
                            "scheme_name"
                        ]
                    ),

                "provider":
                    clean(
                        r[
                            "provider"
                        ]
                    ),

                "support":
                    clean(
                        r[
                            "financial_support"
                        ]
                    ),

                "eligibility":
                    clean(
                        r[
                            "eligible_profiles"
                        ]
                    ),

                "application":
                    clean(
                        r[
                            "application_method"
                        ]
                    ),

                "charges":
                    clean(
                        r[
                            "charges"
                        ]
                    ),

                "best_for":
                    clean(
                        r[
                            "best_for"
                        ]
                    ),

                "source":
                    clean(
                        r[
                            "source_url"
                        ]
                    ),

                "score":
                    score
            })


    # Policy layer

    for _, r in schemes.iterrows():

        hay = (

            clean(
                r[
                    "scheme_name"
                ]
            )

            + " "

            + clean(
                r[
                    "sector"
                ]
            )

        ).lower()

        if (
            sector
            and (
                sector in hay
                or
                clean(
                    r[
                        "sector"
                    ]
                ).lower()
                in query
            )
        ):

            results.append({

                "name":
                    clean(
                        r[
                            "scheme_name"
                        ]
                    ),

                "provider":
                    "Maharashtra government policy layer",

                "support":
                    "Sector support / policy framework; verify current eligibility and benefits.",

                "eligibility":
                    "Depends on the applicable policy and current guidelines.",

                "application":
                    "Use the official department/policy route shown in the source data.",

                "charges":
                    "Not specified in this dataset.",

                "best_for":
                    clean(
                        r[
                            "sector"
                        ]
                    ),

                "source":
                    "",

                "score":
                    1
            })


    results.sort(
        key=lambda x:
            (
                -x[
                    "score"
                ],
                x[
                    "name"
                ]
            )
    )

    unique = []
    seen = set()

    for item in results:

        if item[
            "name"
        ] not in seen:

            seen.add(
                item[
                    "name"
                ]
            )

            unique.append(
                item
            )

        if len(unique) >= 6:

            break

    return unique


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PROFILE PAGE
# ============================================================

@app.get("/profile")
def profile():

    return render_template(
        "profile.html"
    )


# ============================================================
# GENERATE PLAN
# ============================================================

@app.post("/generate-plan")
def generate_plan():

    data = request.form.to_dict()

    # Send the profile to the existing advice API logic
    query = industry_query(
        data
    )

    industry = identify_industry(
        query
    )

    retrieved = retrieve(
        query,
        10
    )

    selected = select_approvals(
        data,
        industry
    )

    documents = documents_for_approvals(
        selected
    )

    scheme_results = select_schemes(
        data,
        industry
    )

    total = len(
        selected
    )

    result = {

        "profile":
            data,

        "industry": {

            "industry_id":
                clean(
                    industry[
                        "industry_id"
                    ]
                )
                if industry is not None
                else "",

            "industry_name":
                clean(
                    industry[
                        "industry_name"
                    ]
                )
                if industry is not None
                else "Business / Industry",

            "sector":
                clean(
                    industry[
                        "sector"
                    ]
                )
                if industry is not None
                else "General",

            "nic_code":
                clean(
                    industry[
                        "nic_code"
                    ]
                )
                if industry is not None
                else "",

            "mpcb_category":
                clean(
                    industry[
                        "mpcb_category"
                    ]
                )
                if industry is not None
                else ""
        },

        "approvals":
            selected,

        "documents":
            documents,

        "schemes":
            scheme_results,

        "dashboard": {

            "total_approvals":
                total,

            "completed":
                0,

            "pending":
                total,

            "percentage":
                0,

            "documents":
                len(
                    documents
                ),

            "schemes":
                len(
                    scheme_results
                )
        },

        "timeline": [

            {
                "step":
                    "Business profile",

                "status":
                    "Completed"
            },

            {
                "step":
                    "Approval checklist",

                "status":
                    "Ready"
            },

            {
                "step":
                    "Documents",

                "status":
                    "Pending"
            },

            {
                "step":
                    "Scheme applications",

                "status":
                    "Pending"
            },

            {
                "step":
                    "Application tracking",

                "status":
                    "Ready"
            }
        ],

        "retrieval": [

            {
                "title":
                    r.get(
                        "title",
                        ""
                    ),

                "type":
                    r.get(
                        "record_type",
                        ""
                    ),

                "score":
                    r.get(
                        "score",
                        0
                    )
            }

            for r in retrieved[:6]
        ],

        "notice":
            "Prototype information assistant. Fees and timelines must be verified against authoritative current department sources before real submission."
    }


    # Store the result temporarily in the Flask session
    # only if needed later.
    # For now render the result page directly.

    return render_template(
        "plan.html",
        result=result
    )


# ============================================================
# EXISTING API
# ============================================================

@app.post("/api/advice")
def advice():

    data = (
        request.get_json(
            silent=True
        )
        or request.form.to_dict()
    )

    profile_data = {
        k: clean(v)
        for k, v in data.items()
    }

    query = industry_query(
        profile_data
    )

    industry = identify_industry(
        query
    )

    retrieved = retrieve(
        query,
        10
    )

    selected = select_approvals(
        profile_data,
        industry
    )

    documents = documents_for_approvals(
        selected
    )

    scheme_results = select_schemes(
        profile_data,
        industry
    )

    total = len(
        selected
    )

    return jsonify({

        "profile":
            profile_data,

        "industry": {

            "industry_id":
                clean(
                    industry[
                        "industry_id"
                    ]
                )
                if industry is not None
                else "",

            "industry_name":
                clean(
                    industry[
                        "industry_name"
                    ]
                )
                if industry is not None
                else "Business / Industry",

            "sector":
                clean(
                    industry[
                        "sector"
                    ]
                )
                if industry is not None
                else "General",

            "nic_code":
                clean(
                    industry[
                        "nic_code"
                    ]
                )
                if industry is not None
                else "",

            "mpcb_category":
                clean(
                    industry[
                        "mpcb_category"
                    ]
                )
                if industry is not None
                else ""
        },

        "approvals":
            selected,

        "documents":
            documents,

        "schemes":
            scheme_results,

        "dashboard": {

            "total_approvals":
                total,

            "completed":
                0,

            "pending":
                total,

            "percentage":
                0,

            "documents":
                len(
                    documents
                ),

            "schemes":
                len(
                    scheme_results
                )
        },

        "timeline": [

            {
                "step":
                    "Business profile",
                "status":
                    "Completed"
            },

            {
                "step":
                    "Approval checklist",
                "status":
                    "Ready"
            },

            {
                "step":
                    "Documents",
                "status":
                    "Pending"
            },

            {
                "step":
                    "Scheme applications",
                "status":
                    "Pending"
            },

            {
                "step":
                    "Application tracking",
                "status":
                    "Ready"
            }
        ],

        "retrieval": [

            {
                "title":
                    r.get(
                        "title",
                        ""
                    ),

                "type":
                    r.get(
                        "record_type",
                        ""
                    ),

                "score":
                    r.get(
                        "score",
                        0
                    )
            }

            for r in retrieved[:6]
        ],

        "notice":
            "Prototype information assistant. Fees and timelines must be verified against authoritative current department sources."
    })


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return jsonify({

        "status":
            "ok",

        "model":
            bundle.get(
                "model_type"
            ),

        "knowledge_records":
            len(
                knowledge_records
            ),

        "authentication":
            False,

        "otp":
            False
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=True
    )