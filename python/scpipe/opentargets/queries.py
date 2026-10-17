"""GraphQL query strings for the Open Targets Platform API (v4).

Kept apart from the client so the recorded test fixtures and the cache keys
(SHA-256 of query + variables) are easy to reason about: changing a query
string here changes every cache key that uses it.

Field names follow the public schema at
https://api.platform.opentargets.org/api/v4/graphql/schema (search, target,
tractability, knownDrugs, associatedDiseases). They could not be checked
against the live endpoint from the build workspace (no network), so if the
schema has moved the error surfaces as ``OpenTargetsError`` with the server's
message and only this file needs changing.
"""

from __future__ import annotations

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

# Resolve a gene symbol to a target (Ensembl ID). Several hits come back; the
# client picks the one whose approvedSymbol matches the query (case-insensitive),
# else the top hit.
TARGET_SEARCH = """
query TargetSearch($text: String!) {
  search(queryString: $text, entityNames: ["target"], page: {index: 0, size: 5}) {
    total
    hits {
      id
      name
      entity
      score
      object {
        ... on Target {
          id
          approvedSymbol
          approvedName
        }
      }
    }
  }
}
""".strip()

# Evidence for one target: tractability, known drugs (name, phase, disease,
# mechanism) and the ten strongest disease associations.
TARGET_EVIDENCE = """
query TargetEvidence($id: String!, $drugs: Int!, $diseases: Int!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    biotype
    tractability {
      modality
      id
      value
    }
    knownDrugs(size: $drugs) {
      count
      rows {
        drug {
          id
          name
        }
        phase
        status
        mechanismOfAction
        disease {
          id
          name
        }
      }
    }
    associatedDiseases(page: {index: 0, size: $diseases}) {
      count
      rows {
        score
        disease {
          id
          name
        }
      }
    }
  }
}
""".strip()

# Free-text search for a disease, target or drug (G2 uses it to resolve EFO
# and UBERON IDs; the S4 summary does not).
ENTITY_SEARCH = """
query EntitySearch($text: String!, $entities: [String!]!, $size: Int!) {
  search(queryString: $text, entityNames: $entities, page: {index: 0, size: $size}) {
    total
    hits {
      id
      name
      entity
      score
      description
      object {
        ... on Disease {
          id
          name
        }
        ... on Target {
          id
          approvedSymbol
        }
        ... on Drug {
          id
          name
        }
      }
    }
  }
}
""".strip()
