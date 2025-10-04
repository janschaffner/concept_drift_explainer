# Purpose of this Directory

This directory serves as the **master archive** for the synthetically generated context document corpus used for evaluating the CDE.

All documents in `uploaded` were uploaded to Pinecone and must be copied into `frontend/static/documents`.
Documents in `to_be_uploaded` were not upserted to Pinecone.

## Contents

- **Gold Documents**: A set of documents, where each is the known correct explanation for one of the specific concept drifts.
- **Noise Documents**: A set of documents that are thematically related to the business process but are *not* the correct explanation for any drift.